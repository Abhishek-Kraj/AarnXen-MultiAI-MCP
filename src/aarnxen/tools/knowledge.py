"""Knowledge base tools — store, search, and manage persistent knowledge.

The ultimate memory system for AarnXen MCP. Supports 3-layer search
(index → timeline → full details) for 10x token savings, observation
types, session tracking, and optional semantic vector search.
"""

from mcp.server.fastmcp import Context


async def kb_store_handler(
    title: str,
    content: str,
    doc_type: str = "note",
    tags: str = "",
    source: str = "",
    ctx: Context = None,
) -> dict:
    """Store a document/note/fact in the knowledge base for future reference.

    Args:
        title: Short title for the document.
        content: Full content to store.
        doc_type: Type — "note", "fact", "code", "decision", "reference".
        tags: Comma-separated tags for categorization.
        source: Where this came from (URL, file path, etc.).
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    doc_id = kb.store_document(title, content, doc_type, tags, source)
    try:
        extractor = deps.extractor
        if extractor:
            extractor.extract_and_store(f"{title} {content}")
    except Exception:
        pass
    return {"id": doc_id, "title": title, "type": doc_type, "stored": True}


async def kb_search_handler(
    query: str,
    limit: int = 5,
    ctx: Context = None,
) -> dict:
    """Search the knowledge base for relevant documents/notes/facts.

    Args:
        query: Search query (supports natural language).
        limit: Max results to return.
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    results = kb.search_hybrid(query, limit)
    return {"query": query, "results": results, "count": len(results)}


async def kb_get_handler(
    doc_id: str,
    ctx: Context = None,
) -> dict:
    """Retrieve a specific document by ID.

    Args:
        doc_id: Document ID from search results.
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    doc = kb.get_document(doc_id)
    if not doc:
        return {"error": f"Document '{doc_id}' not found"}
    return doc


async def kb_remember_handler(
    entity: str,
    fact: str,
    entity_type: str = "concept",
    obs_type: str = "general",
    session_id: str = "",
    ctx: Context = None,
) -> dict:
    """Remember a fact about an entity (person, project, concept, tool).

    Args:
        entity: Name of the entity (e.g., "Python", "AuthService", "Abhishek").
        fact: The fact/observation to remember.
        entity_type: Type — "person", "project", "concept", "tool", "decision".
        obs_type: Observation type — "general", "bugfix", "feature", "decision", "discovery", "refactor".
        session_id: Session identifier for cross-session tracking.
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    kb.add_entity(entity, entity_type)
    obs_id = kb.add_observation(entity, fact, obs_type=obs_type, session_id=session_id)
    try:
        extractor = deps.extractor
        if extractor:
            extractor.extract_and_store(fact)
    except Exception:
        pass
    return {"entity": entity, "fact": fact, "observation_id": obs_id, "remembered": True}


async def kb_recall_handler(
    query: str,
    ctx: Context = None,
) -> dict:
    """Recall everything known about an entity or topic.

    Args:
        query: Entity name or topic to recall.
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    entities = kb.search_entities(query)
    docs = kb.search_documents(query, limit=5)
    return {
        "query": query,
        "entities": entities,
        "related_documents": docs,
    }


async def kb_relate_handler(
    from_entity: str,
    to_entity: str,
    relation: str,
    ctx: Context = None,
) -> dict:
    """Create a relationship between two entities.

    Args:
        from_entity: Source entity name.
        to_entity: Target entity name.
        relation: Relationship type (e.g., "uses", "depends_on", "created_by", "part_of").
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    rel_id = kb.add_relation(from_entity, to_entity, relation)
    return {"from": from_entity, "to": to_entity, "relation": relation, "id": rel_id}


async def kb_stats_handler(ctx: Context = None) -> dict:
    """Show knowledge base statistics."""
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}
    return kb.stats()


# --- 3-Layer Search Handlers ---


async def kb_search_index_handler(
    query: str,
    limit: int = 20,
    obs_type: str = "",
    session_id: str = "",
    ctx: Context = None,
) -> dict:
    """Step 1: Search observations index. Returns lightweight results with IDs.

    Args:
        query: Search query.
        limit: Max results (default 20).
        obs_type: Filter by type (e.g., "bugfix", "decision", "feature").
        session_id: Filter by session ID.
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    results = kb.search_observations_index(
        query, limit, obs_type=obs_type or None, session_id=session_id or None,
    )
    return {"query": query, "results": results, "count": len(results)}


async def kb_timeline_handler(
    anchor: str = "",
    query: str = "",
    depth_before: int = 5,
    depth_after: int = 5,
    ctx: Context = None,
) -> dict:
    """Step 2: Get chronological context around an observation.

    Args:
        anchor: Observation ID to center on.
        query: Alternative — finds anchor by content match.
        depth_before: Number of observations before anchor.
        depth_after: Number of observations after anchor.
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    results = kb.timeline(
        anchor_id=anchor or None, query=query or None,
        depth_before=depth_before, depth_after=depth_after,
    )
    return {"results": results, "count": len(results)}


async def kb_get_observations_handler(
    ids: str,
    ctx: Context = None,
) -> dict:
    """Step 3: Fetch full details for specific observation IDs.

    Args:
        ids: Comma-separated observation IDs.
    """
    deps = ctx.request_context.lifespan_context
    kb = deps.knowledge
    if not kb:
        return {"error": "Knowledge base is disabled in config"}

    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    results = kb.get_observations_by_ids(id_list)
    return {"results": results, "count": len(results)}
