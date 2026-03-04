"""AarnXen Multi-AI MCP Server — entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from mcp.server.fastmcp import Context, FastMCP

from aarnxen.config import AarnXenConfig, load_config
from aarnxen.core.cache import ResponseCache
from aarnxen.core.circuit_breaker import CircuitBreaker
from aarnxen.core.conversation import ConversationMemory
from aarnxen.core.cost import CostTracker
from aarnxen.core.extractor import EntityExtractor
from aarnxen.core.knowledge import KnowledgeBase
from aarnxen.providers.registry import ProviderRegistry

logger = logging.getLogger("aarnxen")


@dataclass
class AppContext:
    registry: ProviderRegistry
    cache: Optional[ResponseCache]
    cost_tracker: CostTracker
    memory: Optional[ConversationMemory]
    knowledge: Optional[KnowledgeBase]
    circuit_breaker: CircuitBreaker
    extractor: EntityExtractor
    config: AarnXenConfig


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    cfg = load_config()

    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
    logger.info("AarnXen Multi-AI MCP Server starting...")

    registry = ProviderRegistry.from_config(cfg)

    cache = None
    if cfg.cache.enabled:
        cache = ResponseCache(max_size=cfg.cache.max_size, ttl_seconds=cfg.cache.ttl_seconds)

    cost_tracker = CostTracker()

    memory = None
    if cfg.memory.enabled:
        memory = ConversationMemory(persist_path=cfg.memory.path)

    knowledge = KnowledgeBase()
    circuit_breaker = CircuitBreaker()
    extractor = EntityExtractor(knowledge_base=knowledge)

    try:
        yield AppContext(
            registry=registry,
            cache=cache,
            cost_tracker=cost_tracker,
            memory=memory,
            knowledge=knowledge,
            circuit_breaker=circuit_breaker,
            extractor=extractor,
            config=cfg,
        )
    finally:
        if memory:
            memory.close()
        knowledge.close()
        logger.info("AarnXen shut down.")


mcp = FastMCP("AarnXen-MultiAI", lifespan=app_lifespan)


# --- Tool registration ---

from aarnxen.tools.chat import chat_handler  # noqa: E402
from aarnxen.tools.codereview import codereview_handler  # noqa: E402
from aarnxen.tools.compare import compare_handler  # noqa: E402
from aarnxen.tools.consensus import consensus_handler  # noqa: E402
from aarnxen.tools.costs import costs_handler  # noqa: E402
from aarnxen.tools.knowledge import (  # noqa: E402
    kb_get_handler,
    kb_recall_handler,
    kb_relate_handler,
    kb_remember_handler,
    kb_search_handler,
    kb_stats_handler,
    kb_store_handler,
)
from aarnxen.tools.think import think_handler  # noqa: E402


@mcp.tool()
async def chat(
    prompt: str,
    model: str = "auto",
    temperature: float = 0.7,
    system_prompt: str = "",
    conversation_id: str = "",
    cascade: bool = False,
    ctx: Context = None,
) -> dict:
    """Chat with any AI model. Set cascade=True for smart routing (tries cheap model first, escalates if needed)."""
    return await chat_handler(prompt, model, temperature, system_prompt, conversation_id, cascade, ctx)


@mcp.tool()
async def consensus(
    prompt: str,
    models: str = "auto",
    temperature: float = 0.7,
    system_prompt: str = "",
    ctx: Context = None,
) -> dict:
    """Query multiple AI models in parallel and synthesize responses."""
    return await consensus_handler(prompt, models, temperature, system_prompt, ctx)


@mcp.tool()
async def compare(
    prompt: str,
    model_a: str = "",
    model_b: str = "",
    temperature: float = 0.7,
    system_prompt: str = "",
    ctx: Context = None,
) -> dict:
    """Compare two AI models side-by-side."""
    return await compare_handler(prompt, model_a, model_b, temperature, system_prompt, ctx)


@mcp.tool()
async def think(
    prompt: str,
    model: str = "auto",
    depth: str = "medium",
    ctx: Context = None,
) -> dict:
    """Deep reasoning — step-by-step analysis of complex problems."""
    return await think_handler(prompt, model, depth, ctx)


@mcp.tool()
async def codereview(
    code: str,
    language: str = "auto",
    model: str = "auto",
    focus: str = "general",
    ctx: Context = None,
) -> dict:
    """Review code for bugs, style, performance, and security."""
    return await codereview_handler(code, language, model, focus, ctx)


@mcp.tool()
async def costs(ctx: Context = None) -> dict:
    """Show session cost summary and cache stats."""
    return await costs_handler(ctx)


@mcp.tool()
async def list_models(ctx: Context = None) -> dict:
    """List all available models across all providers."""
    deps = ctx.request_context.lifespan_context
    return {"models": deps.registry.list_all_models()}


@mcp.tool()
async def provider_health(ctx: Context = None) -> dict:
    """Show circuit breaker health status for all AI providers."""
    deps = ctx.request_context.lifespan_context
    return deps.circuit_breaker.get_all_status()


@mcp.tool()
async def kb_consolidate(ctx: Context = None) -> dict:
    """Run memory maintenance: deduplicate, decay old relations, prune stale data."""
    deps = ctx.request_context.lifespan_context
    return deps.knowledge.consolidate()


# --- Knowledge Base Tools ---


@mcp.tool()
async def kb_store(
    title: str,
    content: str,
    doc_type: str = "note",
    tags: str = "",
    source: str = "",
    ctx: Context = None,
) -> dict:
    """Store a document/note/fact in the knowledge base."""
    return await kb_store_handler(title, content, doc_type, tags, source, ctx)


@mcp.tool()
async def kb_search(query: str, limit: int = 5, ctx: Context = None) -> dict:
    """Search the knowledge base."""
    return await kb_search_handler(query, limit, ctx)


@mcp.tool()
async def kb_get(doc_id: str, ctx: Context = None) -> dict:
    """Retrieve a specific document by ID."""
    return await kb_get_handler(doc_id, ctx)


@mcp.tool()
async def kb_remember(
    entity: str,
    fact: str,
    entity_type: str = "concept",
    ctx: Context = None,
) -> dict:
    """Remember a fact about an entity (person, project, concept)."""
    return await kb_remember_handler(entity, fact, entity_type, ctx)


@mcp.tool()
async def kb_recall(query: str, ctx: Context = None) -> dict:
    """Recall everything known about an entity or topic."""
    return await kb_recall_handler(query, ctx)


@mcp.tool()
async def kb_relate(
    from_entity: str,
    to_entity: str,
    relation: str,
    ctx: Context = None,
) -> dict:
    """Create a relationship between two entities."""
    return await kb_relate_handler(from_entity, to_entity, relation, ctx)


@mcp.tool()
async def kb_stats(ctx: Context = None) -> dict:
    """Show knowledge base statistics."""
    return await kb_stats_handler(ctx)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
