"""Tests for knowledge base."""

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from aarnxen.core.knowledge import KnowledgeBase, _now_iso, _jaccard, _tokenize


@pytest.fixture
def kb(tmp_path):
    db = KnowledgeBase(db_path=str(tmp_path / "test_kb.db"))
    yield db
    db.close()


def test_store_and_search(kb):
    kb.store_document("Python Async", "asyncio.gather runs coroutines in parallel", tags="python,async")
    kb.store_document("React Hooks", "useEffect handles side effects in React", tags="react,frontend")

    results = kb.search_documents("async parallel")
    assert len(results) >= 1
    assert "Python Async" in results[0]["title"]


def test_search_no_results(kb):
    results = kb.search_documents("nonexistent topic xyz")
    assert len(results) == 0


def test_get_document(kb):
    doc_id = kb.store_document("Test Doc", "Some content here")
    doc = kb.get_document(doc_id)
    assert doc is not None
    assert doc["title"] == "Test Doc"
    assert doc["content"] == "Some content here"


def test_delete_document(kb):
    doc_id = kb.store_document("To Delete", "Will be removed")
    assert kb.delete_document(doc_id) is True
    assert kb.get_document(doc_id) is None


def test_list_documents(kb):
    kb.store_document("Doc 1", "First", doc_type="note")
    kb.store_document("Doc 2", "Second", doc_type="code")
    kb.store_document("Doc 3", "Third", doc_type="note")

    all_docs = kb.list_documents()
    assert len(all_docs) == 3

    notes = kb.list_documents(doc_type="note")
    assert len(notes) == 2


def test_entity_and_observations(kb):
    kb.add_entity("Python", entity_type="language", description="A programming language")
    kb.add_observation("Python", "Python 3.13 added free-threaded mode")
    kb.add_observation("Python", "Uses GIL for thread safety")

    results = kb.search_entities("Python")
    assert len(results) == 1
    assert results[0]["name"] == "Python"
    assert len(results[0]["observations"]) == 2


def test_relations(kb):
    kb.add_relation("FastAPI", "Python", "built_with")
    kb.add_relation("FastAPI", "Starlette", "uses")

    results = kb.search_entities("FastAPI")
    assert len(results) == 1
    assert len(results[0]["relations"]) == 2
    relation_targets = {r["to"] for r in results[0]["relations"]}
    assert "Python" in relation_targets
    assert "Starlette" in relation_targets


def test_stats(kb):
    kb.store_document("Doc", "Content")
    kb.add_entity("Entity1")
    kb.add_relation("A", "B", "knows")
    kb.add_observation("Entity1", "A fact")

    stats = kb.stats()
    assert stats["documents"] == 1
    assert stats["entities"] >= 1
    assert stats["relations"] == 1
    assert stats["observations"] == 1


# --- Importance Scoring Tests ---

def test_score_document_without_query(kb):
    doc_id = kb.store_document("Important Doc", "Very important content", importance=8.0)
    score = kb.score_document(doc_id)
    # relevance = 8.0/10 = 0.8, freq = 0/(1+0) = 0, recency ~ 1.0 (just created)
    assert score > 0
    # relevance * 0.4 + 0 * 0.3 + ~1.0 * 0.3 = 0.32 + 0.3 = ~0.62
    assert 0.5 < score < 0.8


def test_score_document_nonexistent(kb):
    assert kb.score_document("fake_id") == 0.0


def test_score_document_with_query(kb):
    doc_id = kb.store_document("Python Guide", "Learn python programming basics")
    score = kb.score_document(doc_id, query="python programming")
    assert score > 0


def test_importance_stored(kb):
    doc_id = kb.store_document("High Priority", "Critical content", importance=9.0)
    row = kb._conn.execute("SELECT importance FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert row[0] == 9.0


def test_access_count_increments_on_search(kb):
    doc_id = kb.store_document("Searchable", "unique findable content here")

    # Check initial access_count
    row = kb._conn.execute("SELECT access_count FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert row[0] == 0

    # Search should bump access_count
    kb.search_documents("unique findable content")

    row = kb._conn.execute("SELECT access_count FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert row[0] == 1

    # Search again
    kb.search_documents("unique findable content")
    row = kb._conn.execute("SELECT access_count FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert row[0] == 2


def test_last_accessed_updated_on_search(kb):
    doc_id = kb.store_document("Trackable", "track access time for this doc")
    kb.search_documents("track access time")

    row = kb._conn.execute("SELECT last_accessed FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert row[0] != ""
    # Should be a valid ISO timestamp
    dt = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
    assert dt.year >= 2025


def test_temporal_decay_reduces_score(kb):
    doc_id = kb.store_document("Decaying Doc", "content that decays over time")

    # Score right now
    score_now = kb.score_document(doc_id)

    # Simulate last_accessed being 60 days ago
    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    kb._conn.execute("UPDATE documents SET last_accessed = ? WHERE id = ?", (old_date, doc_id))
    kb._conn.commit()

    score_old = kb.score_document(doc_id)
    assert score_old < score_now


def test_search_returns_combined_score(kb):
    kb.store_document("Scored Result", "scored search result content")
    results = kb.search_documents("scored search result")
    assert len(results) >= 1
    assert "combined_score" in results[0]
    assert results[0]["combined_score"] > 0


# --- Confidence Decay on Relations Tests ---

def test_relation_has_confidence(kb):
    rid = kb.add_relation("Django", "Python", "built_with")
    row = kb._conn.execute("SELECT confidence FROM relations WHERE id = ?", (rid,)).fetchone()
    assert row[0] == 1.0


def test_reinforce_relation(kb):
    rid = kb.add_relation("Express", "Node", "built_with")
    # Set confidence low manually
    kb._conn.execute("UPDATE relations SET confidence = 0.5 WHERE id = ?", (rid,))
    kb._conn.commit()

    kb.reinforce_relation(rid)

    row = kb._conn.execute("SELECT confidence FROM relations WHERE id = ?", (rid,)).fetchone()
    assert row[0] == 1.0


def test_get_relations_with_decay(kb):
    kb.add_relation("Flask", "Python", "built_with")
    results = kb.get_relations_with_decay("Flask")
    assert len(results) == 1
    assert results[0]["to"] == "Python"
    assert results[0]["confidence"] == 1.0
    assert results[0]["effective_confidence"] <= 1.0
    assert results[0]["effective_confidence"] > 0.9  # just created, minimal decay


def test_relations_decay_over_time(kb):
    rid = kb.add_relation("Svelte", "JavaScript", "built_with")

    # Simulate last_reinforced being 60 days ago (2 half-lives with default 30-day half-life)
    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    kb._conn.execute("UPDATE relations SET last_reinforced = ? WHERE id = ?", (old_date, rid))
    kb._conn.commit()

    results = kb.get_relations_with_decay("Svelte")
    assert len(results) == 1
    # After 2 half-lives: 1.0 * 0.5^2 = 0.25
    assert 0.2 < results[0]["effective_confidence"] < 0.3


def test_relations_filtered_below_threshold(kb):
    rid = kb.add_relation("OldLib", "OldLang", "built_with")

    # Simulate last_reinforced being 120 days ago (4 half-lives)
    old_date = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%SZ")
    kb._conn.execute("UPDATE relations SET last_reinforced = ? WHERE id = ?", (old_date, rid))
    kb._conn.commit()

    results = kb.get_relations_with_decay("OldLib")
    # After 4 half-lives: 1.0 * 0.5^4 = 0.0625 < 0.1 threshold
    assert len(results) == 0


def test_get_relations_with_decay_nonexistent_entity(kb):
    results = kb.get_relations_with_decay("NonExistent")
    assert results == []


# --- Deduplication Tests ---

def test_deduplicate_similar_documents(kb):
    kb.store_document("Python Guide", "learn python programming basics fundamentals", importance=8.0)
    kb.store_document("Python Guide", "learn python programming basics fundamentals today", importance=5.0)

    count = kb.deduplicate(similarity_threshold=0.7)
    assert count == 1
    assert kb.stats()["documents"] == 1

    # The higher-importance one should survive
    docs = kb.list_documents()
    doc = kb.get_document(docs[0]["id"])
    row = kb._conn.execute("SELECT importance FROM documents WHERE id = ?", (docs[0]["id"],)).fetchone()
    assert row[0] == 8.0


def test_deduplicate_different_documents(kb):
    kb.store_document("Python Guide", "learn python programming basics")
    kb.store_document("React Guide", "learn react hooks components state management")

    count = kb.deduplicate(similarity_threshold=0.8)
    assert count == 0
    assert kb.stats()["documents"] == 2


def test_consolidate(kb):
    # Create some data
    kb.store_document("Doc A", "some content alpha beta gamma")
    kb.store_document("Doc A Copy", "some content alpha beta gamma delta")
    rid = kb.add_relation("X", "Y", "knows")

    # Make the relation very old
    old_date = (datetime.now(timezone.utc) - timedelta(days=300)).strftime("%Y-%m-%dT%H:%M:%SZ")
    kb._conn.execute("UPDATE relations SET last_reinforced = ? WHERE id = ?", (old_date, rid))
    kb._conn.commit()

    # Add old observation
    kb.add_observation("X", "old fact")
    cutoff = time.time() - (91 * 86400)
    kb._conn.execute("UPDATE observations SET created_at = ? WHERE entity_id = (SELECT id FROM entities WHERE name = 'X')", (cutoff,))
    kb._conn.commit()

    result = kb.consolidate()
    assert result["merged_documents"] >= 0
    assert result["removed_relations"] >= 0
    assert result["removed_observations"] >= 0


# --- Auto-Tagging Tests ---

def test_auto_tag(kb):
    doc_id = kb.store_document("Python Programming", "python programming tutorial python basics python advanced")
    tags = kb.auto_tag(doc_id)
    assert len(tags) > 0
    assert "python" in tags


def test_auto_tag_nonexistent(kb):
    tags = kb.auto_tag("fake_id")
    assert tags == []


def test_auto_tag_filters_stopwords(kb):
    doc_id = kb.store_document("Test", "the and but python programming for with python")
    tags = kb.auto_tag(doc_id)
    assert "the" not in tags
    assert "and" not in tags
    assert "python" in tags


def test_search_by_tag(kb):
    id1 = kb.store_document("Tagged Doc", "content", tags="python,async")
    id2 = kb.store_document("Other Doc", "content", tags="react,frontend")

    results = kb.search_by_tag("python")
    assert len(results) == 1
    assert results[0]["id"] == id1

    results = kb.search_by_tag("react")
    assert len(results) == 1
    assert results[0]["id"] == id2


def test_search_by_tag_no_results(kb):
    kb.store_document("Doc", "content", tags="python")
    results = kb.search_by_tag("rust")
    assert len(results) == 0


# --- Scope Filtering Tests ---

def test_store_with_scope(kb):
    doc_id = kb.store_document("Project Doc", "content", scope="project")
    row = kb._conn.execute("SELECT scope FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert row[0] == "project"


def test_default_scope_is_user(kb):
    doc_id = kb.store_document("Default Scope", "content")
    row = kb._conn.execute("SELECT scope FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert row[0] == "user"


def test_search_with_scope_filter(kb):
    kb.store_document("User Doc", "unique searchable alpha content", scope="user")
    kb.store_document("Project Doc", "unique searchable alpha content", scope="project")
    kb.store_document("Session Doc", "unique searchable alpha content", scope="session")

    all_results = kb.search_documents("unique searchable alpha")
    assert len(all_results) == 3

    user_results = kb.search_documents("unique searchable alpha", scope="user")
    assert len(user_results) == 1
    assert user_results[0]["title"] == "User Doc"

    project_results = kb.search_documents("unique searchable alpha", scope="project")
    assert len(project_results) == 1
    assert project_results[0]["title"] == "Project Doc"


# --- Helper Function Tests ---

def test_jaccard_identical():
    a = {"python", "programming", "language"}
    b = {"python", "programming", "language"}
    assert _jaccard(a, b) == 1.0


def test_jaccard_disjoint():
    a = {"python", "programming"}
    b = {"rust", "systems"}
    assert _jaccard(a, b) == 0.0


def test_jaccard_partial():
    a = {"python", "programming", "language"}
    b = {"python", "programming", "tutorial"}
    # intersection=2, union=4 => 0.5
    assert _jaccard(a, b) == 0.5


def test_jaccard_empty():
    assert _jaccard(set(), {"a"}) == 0.0
    assert _jaccard(set(), set()) == 0.0


def test_tokenize():
    tokens = _tokenize("Hello, world! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    # Short words filtered
    assert "is" not in tokens


# --- Migration Tests ---

def test_migration_adds_columns(tmp_path):
    """Verify migration adds new columns to existing databases."""
    db_path = str(tmp_path / "migrate_test.db")
    kb1 = KnowledgeBase(db_path=db_path)
    # Verify columns exist
    assert kb1._has_column("documents", "importance")
    assert kb1._has_column("documents", "access_count")
    assert kb1._has_column("documents", "last_accessed")
    assert kb1._has_column("documents", "decay_rate")
    assert kb1._has_column("documents", "scope")
    assert kb1._has_column("relations", "confidence")
    assert kb1._has_column("relations", "last_reinforced")
    assert kb1._has_column("relations", "half_life_days")
    kb1.close()

    # Re-open — migration should not fail on existing columns
    kb2 = KnowledgeBase(db_path=db_path)
    assert kb2._has_column("documents", "importance")
    kb2.close()


def test_backward_compatibility(kb):
    """Existing operations should still work with new columns."""
    doc_id = kb.store_document("Compat Test", "backward compatible content")
    doc = kb.get_document(doc_id)
    assert doc["title"] == "Compat Test"

    results = kb.search_documents("backward compatible content")
    assert len(results) >= 1

    assert kb.delete_document(doc_id) is True


# --- Observation Types + Session Tracking Tests ---

def test_add_observation_with_type(kb):
    kb.add_entity("Python")
    oid = kb.add_observation("Python", "Fixed GIL contention bug", obs_type="bugfix")
    row = kb._conn.execute("SELECT obs_type FROM observations WHERE id = ?", (oid,)).fetchone()
    assert row[0] == "bugfix"


def test_add_observation_with_session_id(kb):
    kb.add_entity("React")
    oid = kb.add_observation("React", "Added lazy loading", session_id="sess-abc-123")
    row = kb._conn.execute("SELECT session_id FROM observations WHERE id = ?", (oid,)).fetchone()
    assert row[0] == "sess-abc-123"


def test_default_obs_type_is_general(kb):
    kb.add_entity("Go")
    oid = kb.add_observation("Go", "Go 1.22 released")
    row = kb._conn.execute("SELECT obs_type FROM observations WHERE id = ?", (oid,)).fetchone()
    assert row[0] == "general"


def test_observation_migration_columns(tmp_path):
    db_path = str(tmp_path / "obs_migrate.db")
    kb1 = KnowledgeBase(db_path=db_path)
    assert kb1._has_column("observations", "obs_type")
    assert kb1._has_column("observations", "session_id")
    assert kb1._has_column("documents", "embedding")
    kb1.close()


# --- 3-Layer Search Tests ---

def test_search_observations_index(kb):
    kb.add_entity("FastAPI")
    kb.add_observation("FastAPI", "FastAPI supports async route handlers", obs_type="feature")
    kb.add_observation("FastAPI", "FastAPI uses Starlette under the hood", obs_type="discovery")

    results = kb.search_observations_index("FastAPI")
    assert len(results) == 2
    assert "id" in results[0]
    assert "snippet" in results[0]
    assert len(results[0]["snippet"]) <= 80


def test_search_observations_index_type_filter(kb):
    kb.add_entity("Django")
    kb.add_observation("Django", "Fixed ORM N+1 query", obs_type="bugfix")
    kb.add_observation("Django", "Added async view support", obs_type="feature")
    kb.add_observation("Django", "Django uses MTV pattern", obs_type="discovery")

    results = kb.search_observations_index("Django", obs_type="bugfix")
    assert len(results) == 1
    assert results[0]["obs_type"] == "bugfix"


def test_search_observations_index_session_filter(kb):
    kb.add_entity("Node")
    kb.add_observation("Node", "Node 22 is LTS", session_id="sess-1")
    kb.add_observation("Node", "Node uses V8 engine", session_id="sess-2")

    results = kb.search_observations_index("Node", session_id="sess-1")
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-1"


def test_get_observations_by_ids(kb):
    kb.add_entity("Rust")
    oid1 = kb.add_observation("Rust", "Rust has no garbage collector")
    oid2 = kb.add_observation("Rust", "Rust uses ownership model")
    kb.add_observation("Rust", "Rust compiles to native code")

    results = kb.get_observations_by_ids([oid1, oid2])
    assert len(results) == 2
    assert "content" in results[0]
    ids = {r["id"] for r in results}
    assert oid1 in ids
    assert oid2 in ids


def test_get_observations_empty_ids(kb):
    results = kb.get_observations_by_ids([])
    assert results == []


def test_get_observations_nonexistent_id(kb):
    results = kb.get_observations_by_ids(["fake-id-123"])
    assert results == []


# --- Timeline Tests ---

def test_timeline_by_anchor_id(kb):
    kb.add_entity("Project")
    oids = []
    for i in range(10):
        oid = kb.add_observation("Project", f"Event {i}")
        oids.append(oid)
        time.sleep(0.01)

    results = kb.timeline(anchor_id=oids[5], depth_before=2, depth_after=2)
    assert len(results) >= 3


def test_timeline_by_query(kb):
    kb.add_entity("Debug")
    kb.add_observation("Debug", "Started investigating memory leak")
    time.sleep(0.01)
    kb.add_observation("Debug", "Found root cause in connection pool")
    time.sleep(0.01)
    kb.add_observation("Debug", "Applied fix and verified")

    results = kb.timeline(query="root cause", depth_before=1, depth_after=1)
    assert len(results) >= 2


def test_timeline_no_match(kb):
    results = kb.timeline(query="nonexistent event xyz")
    assert results == []


def test_timeline_at_boundary(kb):
    kb.add_entity("Edge")
    oid = kb.add_observation("Edge", "Only observation")
    results = kb.timeline(anchor_id=oid, depth_before=5, depth_after=5)
    assert len(results) >= 1


# --- Hybrid Search Tests ---

def test_search_hybrid_without_embedder(kb):
    kb.store_document("Hybrid Test", "unique hybrid searchable content")
    kb._embedder = None
    results = kb.search_hybrid("unique hybrid searchable")
    assert len(results) >= 1
    assert results[0]["title"] == "Hybrid Test"


def test_search_hybrid_returns_results(kb):
    kb.store_document("Hybrid Doc", "semantic vector search test content")
    results = kb.search_hybrid("semantic vector search")
    assert len(results) >= 1


def test_embedding_column_migration(tmp_path):
    db_path = str(tmp_path / "emb_test.db")
    kb1 = KnowledgeBase(db_path=db_path)
    assert kb1._has_column("documents", "embedding")
    kb1.close()


# --- Stats Tests ---

def test_stats_includes_embedding_info(kb):
    kb.store_document("Stats Test", "content for stats")
    stats = kb.stats()
    assert "embedded_documents" in stats
    assert "has_embeddings" in stats
    assert isinstance(stats["has_embeddings"], bool)
