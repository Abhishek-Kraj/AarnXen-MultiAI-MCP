"""Tests for knowledge base."""

import tempfile

import pytest

from aarnxen.core.knowledge import KnowledgeBase


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
