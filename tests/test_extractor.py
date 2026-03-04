"""Tests for auto entity extraction."""

from unittest.mock import MagicMock, call

import pytest

from aarnxen.core.extractor import EntityExtractor


@pytest.fixture
def extractor():
    return EntityExtractor()


@pytest.fixture
def mock_kb():
    kb = MagicMock()
    kb.search_entities.return_value = []
    return kb


@pytest.fixture
def extractor_with_kb(mock_kb):
    return EntityExtractor(knowledge_base=mock_kb)


# --- Technology extraction ---

class TestTechnologyExtraction:
    def test_single_tech(self, extractor):
        result = extractor.extract("I'm building a FastAPI app")
        names = [e["name"] for e in result["entities"]]
        assert "FastAPI" in names

    def test_multiple_techs(self, extractor):
        result = extractor.extract("I'm building a FastAPI app with PostgreSQL")
        names = [e["name"] for e in result["entities"]]
        assert "FastAPI" in names
        assert "PostgreSQL" in names

    def test_case_insensitive(self, extractor):
        result = extractor.extract("i use python and react")
        names = [e["name"] for e in result["entities"]]
        assert "Python" in names
        assert "React" in names

    def test_case_insensitive_upper(self, extractor):
        result = extractor.extract("PYTHON is great")
        names = [e["name"] for e in result["entities"]]
        assert "Python" in names

    def test_normalized_names(self, extractor):
        result = extractor.extract("using postgresql and pytorch")
        names = [e["name"] for e in result["entities"]]
        assert "PostgreSQL" in names
        assert "PyTorch" in names

    def test_tech_type(self, extractor):
        result = extractor.extract("I like Python")
        techs = [e for e in result["entities"] if e["name"] == "Python"]
        assert techs[0]["type"] == "technology"

    def test_k8s_alias(self, extractor):
        result = extractor.extract("deploying to k8s")
        names = [e["name"] for e in result["entities"]]
        assert "Kubernetes" in names

    def test_multi_word_term(self, extractor):
        result = extractor.extract("using spring boot for the backend")
        names = [e["name"] for e in result["entities"]]
        assert "Spring Boot" in names

    def test_dotted_name(self, extractor):
        result = extractor.extract("I use Next.js for SSR")
        names = [e["name"] for e in result["entities"]]
        assert "Next.js" in names

    def test_no_duplicates(self, extractor):
        result = extractor.extract("Python with Python and python")
        python_entries = [e for e in result["entities"] if e["name"] == "Python"]
        assert len(python_entries) == 1


# --- People extraction ---

class TestPeopleExtraction:
    def test_by_pattern(self, extractor):
        result = extractor.extract("Code review by John")
        names = [e["name"] for e in result["entities"]]
        assert "John" in names

    def test_from_pattern(self, extractor):
        result = extractor.extract("Got feedback from Alice")
        names = [e["name"] for e in result["entities"]]
        assert "Alice" in names

    def test_mention(self, extractor):
        result = extractor.extract("cc @dave for review")
        names = [e["name"] for e in result["entities"]]
        assert "dave" in names

    def test_person_type(self, extractor):
        result = extractor.extract("review by Sarah")
        people = [e for e in result["entities"] if e["name"] == "Sarah"]
        assert people[0]["type"] == "person"

    def test_full_name(self, extractor):
        result = extractor.extract("written by John Smith")
        names = [e["name"] for e in result["entities"]]
        assert "John Smith" in names

    def test_no_stopwords_as_people(self, extractor):
        result = extractor.extract("by the project")
        names = [e["name"] for e in result["entities"]]
        people = [e for e in result["entities"] if e["type"] == "person"]
        assert len(people) == 0

    def test_no_tech_as_people(self, extractor):
        result = extractor.extract("made by React developers")
        people = [e for e in result["entities"] if e["type"] == "person"]
        # "React" should not appear as a person
        person_names = [p["name"] for p in people]
        assert "React" not in person_names


# --- Project extraction ---

class TestProjectExtraction:
    def test_github_url(self, extractor):
        result = extractor.extract("check out github.com/vercel/next.js for details")
        names = [e["name"] for e in result["entities"]]
        assert "vercel/next.js" in names

    def test_project_keyword(self, extractor):
        result = extractor.extract("the project myapp needs refactoring")
        names = [e["name"] for e in result["entities"]]
        assert "myapp" in names

    def test_repo_keyword(self, extractor):
        result = extractor.extract("clone the repo backend-api")
        names = [e["name"] for e in result["entities"]]
        assert "backend-api" in names

    def test_project_type(self, extractor):
        result = extractor.extract("the project dashboard is ready")
        projects = [e for e in result["entities"] if e["name"] == "dashboard"]
        assert projects[0]["type"] == "project"


# --- Relationship extraction ---

class TestRelationshipExtraction:
    def test_uses_pattern(self, extractor):
        result = extractor.extract("React uses TypeScript")
        rels = result["relations"]
        assert any(
            r["from"] == "React" and r["to"] == "TypeScript" and r["type"] == "uses"
            for r in rels
        )

    def test_with_pattern(self, extractor):
        result = extractor.extract("Using React with TypeScript")
        rels = result["relations"]
        assert any(
            r["from"] == "React" and r["to"] == "TypeScript" and r["type"] == "uses"
            for r in rels
        )

    def test_depends_on(self, extractor):
        result = extractor.extract("FastAPI depends on Starlette")
        rels = result["relations"]
        assert any(
            r["from"] == "FastAPI" and r["to"] == "Starlette" and r["type"] == "depends_on"
            for r in rels
        )

    def test_migrate_pattern(self, extractor):
        result = extractor.extract("migrate from MySQL to PostgreSQL")
        rels = result["relations"]
        assert any(
            r["from"] == "MySQL" and r["to"] == "PostgreSQL" and r["type"] == "migrated_to"
            for r in rels
        )

    def test_replaces_pattern(self, extractor):
        result = extractor.extract("Vite replaces Webpack")
        rels = result["relations"]
        assert any(
            r["from"] == "Vite" and r["to"] == "Webpack" and r["type"] == "replaces"
            for r in rels
        )

    def test_built_with(self, extractor):
        result = extractor.extract("building Django using Python")
        rels = result["relations"]
        assert any(
            r["from"] == "Django" and r["to"] == "Python" and r["type"] == "built_with"
            for r in rels
        )

    def test_no_self_relations(self, extractor):
        result = extractor.extract("Python uses Python")
        rels = result["relations"]
        self_rels = [r for r in rels if r["from"] == r["to"]]
        assert len(self_rels) == 0

    def test_no_relation_for_unknown_entities(self, extractor):
        result = extractor.extract("banana uses doorknob")
        assert len(result["relations"]) == 0


# --- Edge cases ---

class TestEdgeCases:
    def test_empty_text(self, extractor):
        result = extractor.extract("")
        assert result == {"entities": [], "relations": []}

    def test_very_short_text(self, extractor):
        result = extractor.extract("hi")
        assert result == {"entities": [], "relations": []}

    def test_none_text(self, extractor):
        result = extractor.extract("")
        assert result["entities"] == []

    def test_no_false_positives_common_words(self, extractor):
        result = extractor.extract("The quick brown fox jumps over the lazy dog")
        assert len(result["entities"]) == 0

    def test_multiple_entity_types(self, extractor):
        result = extractor.extract("Review by Alice on the project dashboard using React with TypeScript")
        types = {e["type"] for e in result["entities"]}
        assert "person" in types
        assert "technology" in types
        assert "project" in types

    def test_sentence_with_many_techs(self, extractor):
        result = extractor.extract(
            "We use Python, FastAPI, PostgreSQL, Redis, and Docker in production"
        )
        names = [e["name"] for e in result["entities"]]
        assert "Python" in names
        assert "FastAPI" in names
        assert "PostgreSQL" in names
        assert "Redis" in names
        assert "Docker" in names


# --- extract_and_store ---

class TestExtractAndStore:
    def test_stores_new_entities(self, extractor_with_kb, mock_kb):
        stats = extractor_with_kb.extract_and_store("I'm building with FastAPI and PostgreSQL")
        assert stats["entities_added"] >= 2
        assert mock_kb.add_entity.call_count >= 2

    def test_reinforces_existing(self, extractor_with_kb, mock_kb):
        mock_kb.search_entities.return_value = [{"name": "FastAPI", "type": "technology"}]
        stats = extractor_with_kb.extract_and_store("I'm building with FastAPI")
        assert stats["entities_reinforced"] >= 1

    def test_stores_relations(self, extractor_with_kb, mock_kb):
        stats = extractor_with_kb.extract_and_store("React uses TypeScript")
        assert stats["relations_added"] >= 1
        mock_kb.add_relation.assert_called()

    def test_no_kb(self):
        extractor = EntityExtractor(knowledge_base=None)
        stats = extractor.extract_and_store("Python and React")
        assert stats == {"entities_added": 0, "entities_reinforced": 0, "relations_added": 0}

    def test_empty_text_no_store(self, extractor_with_kb, mock_kb):
        stats = extractor_with_kb.extract_and_store("")
        assert stats["entities_added"] == 0
        assert stats["relations_added"] == 0
        mock_kb.add_entity.assert_not_called()
