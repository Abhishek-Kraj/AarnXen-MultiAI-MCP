"""End-to-end tests — server startup, registry, full pipeline."""

import os
import tempfile

import pytest
import yaml

from aarnxen.config import AarnXenConfig, ProviderConfig, load_config
from aarnxen.core.cache import ResponseCache
from aarnxen.core.conversation import ConversationMemory
from aarnxen.core.cost import CostTracker
from aarnxen.core.knowledge import KnowledgeBase
from aarnxen.providers.base import ModelResponse
from aarnxen.providers.registry import ProviderRegistry


def test_registry_from_config_no_keys():
    """Registry should handle missing API keys gracefully."""
    cfg = AarnXenConfig(
        providers=[
            ProviderConfig(name="openai", api_key=None, priority=1, models=["gpt-4o"]),
        ]
    )
    registry = ProviderRegistry.from_config(cfg)
    assert len(registry._providers) == 0


def test_registry_resolve_errors():
    """Registry should raise clear errors when no providers available."""
    cfg = AarnXenConfig(providers=[])
    registry = ProviderRegistry.from_config(cfg)
    with pytest.raises(RuntimeError, match="No providers configured"):
        registry.resolve()


def test_registry_resolve_unknown_model():
    """Registry should raise ValueError for unknown models when providers exist."""
    cfg = AarnXenConfig(
        providers=[
            ProviderConfig(name="ollama-local", base_url="http://localhost:11434", priority=1, models=["llama3"]),
        ]
    )
    registry = ProviderRegistry.from_config(cfg)
    with pytest.raises(ValueError, match="Unknown model"):
        registry.resolve("totally-fake-model-xyz")


def test_full_cache_cost_pipeline():
    """Test cache + cost tracker working together."""
    cache = ResponseCache(max_size=10, ttl_seconds=60)
    cost_tracker = CostTracker()

    # Simulate a response
    response = ModelResponse(
        text="Hello!", model="gemini-2.5-flash", provider="gemini",
        input_tokens=100, output_tokens=50,
    )

    # First call — cache miss, record cost
    cached = cache.get("gemini", "gemini-2.5-flash", "Hi", "", 0.7)
    assert cached is None

    cache.put("gemini", "gemini-2.5-flash", "Hi", "", 0.7, response)
    entry = cost_tracker.record("gemini", "gemini-2.5-flash", 100, 50)
    assert entry.cost_usd > 0

    # Second call — cache hit, zero cost
    cached = cache.get("gemini", "gemini-2.5-flash", "Hi", "", 0.7)
    assert cached is not None
    assert cached.cached is True

    entry2 = cost_tracker.record("gemini", "gemini-2.5-flash", 100, 50, cached=True)
    assert entry2.cost_usd == 0.0

    # Summary should show 2 requests, one cached
    summary = cost_tracker.summary()
    assert summary["total_requests"] == 2


def test_conversation_memory_persistence(tmp_path):
    """Test conversation memory survives close/reopen."""
    db_path = str(tmp_path / "conv.db")

    # Session 1: write
    mem = ConversationMemory(persist_path=db_path)
    mem.add_message("conv-1", "user", "What is Python?")
    mem.add_message("conv-1", "assistant", "A programming language.", "gemini-2.5-flash", "gemini")
    mem.close()

    # Session 2: read back
    mem2 = ConversationMemory(persist_path=db_path)
    history = mem2.get_history("conv-1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "A programming language."
    mem2.close()


def test_knowledge_base_persistence(tmp_path):
    """Test knowledge base survives close/reopen."""
    db_path = str(tmp_path / "kb.db")

    # Session 1
    kb = KnowledgeBase(db_path=db_path)
    kb.store_document("AarnXen Architecture", "Multi-AI MCP server with parallel consensus", tags="architecture")
    kb.add_entity("AarnXen", entity_type="project")
    kb.add_observation("AarnXen", "Uses asyncio.gather for parallel consensus")
    kb.close()

    # Session 2
    kb2 = KnowledgeBase(db_path=db_path)
    results = kb2.search_documents("parallel consensus")
    assert len(results) >= 1

    entities = kb2.search_entities("AarnXen")
    assert len(entities) == 1
    assert "asyncio.gather" in entities[0]["observations"][0]
    kb2.close()


def test_config_yaml_roundtrip(tmp_path):
    """Test config can be written and read as YAML."""
    config_data = {
        "default_model": "auto",
        "default_temperature": 0.5,
        "log_level": "DEBUG",
        "providers": [
            {"name": "gemini", "api_key": "test", "priority": 1, "models": ["gemini-2.5-flash"]},
            {"name": "ollama-local", "base_url": "http://localhost:11434", "priority": 10, "models": ["llama3"]},
        ],
        "cache": {"enabled": True, "max_size": 50, "ttl_seconds": 1800},
        "memory": {"enabled": True, "path": str(tmp_path / "mem.db")},
        "cost_tracking": True,
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))

    cfg = load_config(str(config_file))
    assert cfg.default_temperature == 0.5
    assert cfg.log_level == "DEBUG"
    assert len(cfg.providers) == 2
    assert cfg.providers[0].name == "gemini"
    assert cfg.providers[1].base_url == "http://localhost:11434"
    assert cfg.cache.max_size == 50
