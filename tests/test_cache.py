"""Tests for response cache."""

import time

import pytest

from aarnxen.core.cache import ResponseCache
from aarnxen.providers.base import ModelResponse


def _make_response(text="hello"):
    return ModelResponse(text=text, model="test", provider="test", input_tokens=10, output_tokens=20)


def test_cache_miss():
    cache = ResponseCache(max_size=10, ttl_seconds=60)
    result = cache.get("p", "m", "prompt", "", 0.7)
    assert result is None
    assert cache.stats()["misses"] == 1


def test_cache_hit():
    cache = ResponseCache(max_size=10, ttl_seconds=60)
    resp = _make_response("cached response")
    cache.put("p", "m", "prompt", "", 0.7, resp)
    result = cache.get("p", "m", "prompt", "", 0.7)
    assert result is not None
    assert result.text == "cached response"
    assert result.cached is True
    assert cache.stats()["hits"] == 1


def test_cache_different_params_miss():
    cache = ResponseCache(max_size=10, ttl_seconds=60)
    resp = _make_response()
    cache.put("p", "m", "prompt", "", 0.7, resp)
    # Different temperature = different key
    assert cache.get("p", "m", "prompt", "", 0.9) is None


def test_cache_lru_eviction():
    cache = ResponseCache(max_size=2, ttl_seconds=60)
    cache.put("p", "m", "a", "", 0.7, _make_response("a"))
    cache.put("p", "m", "b", "", 0.7, _make_response("b"))
    cache.put("p", "m", "c", "", 0.7, _make_response("c"))
    # "a" should be evicted
    assert cache.get("p", "m", "a", "", 0.7) is None
    assert cache.get("p", "m", "b", "", 0.7) is not None
    assert cache.stats()["size"] == 2


def test_cache_stats():
    cache = ResponseCache(max_size=10, ttl_seconds=60)
    cache.put("p", "m", "prompt", "", 0.7, _make_response())
    cache.get("p", "m", "prompt", "", 0.7)  # hit
    cache.get("p", "m", "other", "", 0.7)   # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
