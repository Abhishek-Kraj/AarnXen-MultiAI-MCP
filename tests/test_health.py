"""Tests for health check tool."""

import pytest
from unittest.mock import MagicMock


def _make_ctx(provider_count=5, cb_status=None, kb_ok=True, cache_ok=True):
    deps = MagicMock()
    deps.registry.list_all_models.return_value = [{"model": f"m{i}"} for i in range(provider_count)]
    deps.circuit_breaker.get_all_status.return_value = cb_status or {}

    if kb_ok:
        deps.knowledge.stats.return_value = {"docs": 10}
    else:
        deps.knowledge = None

    deps.cache = MagicMock() if cache_ok else None

    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    return ctx


@pytest.mark.asyncio
async def test_health_healthy():
    from aarnxen.server import health
    ctx = _make_ctx()
    result = await health(ctx=ctx)
    assert result["status"] == "healthy"
    assert result["providers"]["total"] == 5
    assert result["cache"]["enabled"] is True
    assert result["knowledge_base"]["connected"] is True


@pytest.mark.asyncio
async def test_health_degraded_no_kb():
    from aarnxen.server import health
    ctx = _make_ctx(kb_ok=False)
    result = await health(ctx=ctx)
    assert result["status"] == "degraded"
    assert result["knowledge_base"]["connected"] is False
