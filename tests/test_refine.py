"""Tests for refine tool (Self-Refine)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aarnxen.providers.base import ModelResponse


def _make_ctx():
    deps = MagicMock()
    deps.event_bus = MagicMock()
    deps.event_bus.emit = AsyncMock()
    deps.rate_limiter = None
    deps.guardrails = None
    deps.cost_tracker.record.return_value = MagicMock(cost_usd=0.001)
    deps.circuit_breaker = None

    provider = MagicMock()
    provider.provider_name.return_value = "test"
    deps.registry.resolve.return_value = (provider, "test-model")
    deps.registry.get_top_n_models.return_value = [("test", "m1"), ("test", "m2")]
    deps.registry.get_fallbacks.return_value = []

    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    return ctx, deps, provider


@pytest.mark.asyncio
async def test_refine_iterates():
    ctx, deps, provider = _make_ctx()

    call_count = 0
    async def mock_generate(prompt, model, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(text="Initial draft v1", model=model, provider="test", input_tokens=10, output_tokens=20, latency_ms=100)
        elif call_count % 2 == 0:
            return ModelResponse(text=f"Critique: needs improvement in area {call_count}", model=model, provider="test", input_tokens=10, output_tokens=15, latency_ms=80)
        else:
            return ModelResponse(text=f"Refined version {call_count}", model=model, provider="test", input_tokens=10, output_tokens=20, latency_ms=100)

    provider.generate = mock_generate

    from aarnxen.tools.refine import refine_handler
    result = await refine_handler("Write a function to sort", iterations=2, ctx=ctx)

    assert result["iterations"] == 2
    assert result["final_response"]
    # 1 initial + 2*(critique + refinement) = 5 total calls
    assert len(result["history"]) == 5
    assert result["history"][0]["type"] == "generation"
    assert result["history"][1]["type"] == "critique"
    assert result["history"][2]["type"] == "refinement"


@pytest.mark.asyncio
async def test_refine_single_iteration():
    ctx, deps, provider = _make_ctx()

    async def mock_generate(prompt, model, **kwargs):
        return ModelResponse(text="Response text", model=model, provider="test", input_tokens=5, output_tokens=10, latency_ms=50)

    provider.generate = mock_generate

    from aarnxen.tools.refine import refine_handler
    result = await refine_handler("Test", iterations=1, ctx=ctx)

    assert result["iterations"] == 1
    assert len(result["history"]) == 3  # 1 gen + 1 critique + 1 refine
