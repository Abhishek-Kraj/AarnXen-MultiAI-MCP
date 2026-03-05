"""Tests for debate tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
    deps.registry.get_top_n_models.return_value = [("test", "m1"), ("test", "m2"), ("test", "m3")]
    deps.registry.get_fallbacks.return_value = []
    deps.knowledge = MagicMock()

    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    return ctx, deps, provider


@pytest.mark.asyncio
async def test_debate_runs_rounds():
    ctx, deps, provider = _make_ctx()

    call_count = 0
    responses = [
        "I argue FOR because X",
        "I argue AGAINST because Y",
        "FOR responds with Z",
        "AGAINST counters with W",
        "The FOR side wins because...",
    ]

    async def mock_generate(prompt, model, **kwargs):
        nonlocal call_count
        text = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return ModelResponse(text=text, model=model, provider="test", input_tokens=10, output_tokens=20, latency_ms=100)

    provider.generate = mock_generate

    from aarnxen.tools.debate import debate_handler
    result = await debate_handler("Is Python better than Rust?", rounds=2, ctx=ctx)

    assert result["rounds"] == 2
    assert len(result["transcript"]) == 4  # 2 rounds × 2 sides
    assert result["verdict"]
    assert result["total_cost_usd"] > 0


@pytest.mark.asyncio
async def test_debate_clamps_rounds():
    ctx, deps, provider = _make_ctx()

    async def mock_generate(prompt, model, **kwargs):
        return ModelResponse(text="arg", model=model, provider="test", input_tokens=5, output_tokens=5, latency_ms=50)

    provider.generate = mock_generate

    from aarnxen.tools.debate import debate_handler
    result = await debate_handler("test topic", rounds=10, ctx=ctx)

    # Should be clamped to 5
    assert result["rounds"] == 5
