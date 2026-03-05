"""Tests for jury tool (N-model voting)."""

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
    deps.registry.get_top_n_models.return_value = [("test", "m1"), ("test", "m2"), ("test", "m3")]
    deps.registry.get_fallbacks.return_value = []

    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    return ctx, deps, provider


@pytest.mark.asyncio
async def test_jury_returns_aggregate():
    ctx, deps, provider = _make_ctx()

    async def mock_generate(prompt, model, **kwargs):
        return ModelResponse(
            text='{"score": 8, "reasoning": "Good quality", "strengths": ["clear"], "weaknesses": ["verbose"]}',
            model=model, provider="test", input_tokens=20, output_tokens=30, latency_ms=150,
        )

    provider.generate = mock_generate

    from aarnxen.tools.jury import jury_handler
    result = await jury_handler("Some code to review", criteria="code", num_jurors=3, ctx=ctx)

    assert result["juror_count"] == 3
    assert result["aggregate"]["average_score"] == 8.0
    assert result["aggregate"]["consensus"] is True


@pytest.mark.asyncio
async def test_jury_handles_invalid_json():
    ctx, deps, provider = _make_ctx()

    async def mock_generate(prompt, model, **kwargs):
        return ModelResponse(
            text="This is great! Score: 9/10",
            model=model, provider="test", input_tokens=10, output_tokens=10, latency_ms=100,
        )

    provider.generate = mock_generate

    from aarnxen.tools.jury import jury_handler
    result = await jury_handler("Test content", num_jurors=3, ctx=ctx)

    # Should fallback to score 5 on parse error
    assert result["juror_count"] == 3
    assert all(v["score"] == 5 for v in result["verdicts"])
