"""Tests for verify tool (Chain of Verification)."""

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
    deps.registry.get_fallbacks.return_value = []

    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    return ctx, deps, provider


@pytest.mark.asyncio
async def test_verify_runs_4_steps():
    ctx, deps, provider = _make_ctx()

    step = 0
    async def mock_generate(prompt, model, **kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(text="Initial assessment: claim seems plausible", model=model, provider="test", input_tokens=10, output_tokens=20, latency_ms=100)
        elif step == 2:
            return ModelResponse(text="1. Is X true?\n2. What evidence exists?\n3. Are there counter-examples?", model=model, provider="test", input_tokens=10, output_tokens=20, latency_ms=100)
        elif step <= 5:
            return ModelResponse(text=f"Answer to question {step-2}", model=model, provider="test", input_tokens=10, output_tokens=20, latency_ms=100)
        else:
            return ModelResponse(text="Revised: HIGH confidence. The claim is verified.", model=model, provider="test", input_tokens=10, output_tokens=30, latency_ms=100)

    provider.generate = mock_generate

    from aarnxen.tools.verify import verify_handler
    result = await verify_handler("Python is the most popular language", ctx=ctx)

    assert result["claim"] == "Python is the most popular language"
    assert result["draft_assessment"]
    assert len(result["verification_questions"]) >= 1
    assert result["revised_assessment"]
    assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")


@pytest.mark.asyncio
async def test_verify_extracts_confidence():
    ctx, deps, provider = _make_ctx()

    async def mock_generate(prompt, model, **kwargs):
        if "revised" in prompt.lower() or "revised" in kwargs.get("system_prompt", "").lower() or "synthesize" in kwargs.get("system_prompt", "").lower():
            return ModelResponse(text="LOW confidence. Evidence is contradictory.", model=model, provider="test", input_tokens=10, output_tokens=20, latency_ms=100)
        if "verification question" in prompt.lower() or "generate" in kwargs.get("system_prompt", "").lower():
            return ModelResponse(text="1. Is this accurate?", model=model, provider="test", input_tokens=10, output_tokens=10, latency_ms=50)
        return ModelResponse(text="Some answer here", model=model, provider="test", input_tokens=10, output_tokens=10, latency_ms=50)

    provider.generate = mock_generate

    from aarnxen.tools.verify import verify_handler
    result = await verify_handler("Unverifiable claim", ctx=ctx)

    assert result["confidence"] == "LOW"
