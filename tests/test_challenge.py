"""Tests for devil's advocate challenge tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aarnxen.providers.base import ModelResponse
from aarnxen.tools.challenge import challenge_handler


def _make_ctx(response_text="The claim has merit. Verdict: MODERATE"):
    provider = MagicMock()
    provider.provider_name.return_value = "test-provider"
    provider.generate = AsyncMock(return_value=ModelResponse(
        text=response_text,
        model="test-model",
        provider="test-provider",
        input_tokens=300,
        output_tokens=200,
        latency_ms=250.0,
    ))

    cost_entry = MagicMock(cost_usd=0.002)

    deps = MagicMock()
    deps.registry.resolve.return_value = (provider, "test-model")
    deps.cost_tracker.record.return_value = cost_entry

    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    return ctx, provider, deps


@pytest.mark.asyncio
async def test_challenge_calls_generate_with_system_prompt():
    ctx, provider, _ = _make_ctx()
    await challenge_handler("Python is the best language", ctx=ctx)

    provider.generate.assert_called_once()
    call_kwargs = provider.generate.call_args
    assert call_kwargs.kwargs.get("temperature", call_kwargs[1].get("temperature")) == 0.4


@pytest.mark.asyncio
async def test_challenge_records_cost():
    ctx, _, deps = _make_ctx()
    await challenge_handler("Microservices are always better", ctx=ctx)

    deps.cost_tracker.record.assert_called_once_with(
        "test-provider", "test-model", 300, 200,
    )


@pytest.mark.asyncio
async def test_challenge_response_shape():
    ctx, _, _ = _make_ctx()
    result = await challenge_handler("AI will replace all developers", ctx=ctx)

    assert "result" in result
    assert "model" in result
    assert "provider" in result
    assert "tokens" in result
    assert "cost_usd" in result
    assert "latency_ms" in result
    assert "verdict" in result
    assert result["model"] == "test-model"
    assert result["provider"] == "test-provider"
    assert result["verdict"] == "MODERATE"


@pytest.mark.asyncio
async def test_challenge_includes_evidence_in_prompt():
    ctx, provider, _ = _make_ctx()
    await challenge_handler("Rust is memory-safe", evidence="No GC needed", ctx=ctx)

    call_args = provider.generate.call_args
    prompt = call_args[0][0]
    assert "Rust is memory-safe" in prompt
    assert "No GC needed" in prompt


@pytest.mark.asyncio
async def test_challenge_verdict_strong():
    ctx, _, _ = _make_ctx("This is a STRONG claim with solid reasoning.")
    result = await challenge_handler("Water is wet", ctx=ctx)
    assert result["verdict"] == "STRONG"


@pytest.mark.asyncio
async def test_challenge_verdict_weak():
    ctx, _, _ = _make_ctx("This is a WEAK argument with many flaws.")
    result = await challenge_handler("Earth is flat", ctx=ctx)
    assert result["verdict"] == "WEAK"
