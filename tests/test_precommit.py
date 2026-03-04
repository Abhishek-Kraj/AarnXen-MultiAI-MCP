"""Tests for pre-commit review tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aarnxen.providers.base import ModelResponse
from aarnxen.tools.precommit import precommit_handler


def _make_ctx(response_text="No issues found. PASS"):
    provider = MagicMock()
    provider.provider_name.return_value = "test-provider"
    provider.generate = AsyncMock(return_value=ModelResponse(
        text=response_text,
        model="test-model",
        provider="test-provider",
        input_tokens=200,
        output_tokens=100,
        latency_ms=150.0,
    ))

    cost_entry = MagicMock(cost_usd=0.001)

    deps = {
        "registry": MagicMock(),
        "cost_tracker": MagicMock(),
    }
    deps["registry"].resolve.return_value = (provider, "test-model")
    deps["cost_tracker"].record.return_value = cost_entry

    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    return ctx, provider, deps


@pytest.mark.asyncio
async def test_precommit_calls_generate_with_system_prompt():
    ctx, provider, _ = _make_ctx()
    await precommit_handler("diff --git a/file.py", ctx=ctx)

    provider.generate.assert_called_once()
    call_kwargs = provider.generate.call_args
    assert "senior code reviewer" in call_kwargs.kwargs.get("system_prompt", call_kwargs[1].get("system_prompt", "")).lower() or \
           "senior code reviewer" in str(call_kwargs).lower()
    assert call_kwargs.kwargs.get("temperature", call_kwargs[1].get("temperature")) == 0.3


@pytest.mark.asyncio
async def test_precommit_records_cost():
    ctx, _, deps = _make_ctx()
    await precommit_handler("diff --git a/file.py", ctx=ctx)

    deps["cost_tracker"].record.assert_called_once_with(
        "test-provider", "test-model", 200, 100,
    )


@pytest.mark.asyncio
async def test_precommit_response_shape():
    ctx, _, _ = _make_ctx()
    result = await precommit_handler("diff --git a/file.py", ctx=ctx)

    assert "result" in result
    assert "model" in result
    assert "provider" in result
    assert "tokens" in result
    assert "cost_usd" in result
    assert "latency_ms" in result
    assert "verdict" in result
    assert result["model"] == "test-model"
    assert result["provider"] == "test-provider"
    assert result["verdict"] == "PASS"


@pytest.mark.asyncio
async def test_precommit_verdict_fail():
    ctx, _, _ = _make_ctx("Critical bug found. FAIL")
    result = await precommit_handler("diff --git a/file.py", ctx=ctx)
    assert result["verdict"] == "FAIL"
