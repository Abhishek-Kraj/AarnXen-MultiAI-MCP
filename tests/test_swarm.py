"""Tests for multi-agent swarm tool."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from aarnxen.providers.base import ModelResponse
from aarnxen.tools.swarm import swarm_handler


def _make_response(text="Agent response", input_tokens=100, output_tokens=50):
    return ModelResponse(
        text=text,
        model="test-model",
        provider="test-provider",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=100.0,
    )


def _make_ctx(response=None, cache_hit=False):
    provider = MagicMock()
    provider.provider_name.return_value = "test-provider"
    provider.generate = AsyncMock(return_value=response or _make_response())

    cost_entry = MagicMock(cost_usd=0.001)

    registry = MagicMock()
    registry.resolve.return_value = (provider, "test-model")
    registry.get_fallbacks.return_value = []

    cost_tracker = MagicMock()
    cost_tracker.record.return_value = cost_entry

    cache = MagicMock()
    if cache_hit:
        cache.get.return_value = response or _make_response()
    else:
        cache.get.return_value = None

    circuit_breaker = MagicMock()
    circuit_breaker.can_execute.return_value = True

    deps = MagicMock()
    deps.registry = registry
    deps.cost_tracker = cost_tracker
    deps.cache = cache
    deps.circuit_breaker = circuit_breaker

    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    ctx.report_progress = AsyncMock()
    return ctx, provider, deps


class TestSwarmValidation:
    @pytest.mark.asyncio
    async def test_invalid_json(self):
        ctx, _, _ = _make_ctx()
        result = await swarm_handler("not json", ctx=ctx)
        assert result["isError"] is True
        assert "Invalid JSON" in result["message"]

    @pytest.mark.asyncio
    async def test_empty_array(self):
        ctx, _, _ = _make_ctx()
        result = await swarm_handler("[]", ctx=ctx)
        assert result["isError"] is True
        assert "empty" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_not_array(self):
        ctx, _, _ = _make_ctx()
        result = await swarm_handler('{"prompt": "hi"}', ctx=ctx)
        assert result["isError"] is True
        assert "array" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_too_many_agents(self):
        ctx, _, _ = _make_ctx()
        tasks = [{"prompt": f"task {i}"} for i in range(101)]
        result = await swarm_handler(json.dumps(tasks), ctx=ctx)
        assert result["isError"] is True
        assert "100" in result["message"]


class TestSwarmExecution:
    @pytest.mark.asyncio
    async def test_single_agent(self):
        ctx, provider, _ = _make_ctx()
        tasks = [{"prompt": "analyze auth module", "label": "Security"}]
        result = await swarm_handler(json.dumps(tasks), ctx=ctx)

        assert result["agent_count"] == 1
        assert len(result["responses"]) == 1
        assert result["responses"][0]["label"] == "Security"
        assert result["responses"][0]["model"] == "test-model"
        assert result["responses"][0]["response"] == "Agent response"
        assert result["summary"]["succeeded"] == 1
        assert result["summary"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_multiple_agents(self):
        ctx, provider, _ = _make_ctx()
        tasks = [
            {"prompt": "review security", "label": "Security"},
            {"prompt": "review performance", "label": "Performance"},
            {"prompt": "review errors", "label": "Error Handling"},
        ]
        result = await swarm_handler(json.dumps(tasks), ctx=ctx)

        assert result["agent_count"] == 3
        assert len(result["responses"]) == 3
        assert result["summary"]["succeeded"] == 3
        assert result["summary"]["failed"] == 0
        assert result["summary"]["total_tokens"] > 0
        assert "instruction" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_skipped(self):
        ctx, _, _ = _make_ctx()
        tasks = [{"prompt": "", "label": "Empty"}, {"prompt": "valid task"}]
        result = await swarm_handler(json.dumps(tasks), ctx=ctx)

        assert result["summary"]["succeeded"] == 1
        assert result["summary"]["failed"] == 1
        assert result["errors"][0]["label"] == "Empty"

    @pytest.mark.asyncio
    async def test_default_labels(self):
        ctx, _, _ = _make_ctx()
        tasks = [{"prompt": "task 1"}, {"prompt": "task 2"}]
        result = await swarm_handler(json.dumps(tasks), ctx=ctx)

        labels = [r["label"] for r in result["responses"]]
        assert "Agent 1" in labels
        assert "Agent 2" in labels

    @pytest.mark.asyncio
    async def test_per_task_model_override(self):
        ctx, _, deps = _make_ctx()
        tasks = [
            {"prompt": "task 1", "model": "kimi-k2-thinking"},
            {"prompt": "task 2", "model": "glm-5"},
        ]
        result = await swarm_handler(json.dumps(tasks), ctx=ctx)

        calls = deps.registry.resolve.call_args_list
        resolved_models = [c[0][0] for c in calls]
        assert "kimi-k2-thinking" in resolved_models
        assert "glm-5" in resolved_models

    @pytest.mark.asyncio
    async def test_provider_failure_captured(self):
        provider = MagicMock()
        provider.provider_name.return_value = "test-provider"
        provider.generate = AsyncMock(side_effect=RuntimeError("provider down"))

        deps = MagicMock()
        deps.registry = MagicMock()
        deps.registry.resolve.return_value = (provider, "test-model")
        deps.registry.get_fallbacks.return_value = []
        deps.cost_tracker = MagicMock()
        deps.cache = MagicMock()
        deps.cache.get.return_value = None
        deps.circuit_breaker = MagicMock()
        deps.circuit_breaker.can_execute.return_value = True

        ctx = MagicMock()
        ctx.request_context.lifespan_context = deps
        ctx.report_progress = AsyncMock()

        tasks = [{"prompt": "will fail"}]
        result = await swarm_handler(json.dumps(tasks), ctx=ctx)

        assert result["summary"]["succeeded"] == 0
        assert result["summary"]["failed"] == 1
        assert result["errors"][0]["error"]  # sanitized error (class name only)


class TestSwarmCaching:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        ctx, provider, _ = _make_ctx(cache_hit=True)
        tasks = [{"prompt": "cached task"}]
        result = await swarm_handler(json.dumps(tasks), ctx=ctx)

        assert result["responses"][0]["cached"] is True
        provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_stores_result(self):
        ctx, _, deps = _make_ctx(cache_hit=False)
        tasks = [{"prompt": "uncached task"}]
        await swarm_handler(json.dumps(tasks), ctx=ctx)

        deps.cache.put.assert_called_once()


class TestSwarmConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_clamped(self):
        ctx, _, _ = _make_ctx()
        tasks = [{"prompt": "task"}]
        result = await swarm_handler(json.dumps(tasks), concurrency=0, ctx=ctx)
        assert result["summary"]["concurrency"] == 1

    @pytest.mark.asyncio
    async def test_concurrency_max(self):
        ctx, _, _ = _make_ctx()
        tasks = [{"prompt": "task"}]
        result = await swarm_handler(json.dumps(tasks), concurrency=200, ctx=ctx)
        assert result["summary"]["concurrency"] == 100

    @pytest.mark.asyncio
    async def test_progress_reported(self):
        ctx, _, _ = _make_ctx()
        tasks = [{"prompt": "task 1"}, {"prompt": "task 2"}]
        await swarm_handler(json.dumps(tasks), ctx=ctx)

        ctx.report_progress.assert_called()
        calls = ctx.report_progress.call_args_list
        assert len(calls) >= 2
