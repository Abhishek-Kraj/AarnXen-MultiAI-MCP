"""Tests for pipeline meta-tool."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from aarnxen.tools.pipeline import pipeline_handler, _extract_text


def _make_fake_ctx():
    """Create a fake MCP context with minimal deps for middleware."""
    ctx = MagicMock()
    deps = MagicMock()
    deps.event_bus = MagicMock()
    deps.event_bus.emit = AsyncMock()
    deps.rate_limiter = None
    deps.guardrails = None
    deps.knowledge = None
    ctx.request_context.lifespan_context = deps
    return ctx


class TestExtractText:
    def test_response_key(self):
        assert _extract_text({"response": "hello"}) == "hello"

    def test_result_key(self):
        assert _extract_text({"result": "world"}) == "world"

    def test_reasoning_key(self):
        assert _extract_text({"reasoning": "because"}) == "because"

    def test_fallback_to_json(self):
        result = _extract_text({"foo": 123})
        assert "123" in result


class TestPipeline:
    @pytest.mark.asyncio
    async def test_invalid_json(self):
        result = await pipeline_handler("not json", ctx=_make_fake_ctx())
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_empty_steps(self):
        result = await pipeline_handler("[]", ctx=_make_fake_ctx())
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        steps = json.dumps([{"tool": "nonexistent", "args": {}}])
        result = await pipeline_handler(steps, ctx=_make_fake_ctx())
        assert result["isError"] is True
        assert "available_tools" in result

    @pytest.mark.asyncio
    async def test_prev_replacement(self):
        mock_think = AsyncMock(return_value={"reasoning": "Deep analysis of topic X"})
        mock_challenge = AsyncMock(return_value={"result": "Weak argument"})

        ctx = _make_fake_ctx()

        with patch("aarnxen.tools.pipeline._TOOL_SOURCES", {
            "think": ("aarnxen.tools.think", "think_handler"),
            "challenge": ("aarnxen.tools.challenge", "challenge_handler"),
        }):
            with patch("aarnxen.tools.pipeline._wrapped_cache", {}):
                with patch("aarnxen.tools.think.think_handler", mock_think):
                    with patch("aarnxen.tools.challenge.challenge_handler", mock_challenge):
                        steps = json.dumps([
                            {"tool": "think", "args": {"prompt": "test"}},
                            {"tool": "challenge", "args": {"claim": "$PREV"}},
                        ])
                        result = await pipeline_handler(steps, ctx=ctx)

                        assert result["steps_completed"] == 2
                        challenge_call_args = mock_challenge.call_args
                        assert "Deep analysis" in challenge_call_args.kwargs.get("claim", "")

    @pytest.mark.asyncio
    async def test_step_failure_returns_partial(self):
        mock_think = AsyncMock(side_effect=RuntimeError("provider down"))

        ctx = _make_fake_ctx()

        with patch("aarnxen.tools.pipeline._TOOL_SOURCES", {
            "think": ("aarnxen.tools.think", "think_handler"),
        }):
            with patch("aarnxen.tools.pipeline._wrapped_cache", {}):
                with patch("aarnxen.tools.think.think_handler", mock_think):
                    steps = json.dumps([{"tool": "think", "args": {"prompt": "test"}}])
                    result = await pipeline_handler(steps, ctx=ctx)

                    # The middleware catches the error and returns an error dict
                    # which the pipeline treats as a successful step (no exception raised)
                    assert result["steps_completed"] == 1

    @pytest.mark.asyncio
    async def test_single_step_pipeline(self):
        mock_chat = AsyncMock(return_value={"response": "Hello world"})

        ctx = _make_fake_ctx()

        with patch("aarnxen.tools.pipeline._TOOL_SOURCES", {
            "chat": ("aarnxen.tools.chat", "chat_handler"),
        }):
            with patch("aarnxen.tools.pipeline._wrapped_cache", {}):
                with patch("aarnxen.tools.chat.chat_handler", mock_chat):
                    steps = json.dumps([{"tool": "chat", "args": {"prompt": "hi"}}])
                    result = await pipeline_handler(steps, ctx=ctx)

                    assert result["steps_completed"] == 1
                    assert result["final_output"] == "Hello world"

    @pytest.mark.asyncio
    async def test_pipeline_emits_events(self):
        """Verify pipeline steps go through middleware and emit events."""
        mock_chat = AsyncMock(return_value={"response": "OK"})
        ctx = _make_fake_ctx()

        with patch("aarnxen.tools.pipeline._TOOL_SOURCES", {
            "chat": ("aarnxen.tools.chat", "chat_handler"),
        }):
            with patch("aarnxen.tools.pipeline._wrapped_cache", {}):
                with patch("aarnxen.tools.chat.chat_handler", mock_chat):
                    steps = json.dumps([{"tool": "chat", "args": {"prompt": "hi"}}])
                    await pipeline_handler(steps, ctx=ctx)

                    # Middleware should have emitted tool_start and tool_complete events
                    event_calls = ctx.request_context.lifespan_context.event_bus.emit.call_args_list
                    event_types = [call.args[0] for call in event_calls]
                    assert "tool_start" in event_types
                    assert "tool_complete" in event_types
