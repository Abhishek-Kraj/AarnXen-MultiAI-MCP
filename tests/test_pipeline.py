"""Tests for pipeline meta-tool."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from aarnxen.tools.pipeline import pipeline_handler, _extract_text


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
        result = await pipeline_handler("not json", ctx=None)
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_empty_steps(self):
        result = await pipeline_handler("[]", ctx=None)
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        steps = json.dumps([{"tool": "nonexistent", "args": {}}])
        result = await pipeline_handler(steps, ctx=None)
        assert result["isError"] is True
        assert "available_tools" in result

    @pytest.mark.asyncio
    async def test_prev_replacement(self):
        mock_think = AsyncMock(return_value={"reasoning": "Deep analysis of topic X"})
        mock_challenge = AsyncMock(return_value={"result": "Weak argument"})

        with patch("aarnxen.tools.pipeline.TOOL_REGISTRY", {
            "think": ("aarnxen.tools.think", "think_handler"),
            "challenge": ("aarnxen.tools.challenge", "challenge_handler"),
        }):
            with patch("aarnxen.tools.think.think_handler", mock_think):
                with patch("aarnxen.tools.challenge.challenge_handler", mock_challenge):
                    steps = json.dumps([
                        {"tool": "think", "args": {"prompt": "test"}},
                        {"tool": "challenge", "args": {"claim": "$PREV"}},
                    ])
                    result = await pipeline_handler(steps, ctx=None)

                    assert result["steps_completed"] == 2
                    challenge_call_args = mock_challenge.call_args
                    assert "Deep analysis" in challenge_call_args.kwargs.get("claim", "")

    @pytest.mark.asyncio
    async def test_step_failure_returns_partial(self):
        mock_think = AsyncMock(side_effect=RuntimeError("provider down"))

        with patch("aarnxen.tools.pipeline.TOOL_REGISTRY", {
            "think": ("aarnxen.tools.think", "think_handler"),
        }):
            with patch("aarnxen.tools.think.think_handler", mock_think):
                steps = json.dumps([{"tool": "think", "args": {"prompt": "test"}}])
                result = await pipeline_handler(steps, ctx=None)

                assert result["isError"] is True
                assert result["completed_steps"] == 0

    @pytest.mark.asyncio
    async def test_single_step_pipeline(self):
        mock_chat = AsyncMock(return_value={"response": "Hello world"})

        with patch("aarnxen.tools.pipeline.TOOL_REGISTRY", {
            "chat": ("aarnxen.tools.chat", "chat_handler"),
        }):
            with patch("aarnxen.tools.chat.chat_handler", mock_chat):
                steps = json.dumps([{"tool": "chat", "args": {"prompt": "hi"}}])
                result = await pipeline_handler(steps, ctx=None)

                assert result["steps_completed"] == 1
                assert result["final_output"] == "Hello world"
