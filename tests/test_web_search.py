"""Tests for web search tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_ctx():
    deps = MagicMock()
    deps.event_bus = MagicMock()
    deps.event_bus.emit = AsyncMock()
    deps.rate_limiter = None
    deps.guardrails = None
    ctx = MagicMock()
    ctx.request_context.lifespan_context = deps
    return ctx, deps


@pytest.mark.asyncio
async def test_web_search_returns_results():
    ctx, deps = _make_ctx()
    fake_results = [
        {"title": "Python Docs", "href": "https://python.org", "body": "Python language"},
        {"title": "PEP 8", "href": "https://pep8.org", "body": "Style guide"},
    ]

    with patch("duckduckgo_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ddgs.return_value)
        mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
        mock_ddgs.return_value.text.return_value = fake_results

        from aarnxen.tools.web_search import web_search_handler
        result = await web_search_handler("python", max_results=2, ctx=ctx)

    assert result["count"] == 2
    assert result["results"][0]["title"] == "Python Docs"
    assert result["results"][0]["url"] == "https://python.org"


@pytest.mark.asyncio
async def test_web_search_handles_error():
    ctx, deps = _make_ctx()

    with patch("duckduckgo_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ddgs.return_value)
        mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
        mock_ddgs.return_value.text.side_effect = Exception("Network error")

        from aarnxen.tools.web_search import web_search_handler
        result = await web_search_handler("test", ctx=ctx)

    assert "error" in result


@pytest.mark.asyncio
async def test_web_search_clamps_max_results():
    ctx, deps = _make_ctx()

    with patch("duckduckgo_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ddgs.return_value)
        mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
        mock_ddgs.return_value.text.return_value = []

        from aarnxen.tools.web_search import web_search_handler
        result = await web_search_handler("test", max_results=100, ctx=ctx)

    # Should have been clamped to 20
    mock_ddgs.return_value.text.assert_called_once()
    call_kwargs = mock_ddgs.return_value.text.call_args
    assert call_kwargs[1]["max_results"] == 20 or call_kwargs[0][1] == 20
