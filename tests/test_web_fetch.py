"""Tests for web fetch tool."""

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


def _mock_async_client(get_fn=None, get_return=None):
    """Create a mock httpx.AsyncClient that works with async with."""
    client_instance = MagicMock()
    if get_fn is not None:
        client_instance.get = get_fn
    else:
        client_instance.get = AsyncMock(return_value=get_return)
    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
    client_instance.__aexit__ = AsyncMock(return_value=False)
    return client_instance


@pytest.mark.asyncio
async def test_web_fetch_jina_success():
    ctx, deps = _make_ctx()

    jina_resp = MagicMock()
    jina_resp.status_code = 200
    jina_resp.text = "# Page Title\n\nSome markdown content from Jina Reader"

    mock_client = _mock_async_client(get_return=jina_resp)

    with patch("aarnxen.tools.web_fetch.httpx.AsyncClient", return_value=mock_client):
        from aarnxen.tools.web_fetch import web_fetch_handler
        result = await web_fetch_handler("https://example.com", ctx=ctx)

    assert result["url"] == "https://example.com"
    assert "markdown content" in result["content"]
    assert result["source"] == "jina"


@pytest.mark.asyncio
async def test_web_fetch_invalid_url():
    ctx, deps = _make_ctx()

    from aarnxen.tools.web_fetch import web_fetch_handler
    result = await web_fetch_handler("not-a-url", ctx=ctx)

    assert "error" in result


@pytest.mark.asyncio
async def test_web_fetch_fallback_to_raw():
    ctx, deps = _make_ctx()

    jina_resp = MagicMock()
    jina_resp.status_code = 500
    jina_resp.text = ""

    raw_resp = MagicMock()
    raw_resp.status_code = 200
    raw_resp.text = "<html><body><p>Hello World</p></body></html>"
    raw_resp.raise_for_status = MagicMock()

    call_count = 0

    async def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return jina_resp
        return raw_resp

    mock_client = _mock_async_client(get_fn=mock_get)

    with patch("aarnxen.tools.web_fetch.httpx.AsyncClient", return_value=mock_client):
        from aarnxen.tools.web_fetch import web_fetch_handler
        result = await web_fetch_handler("https://example.com", ctx=ctx)

    assert result["source"] == "raw"
    assert "Hello World" in result["content"]
