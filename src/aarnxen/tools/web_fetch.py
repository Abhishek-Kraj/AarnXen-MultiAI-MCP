"""Web fetch tool — retrieve and extract content from URLs."""

import logging
import re
from typing import Optional

import httpx
from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt

logger = logging.getLogger(__name__)

_JINA_PREFIX = "https://r.jina.ai/"


def _strip_html(html: str) -> str:
    """Basic HTML tag stripping for fallback mode."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:50_000]


async def web_fetch_handler(
    url: str,
    model: str = "",
    max_length: int = 10000,
    ctx: Context = None,
) -> dict:
    """Fetch a URL and extract its content as clean text.

    Uses Jina Reader API (free) for markdown extraction, falls back to
    raw HTML stripping. Optionally summarize with an AI model.

    Args:
        url: The URL to fetch.
        model: Optional model for AI summarization. Empty = raw content.
        max_length: Max characters to return (default 10000).
    """
    deps = ctx.request_context.lifespan_context
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    max_length = max(100, min(50_000, max_length))
    content = None
    source = None

    # Try Jina Reader API first (returns clean markdown)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                _JINA_PREFIX + url,
                headers={"Accept": "text/plain"},
            )
            if resp.status_code == 200 and len(resp.text) > 50:
                content = resp.text[:max_length]
                source = "jina"
    except Exception as exc:
        logger.debug("Jina fetch failed: %s", exc)

    # Fallback: raw fetch + HTML stripping
    if not content:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                content = _strip_html(resp.text)[:max_length]
                source = "raw"
        except Exception as exc:
            return {"error": f"Failed to fetch URL: {exc}", "url": url}

    response = {
        "url": url,
        "content": content,
        "length": len(content),
        "source": source,
    }

    if model and content:
        summary_prompt = f"Summarize the following web page content concisely:\n\n{content[:8000]}"
        try:
            registry = deps.registry
            provider, resolved_model = registry.resolve(model)
            fallbacks = registry.get_fallbacks(resolved_model)
            ai_resp = await call_with_retry(
                provider, resolved_model, summary_prompt,
                system_prompt="Provide a clear, concise summary of the web content.",
                temperature=0.3,
                fallback_providers=fallbacks,
                circuit_breaker=deps.circuit_breaker,
                max_retries=1,
            )
            response["summary"] = ai_resp.text
            response["summary_model"] = resolved_model
            deps.cost_tracker.record(
                provider.provider_name(), resolved_model,
                ai_resp.input_tokens, ai_resp.output_tokens,
            )
        except Exception as exc:
            logger.warning("AI summary failed: %s", exc)
            response["summary_error"] = str(exc)

    return response
