"""Web search tool — DuckDuckGo search with optional AI summarization."""

import logging

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt

logger = logging.getLogger(__name__)


async def web_search_handler(
    query: str,
    max_results: int = 5,
    model: str = "",
    ctx: Context = None,
) -> dict:
    """Search the web via DuckDuckGo and optionally summarize results with an AI model.

    Args:
        query: Search query string.
        max_results: Number of results to return (1-20).
        model: Optional model for AI summarization of results. Empty = raw results only.
    """
    deps = ctx.request_context.lifespan_context
    query = validate_prompt(query)
    max_results = max(1, min(20, max_results))

    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return {"error": f"Search failed: {exc}", "query": query}

    results = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
    ]

    response = {"query": query, "results": results, "count": len(results)}

    if model and results:
        summary_prompt = (
            f"Based on these web search results for '{query}', provide a concise summary:\n\n"
            + "\n".join(f"- {r['title']}: {r['snippet']}" for r in results)
        )
        try:
            registry = deps.registry
            provider, resolved_model = registry.resolve(model)
            fallbacks = registry.get_fallbacks(resolved_model)
            ai_resp = await call_with_retry(
                provider, resolved_model, summary_prompt,
                system_prompt="Summarize search results concisely. Cite sources.",
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
