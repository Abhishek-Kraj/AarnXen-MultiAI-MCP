"""Cost summary tool — shows session spending and cache stats."""

from mcp.server.fastmcp import Context


async def costs_handler(ctx: Context = None) -> dict:
    """Show session cost summary — total spending, per-model breakdown, and cache stats."""
    deps = ctx.request_context.lifespan_context
    cost_tracker = deps["cost_tracker"]
    cache = deps["cache"]

    result = cost_tracker.summary()
    if cache:
        result["cache"] = cache.stats()
    return result
