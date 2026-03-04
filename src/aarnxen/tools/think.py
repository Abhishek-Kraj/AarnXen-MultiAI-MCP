"""Deep reasoning tool — uses models with extended thinking."""

from mcp.server.fastmcp import Context


async def think_handler(
    prompt: str,
    model: str = "auto",
    depth: str = "medium",
    ctx: Context = None,
) -> dict:
    """Deep reasoning — ask a model to think step-by-step about a complex problem.

    Args:
        prompt: The complex question or problem to reason about.
        model: Model to use. "auto" picks the best reasoning model available.
        depth: Reasoning depth — "light", "medium", or "deep". Affects token budget.
    """
    deps = ctx.request_context.lifespan_context
    registry = deps["registry"]
    cost_tracker = deps["cost_tracker"]

    provider, resolved_model = registry.resolve(model)

    depth_configs = {
        "light": {"max_tokens": 2048, "temp": 0.3},
        "medium": {"max_tokens": 4096, "temp": 0.5},
        "deep": {"max_tokens": 8192, "temp": 0.7},
    }
    config = depth_configs.get(depth, depth_configs["medium"])

    system_prompt = (
        "You are a deep reasoning assistant. Think step-by-step through this problem. "
        "Break it into parts, consider edge cases, evaluate trade-offs, and arrive at a "
        "well-reasoned conclusion. Show your reasoning process clearly."
    )

    response = await provider.generate(
        prompt, resolved_model,
        system_prompt=system_prompt,
        temperature=config["temp"],
        max_tokens=config["max_tokens"],
    )

    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    return {
        "reasoning": response.text,
        "model": resolved_model,
        "provider": provider.provider_name(),
        "depth": depth,
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
        "latency_ms": round(response.latency_ms, 1),
    }
