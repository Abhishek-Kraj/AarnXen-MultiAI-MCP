"""Deep reasoning tool — uses models with extended thinking."""

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt, truncate_response

DEPTH_CONFIG = {
    "light": {
        "temperature": 0.5,
        "max_tokens": 1000,
        "system_prompt": (
            "Provide a brief, focused analysis. Hit the key points quickly. "
            "Skip obvious details — focus on what matters most."
        ),
    },
    "medium": {
        "temperature": 0.7,
        "max_tokens": 4000,
        "system_prompt": (
            "Analyze this step by step. Consider multiple angles. "
            "Structure your response with clear sections. "
            "Identify trade-offs and make a recommendation."
        ),
    },
    "deep": {
        "temperature": 0.8,
        "max_tokens": 8000,
        "system_prompt": (
            "Perform a thorough, exhaustive analysis. Consider every angle: "
            "assumptions, edge cases, second-order effects, historical parallels. "
            "Challenge your own reasoning. Present both the strongest argument "
            "and the strongest counter-argument. Structure with: "
            "1) Problem decomposition 2) Analysis of each component "
            "3) Synthesis 4) Risks and unknowns 5) Final recommendation with confidence level."
        ),
    },
}

DEPTH_ALIASES = {
    "l": "light", "1": "light", "quick": "light", "shallow": "light",
    "m": "medium", "2": "medium", "normal": "medium", "default": "medium",
    "d": "deep", "3": "deep", "thorough": "deep", "extended": "deep",
}


def _resolve_depth(depth: str) -> str:
    depth = depth.lower().strip()
    return DEPTH_ALIASES.get(depth, depth) if depth not in DEPTH_CONFIG else depth


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
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    prompt = validate_prompt(prompt)

    provider, resolved_model = registry.resolve(model)

    depth = _resolve_depth(depth)
    config = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])

    system_prompt = config["system_prompt"]

    fallbacks = registry.get_fallbacks(resolved_model)
    response = await call_with_retry(
        provider, resolved_model, prompt,
        system_prompt=system_prompt,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        fallback_providers=fallbacks,
        circuit_breaker=deps.circuit_breaker,
        max_retries=2,
    )

    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    return {
        "reasoning": truncate_response(response.text),
        "model": resolved_model,
        "provider": provider.provider_name(),
        "depth": depth,
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
        "latency_ms": round(response.latency_ms, 1),
    }
