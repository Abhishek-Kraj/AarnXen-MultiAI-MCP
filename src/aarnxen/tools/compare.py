"""Side-by-side model comparison tool."""

import asyncio

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry


async def compare_handler(
    prompt: str,
    model_a: str = "",
    model_b: str = "",
    temperature: float = 0.7,
    system_prompt: str = "",
    ctx: Context = None,
) -> dict:
    """Compare two AI models side-by-side on the same prompt.

    Args:
        prompt: Question or task for both models.
        model_a: First model (name or provider). Defaults to highest-priority.
        model_b: Second model (name or provider). Defaults to second-highest.
        temperature: Creativity level (0.0-1.0).
        system_prompt: Optional system instructions.
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    # Resolve models
    top = registry.get_top_n_models(2)
    if not model_a:
        model_a = top[0][1] if len(top) > 0 else "auto"
    if not model_b:
        model_b = top[1][1] if len(top) > 1 else "auto"

    async def query(model: str, label: str) -> dict:
        provider, resolved = registry.resolve(model)
        response = await call_with_retry(
            provider, resolved, prompt,
            system_prompt=system_prompt or None,
            temperature=temperature,
            max_retries=2,
        )
        cost_entry = cost_tracker.record(provider.provider_name(), resolved, response.input_tokens, response.output_tokens)
        return {
            "label": label,
            "model": resolved,
            "provider": provider.provider_name(),
            "response": response.text,
            "tokens": {"input": response.input_tokens, "output": response.output_tokens},
            "cost_usd": round(cost_entry.cost_usd, 6),
            "latency_ms": round(response.latency_ms, 1),
        }

    if ctx:
        await ctx.report_progress(0, 2, "Starting comparison...")
    result_a, result_b = await asyncio.gather(
        query(model_a, "Model A"),
        query(model_b, "Model B"),
        return_exceptions=False,
    )
    if ctx:
        await ctx.report_progress(2, 2, "Both models responded")

    return {
        "prompt": prompt,
        "model_a": result_a,
        "model_b": result_b,
        "instruction": "Compare the two responses. Which is better and why? Highlight key differences.",
    }
