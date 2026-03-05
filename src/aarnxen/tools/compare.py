"""Side-by-side model comparison tool."""

import asyncio

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt, sanitize_system_prompt, truncate_response


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

    prompt = validate_prompt(prompt)
    if system_prompt:
        system_prompt = sanitize_system_prompt(system_prompt)

    # Resolve models
    top = registry.get_top_n_models(2)
    if not model_a:
        model_a = top[0][1] if len(top) > 0 else "auto"
    if not model_b:
        model_b = top[1][1] if len(top) > 1 else "auto"

    async def query(model: str, label: str) -> dict:
        provider, resolved = registry.resolve(model)
        fallbacks = registry.get_fallbacks(resolved)
        response = await call_with_retry(
            provider, resolved, prompt,
            system_prompt=system_prompt or None,
            temperature=temperature,
            max_retries=2,
            fallback_providers=fallbacks,
            circuit_breaker=deps.circuit_breaker,
        )
        cost_entry = cost_tracker.record(provider.provider_name(), resolved, response.input_tokens, response.output_tokens)
        return {
            "label": label,
            "model": resolved,
            "provider": provider.provider_name(),
            "response": truncate_response(response.text),
            "tokens": {"input": response.input_tokens, "output": response.output_tokens},
            "cost_usd": round(cost_entry.cost_usd, 6),
            "latency_ms": round(response.latency_ms, 1),
        }

    if ctx:
        await ctx.report_progress(0, 2, "Starting comparison...")
    results = await asyncio.gather(
        query(model_a, "Model A"),
        query(model_b, "Model B"),
        return_exceptions=True,
    )

    errors = []
    model_results = []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(type(r).__name__))
        else:
            model_results.append(r)

    if ctx:
        await ctx.report_progress(2, 2, "Both models responded")

    response = {"prompt": prompt}
    if len(model_results) >= 2:
        response["model_a"] = model_results[0]
        response["model_b"] = model_results[1]
    elif len(model_results) == 1:
        response["model_a"] = model_results[0]
        response["model_b"] = None
        response["errors"] = errors
    else:
        response["model_a"] = None
        response["model_b"] = None
        response["errors"] = errors
    response["instruction"] = "Compare the two responses. Which is better and why? Highlight key differences."
    return response
