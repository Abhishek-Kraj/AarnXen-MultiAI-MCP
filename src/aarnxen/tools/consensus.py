"""Parallel multi-model consensus tool."""

import asyncio

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry


async def consensus_handler(
    prompt: str,
    models: str = "auto",
    temperature: float = 0.7,
    system_prompt: str = "",
    ctx: Context = None,
) -> dict:
    """Query multiple AI models in parallel and get all responses for synthesis.

    Args:
        prompt: Question or task for all models.
        models: Comma-separated models, e.g. "gemini-2.5-flash,groq,openai". Use "auto" for top 3.
        temperature: Creativity level (0.0-1.0).
        system_prompt: Optional system instructions applied to all models.
    """
    deps = ctx.request_context.lifespan_context
    registry = deps["registry"]
    cache = deps["cache"]
    cost_tracker = deps["cost_tracker"]

    # Resolve models
    if models == "auto":
        model_pairs = registry.get_top_n_models(3)
    else:
        model_pairs = []
        for m in models.split(","):
            m = m.strip()
            provider, resolved = registry.resolve(m)
            model_pairs.append((provider.provider_name(), resolved))

    labels = ["Model A", "Model B", "Model C", "Model D", "Model E"]

    async def query_one(idx: int, provider_name: str, model_id: str) -> dict:
        provider, resolved = registry.resolve(model_id if model_id != "default" else provider_name)

        # Check cache
        if cache:
            cached = cache.get(provider.provider_name(), resolved, prompt, system_prompt, temperature)
            if cached:
                cost_tracker.record(provider.provider_name(), resolved, cached.input_tokens, cached.output_tokens, cached=True)
                return {
                    "label": labels[idx] if idx < len(labels) else f"Model {idx+1}",
                    "model": resolved,
                    "provider": provider.provider_name(),
                    "response": cached.text,
                    "tokens": {"input": cached.input_tokens, "output": cached.output_tokens},
                    "cost_usd": 0.0,
                    "cached": True,
                }

        fallbacks = registry.get_fallbacks(resolved)
        response = await call_with_retry(
            provider, resolved, prompt,
            system_prompt=system_prompt or None,
            temperature=temperature,
            fallback_providers=fallbacks,
            max_retries=2,
        )

        if cache:
            cache.put(provider.provider_name(), resolved, prompt, system_prompt, temperature, response)
        cost_entry = cost_tracker.record(provider.provider_name(), resolved, response.input_tokens, response.output_tokens)

        return {
            "label": labels[idx] if idx < len(labels) else f"Model {idx+1}",
            "model": resolved,
            "provider": provider.provider_name(),
            "response": response.text,
            "tokens": {"input": response.input_tokens, "output": response.output_tokens},
            "cost_usd": round(cost_entry.cost_usd, 6),
            "cached": False,
            "latency_ms": round(response.latency_ms, 1),
        }

    # Fire all queries in parallel
    total = len(model_pairs)
    if ctx:
        await ctx.report_progress(0, total, "Starting consensus...")
    tasks = [query_one(i, pname, mid) for i, (pname, mid) in enumerate(model_pairs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if ctx:
        await ctx.report_progress(total, total, "All models responded")

    responses = []
    errors = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            errors.append({"index": i, "error": str(r)})
        else:
            responses.append(r)

    total_cost = sum(r["cost_usd"] for r in responses)
    total_tokens = sum(r["tokens"]["input"] + r["tokens"]["output"] for r in responses)

    return {
        "prompt": prompt,
        "model_count": len(model_pairs),
        "responses": responses,
        "errors": errors if errors else None,
        "summary": {
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "succeeded": len(responses),
            "failed": len(errors),
        },
        "instruction": "Analyze the responses above. Identify agreements and disagreements. Synthesize the best answer.",
    }
