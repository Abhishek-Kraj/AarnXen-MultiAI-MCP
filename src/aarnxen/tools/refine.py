"""Refine tool — Self-Refine iterative improvement."""

import logging

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt

logger = logging.getLogger(__name__)


async def refine_handler(
    prompt: str,
    model: str = "auto",
    critic: str = "auto",
    iterations: int = 2,
    ctx: Context = None,
) -> dict:
    """Generate → critic feedback → refine, for N iterations.

    Uses Self-Refine technique (~20% improvement per iteration).

    Args:
        prompt: The task/question to generate and iteratively refine.
        model: Model for generation and refinement.
        critic: Model for critique (can be different for diverse feedback).
        iterations: Number of refine cycles (1-5).
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    prompt = validate_prompt(prompt)
    iterations = max(1, min(5, iterations))

    prov_gen, model_gen = registry.resolve(model)
    if critic == "auto" and model == "auto":
        # Try to pick a different model for the critic
        models = registry.get_top_n_models(2)
        if len(models) >= 2:
            prov_crit, model_crit = registry.resolve(models[1][1])
        else:
            prov_crit, model_crit = prov_gen, model_gen
    else:
        prov_crit, model_crit = registry.resolve(critic)

    total_cost = 0.0
    total_tokens = 0

    async def _call(prov, mod, prompt_text, sys="", temp=0.7):
        nonlocal total_cost, total_tokens
        fallbacks = registry.get_fallbacks(mod)
        resp = await call_with_retry(
            prov, mod, prompt_text,
            system_prompt=sys, temperature=temp,
            fallback_providers=fallbacks,
            circuit_breaker=deps.circuit_breaker, max_retries=1,
        )
        cost = cost_tracker.record(prov.provider_name(), mod, resp.input_tokens, resp.output_tokens)
        total_cost += cost.cost_usd
        total_tokens += resp.input_tokens + resp.output_tokens
        return resp.text

    # Step 1: Initial generation
    current = await _call(
        prov_gen, model_gen, prompt,
        sys="Provide a thorough, high-quality response.",
    )

    history = [{"iteration": 0, "type": "generation", "model": model_gen, "text": current}]

    for i in range(iterations):
        # Critique
        critique = await _call(
            prov_crit, model_crit,
            f"Original task: {prompt}\n\nCurrent response:\n{current}\n\n"
            f"Provide specific, actionable feedback to improve this response. "
            f"Note what's good and what needs improvement. Be constructive and precise.",
            sys="You are a constructive critic. Give specific, actionable feedback.",
            temp=0.4,
        )
        history.append({"iteration": i + 1, "type": "critique", "model": model_crit, "text": critique})

        # Refine based on critique
        current = await _call(
            prov_gen, model_gen,
            f"Original task: {prompt}\n\n"
            f"Your previous response:\n{current}\n\n"
            f"Critic feedback:\n{critique}\n\n"
            f"Revise your response, addressing the feedback while keeping what was good:",
            sys="Improve your response based on the critique. Keep strengths, fix weaknesses.",
        )
        history.append({"iteration": i + 1, "type": "refinement", "model": model_gen, "text": current})

    return {
        "prompt": prompt,
        "models": {"generator": model_gen, "critic": model_crit},
        "iterations": iterations,
        "final_response": current,
        "history": history,
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
    }
