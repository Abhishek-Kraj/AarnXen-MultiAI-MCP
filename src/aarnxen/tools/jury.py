"""Jury tool — N models independently score content, aggregate votes."""

import asyncio
import json
import logging

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt

logger = logging.getLogger(__name__)

CRITERIA_PROMPTS = {
    "general": "Rate the quality, accuracy, and usefulness of this content.",
    "code": "Rate this code for correctness, readability, performance, and security.",
    "writing": "Rate this writing for clarity, coherence, grammar, and persuasiveness.",
    "factual": "Rate the factual accuracy and reliability of this content.",
}


async def jury_handler(
    content: str,
    criteria: str = "general",
    models: str = "auto",
    num_jurors: int = 3,
    ctx: Context = None,
) -> dict:
    """N AI models independently score content and aggregate votes.

    Args:
        content: The content to evaluate.
        criteria: Evaluation criteria — "general", "code", "writing", "factual".
        models: Comma-separated models, or "auto" for top N.
        num_jurors: Number of juror models (1-5).
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    content = validate_prompt(content)
    num_jurors = max(1, min(5, num_jurors))
    criteria_prompt = CRITERIA_PROMPTS.get(criteria, CRITERIA_PROMPTS["general"])

    if models == "auto":
        model_pairs = registry.get_top_n_models(num_jurors)
    else:
        model_pairs = []
        for m in models.split(","):
            m = m.strip()
            p, r = registry.resolve(m)
            model_pairs.append((p.provider_name(), r))

    scoring_prompt = (
        f"{criteria_prompt}\n\n"
        f"Content to evaluate:\n{content[:5000]}\n\n"
        f"Respond with a JSON object containing:\n"
        f'- "score": integer 1-10\n'
        f'- "reasoning": brief explanation (2-3 sentences)\n'
        f'- "strengths": list of strengths\n'
        f'- "weaknesses": list of weaknesses\n'
        f"Return ONLY the JSON, no other text."
    )

    async def judge(provider_name: str, model_id: str) -> dict:
        provider, resolved = registry.resolve(model_id if model_id != "default" else provider_name)
        fallbacks = registry.get_fallbacks(resolved)
        resp = await call_with_retry(
            provider, resolved, scoring_prompt,
            system_prompt="You are a fair evaluator. Score content objectively. Respond only with JSON.",
            temperature=0.3,
            fallback_providers=fallbacks,
            circuit_breaker=deps.circuit_breaker,
            max_retries=1,
        )
        cost = cost_tracker.record(provider.provider_name(), resolved, resp.input_tokens, resp.output_tokens)

        # Parse the JSON response
        try:
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            score = max(1, min(10, int(parsed.get("score", 5))))
        except (json.JSONDecodeError, ValueError):
            score = 5
            parsed = {"score": 5, "reasoning": resp.text[:200], "strengths": [], "weaknesses": []}

        return {
            "model": resolved,
            "provider": provider.provider_name(),
            "score": score,
            "reasoning": parsed.get("reasoning", ""),
            "strengths": parsed.get("strengths", []),
            "weaknesses": parsed.get("weaknesses", []),
            "cost_usd": round(cost.cost_usd, 6),
            "latency_ms": round(resp.latency_ms, 1),
        }

    tasks = [judge(pname, mid) for pname, mid in model_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verdicts = []
    errors = []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
        else:
            verdicts.append(r)

    scores = [v["score"] for v in verdicts]
    avg_score = sum(scores) / len(scores) if scores else 0
    total_cost = sum(v["cost_usd"] for v in verdicts)

    return {
        "criteria": criteria,
        "juror_count": len(verdicts),
        "verdicts": verdicts,
        "errors": errors if errors else None,
        "aggregate": {
            "average_score": round(avg_score, 1),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "consensus": max(scores) - min(scores) <= 2 if len(scores) >= 2 else True,
        },
        "total_cost_usd": round(total_cost, 6),
    }
