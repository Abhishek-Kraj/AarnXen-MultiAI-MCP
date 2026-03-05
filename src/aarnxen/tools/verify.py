"""Verify tool — Chain of Verification (CoVe) for fact-checking claims."""

import logging

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt

logger = logging.getLogger(__name__)


async def verify_handler(
    claim: str,
    model: str = "auto",
    web_check: bool = False,
    ctx: Context = None,
) -> dict:
    """Verify a claim using Chain of Verification (CoVe).

    4-step process:
    1. Draft an initial answer/assessment
    2. Generate verification questions
    3. Answer each question independently
    4. Produce a revised, verified answer

    Optionally uses web search for grounding (web_check=True).

    Args:
        claim: The claim or statement to verify.
        model: Model to use for verification steps.
        web_check: If True, search the web to ground verification.
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    claim = validate_prompt(claim)
    provider, resolved_model = registry.resolve(model)
    fallbacks = registry.get_fallbacks(resolved_model)
    total_cost = 0.0
    total_tokens = 0

    async def _call(prompt, sys="", temp=0.3):
        nonlocal total_cost, total_tokens
        resp = await call_with_retry(
            provider, resolved_model, prompt,
            system_prompt=sys, temperature=temp,
            fallback_providers=fallbacks,
            circuit_breaker=deps.circuit_breaker, max_retries=1,
        )
        cost = cost_tracker.record(provider.provider_name(), resolved_model, resp.input_tokens, resp.output_tokens)
        total_cost += cost.cost_usd
        total_tokens += resp.input_tokens + resp.output_tokens
        return resp.text

    # Step 1: Draft initial assessment
    draft = await _call(
        f"Assess this claim and provide your initial analysis:\n\n{claim}",
        sys="You are a fact-checker. Provide a thorough initial assessment.",
    )

    # Step 2: Generate verification questions
    questions_text = await _call(
        f"Claim: {claim}\n\nInitial assessment: {draft}\n\n"
        f"Generate 3-5 specific verification questions that would help confirm or deny this claim. "
        f"Focus on checkable facts. Return one question per line, numbered.",
        sys="Generate precise verification questions.",
    )

    questions = [q.strip().lstrip("0123456789.) ") for q in questions_text.strip().split("\n") if q.strip() and len(q.strip()) > 10][:5]

    # Step 3: Answer each question independently (optionally with web search)
    verifications = []
    for q in questions:
        web_context = ""
        if web_check:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(q, max_results=3))
                    web_context = "\n".join(f"- {r.get('body', '')}" for r in results)
            except Exception:
                pass

        prompt = f"Answer this verification question independently (without referring to the original claim):\n\n{q}"
        if web_context:
            prompt += f"\n\nRelevant web search results:\n{web_context}"

        answer = await _call(prompt, sys="Answer factually and precisely.")
        verifications.append({"question": q, "answer": answer, "web_grounded": bool(web_context)})

    # Step 4: Final revised answer
    verif_text = "\n".join(f"Q: {v['question']}\nA: {v['answer']}" for v in verifications)
    revised = await _call(
        f"Original claim: {claim}\n\n"
        f"Initial draft: {draft}\n\n"
        f"Verification results:\n{verif_text}\n\n"
        f"Based on the verification, provide a REVISED assessment. "
        f"Rate confidence as HIGH/MEDIUM/LOW. Note any contradictions found.",
        sys="Synthesize verification results into a final assessment with confidence rating.",
    )

    # Extract confidence from the revised text
    confidence = "MEDIUM"
    revised_lower = revised.lower()
    if "high" in revised_lower and "confidence" in revised_lower:
        confidence = "HIGH"
    elif "low" in revised_lower and "confidence" in revised_lower:
        confidence = "LOW"

    return {
        "claim": claim,
        "model": resolved_model,
        "draft_assessment": draft,
        "verification_questions": [v["question"] for v in verifications],
        "verifications": verifications,
        "revised_assessment": revised,
        "confidence": confidence,
        "web_grounded": web_check,
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
    }
