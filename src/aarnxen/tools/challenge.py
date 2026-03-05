"""Devil's advocate tool — critically evaluates claims and approaches."""

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt, truncate_response


async def challenge_handler(
    claim: str,
    evidence: str = "",
    model: str = "auto",
    ctx: Context = None,
) -> dict:
    """Critically evaluate a claim or approach — find flaws, weaknesses, and blind spots."""
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    claim = validate_prompt(claim)

    provider, resolved_model = registry.resolve(model)

    system_prompt = (
        "You are a critical analyst. Your job is to find flaws, weaknesses, and blind spots "
        "in the given claim or approach. Be thorough but constructive. "
        "Structure your response as:\n"
        "1) Key Weaknesses\n"
        "2) Missing Considerations\n"
        "3) Counter-Arguments\n"
        "4) Verdict (STRONG/MODERATE/WEAK)"
    )

    prompt = f"Claim: {claim}"
    if evidence:
        prompt += f"\n\nSupporting evidence: {evidence}"

    fallbacks = registry.get_fallbacks(resolved_model)
    response = await call_with_retry(
        provider, resolved_model, prompt,
        system_prompt=system_prompt,
        temperature=0.4,
        fallback_providers=fallbacks,
        circuit_breaker=deps.circuit_breaker,
        max_retries=2,
    )

    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    text_upper = response.text.upper()
    if "STRONG" in text_upper:
        verdict = "STRONG"
    elif "WEAK" in text_upper:
        verdict = "WEAK"
    else:
        verdict = "MODERATE"

    return {
        "result": truncate_response(response.text),
        "model": resolved_model,
        "provider": provider.provider_name(),
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
        "latency_ms": round(response.latency_ms, 1),
        "verdict": verdict,
    }
