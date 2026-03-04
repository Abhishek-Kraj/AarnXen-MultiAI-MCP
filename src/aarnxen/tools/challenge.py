"""Devil's advocate tool — critically evaluates claims and approaches."""

from mcp.server.fastmcp import Context


async def challenge_handler(
    claim: str,
    evidence: str = "",
    model: str = "auto",
    ctx: Context = None,
) -> dict:
    """Critically evaluate a claim or approach — find flaws, weaknesses, and blind spots.

    Args:
        claim: The claim or approach to challenge.
        evidence: Optional supporting evidence to also evaluate.
        model: Model to use for analysis.
    """
    deps = ctx.request_context.lifespan_context
    registry = deps["registry"]
    cost_tracker = deps["cost_tracker"]

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

    response = await provider.generate(
        prompt, resolved_model,
        system_prompt=system_prompt,
        temperature=0.4,
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
        "result": response.text,
        "model": resolved_model,
        "provider": provider.provider_name(),
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
        "latency_ms": round(response.latency_ms, 1),
        "verdict": verdict,
    }
