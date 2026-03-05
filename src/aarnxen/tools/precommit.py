"""Pre-commit review tool — reviews git diffs before committing."""

from mcp.server.fastmcp import Context


async def precommit_handler(
    diff: str,
    model: str = "auto",
    ctx: Context = None,
) -> dict:
    """Review code changes before committing.

    Args:
        diff: The git diff to review.
        model: Model to use for review.
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    provider, resolved_model = registry.resolve(model)

    system_prompt = (
        "You are a senior code reviewer. Review this git diff for:\n"
        "1) Bugs and logic errors\n"
        "2) Security vulnerabilities\n"
        "3) Performance issues\n"
        "4) Style/consistency problems\n"
        "Return a structured review with a PASS/FAIL verdict at the end."
    )

    prompt = f"Review this diff:\n\n```diff\n{diff}\n```"

    response = await provider.generate(
        prompt, resolved_model,
        system_prompt=system_prompt,
        temperature=0.3,
    )

    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    verdict = "FAIL" if "FAIL" in response.text.upper() else "PASS"

    return {
        "result": response.text,
        "model": resolved_model,
        "provider": provider.provider_name(),
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
        "latency_ms": round(response.latency_ms, 1),
        "verdict": verdict,
    }
