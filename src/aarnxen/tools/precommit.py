"""Pre-commit review tool — reviews git diffs before committing."""

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt, truncate_response


async def precommit_handler(
    diff: str,
    model: str = "auto",
    ctx: Context = None,
) -> dict:
    """Review code changes before committing."""
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    diff = validate_prompt(diff, max_length=200_000)

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

    fallbacks = registry.get_fallbacks(resolved_model)
    response = await call_with_retry(
        provider, resolved_model, prompt,
        system_prompt=system_prompt,
        temperature=0.3,
        fallback_providers=fallbacks,
        circuit_breaker=deps.circuit_breaker,
        max_retries=2,
    )

    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    verdict = "FAIL" if "FAIL" in response.text.upper() else "PASS"

    return {
        "result": truncate_response(response.text),
        "model": resolved_model,
        "provider": provider.provider_name(),
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
        "latency_ms": round(response.latency_ms, 1),
        "verdict": verdict,
    }
