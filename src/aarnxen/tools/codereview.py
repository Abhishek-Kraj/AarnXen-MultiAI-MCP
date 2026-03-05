"""Code review tool — reviews code with chosen model."""

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt, truncate_response


async def codereview_handler(
    code: str,
    language: str = "auto",
    model: str = "auto",
    focus: str = "general",
    ctx: Context = None,
) -> dict:
    """Review code for bugs, style, performance, and security issues."""
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    code = validate_prompt(code, max_length=200_000)

    provider, resolved_model = registry.resolve(model)

    focus_instructions = {
        "general": "Review for bugs, style, performance, and security. Prioritize issues by severity.",
        "security": "Focus on security vulnerabilities: injection, XSS, auth issues, data exposure, OWASP Top 10.",
        "performance": "Focus on performance: algorithmic complexity, memory usage, unnecessary allocations, N+1 queries.",
        "bugs": "Focus on logic errors, edge cases, null/undefined handling, off-by-one errors, race conditions.",
    }

    system_prompt = (
        f"You are an expert code reviewer. {focus_instructions.get(focus, focus_instructions['general'])}\n"
        "Format your review as:\n"
        "1. **Critical Issues** (must fix)\n"
        "2. **Warnings** (should fix)\n"
        "3. **Suggestions** (nice to have)\n"
        "4. **Summary** (overall assessment)\n"
        "Be specific — reference line numbers and provide fix suggestions."
    )

    lang_hint = f" (Language: {language})" if language != "auto" else ""
    prompt = f"Review this code{lang_hint}:\n\n```\n{code}\n```"

    fallbacks = registry.get_fallbacks(resolved_model)
    response = await call_with_retry(
        provider, resolved_model, prompt,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=4096,
        fallback_providers=fallbacks,
        circuit_breaker=deps.circuit_breaker,
        max_retries=2,
    )

    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    return {
        "review": truncate_response(response.text),
        "model": resolved_model,
        "provider": provider.provider_name(),
        "focus": focus,
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
    }
