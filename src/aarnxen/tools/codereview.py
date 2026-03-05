"""Code review tool — reviews code with chosen model."""

from mcp.server.fastmcp import Context


async def codereview_handler(
    code: str,
    language: str = "auto",
    model: str = "auto",
    focus: str = "general",
    ctx: Context = None,
) -> dict:
    """Review code for bugs, style, performance, and security issues.

    Args:
        code: The code to review (paste directly or file contents).
        language: Programming language. "auto" to detect.
        model: Model to use for review.
        focus: Review focus — "general", "security", "performance", or "bugs".
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

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

    response = await provider.generate(
        prompt, resolved_model,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=4096,
    )

    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    return {
        "review": response.text,
        "model": resolved_model,
        "provider": provider.provider_name(),
        "focus": focus,
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
    }
