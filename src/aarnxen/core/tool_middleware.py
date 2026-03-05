"""Tool execution middleware — rate limiting, guardrails, event emission, error handling.

Wraps tool handlers with cross-cutting concerns so individual handlers
stay focused on business logic.
"""

import logging
import time
from functools import wraps
from typing import Callable

from aarnxen.core.guardrails import Guardrails
from aarnxen.core.rate_limit import RateLimiter, RateLimitExceeded

logger = logging.getLogger(__name__)


def tool_wrapper(handler: Callable, tool_name: str) -> Callable:
    """Wrap a tool handler with rate limiting, guardrails, events, and error handling."""

    @wraps(handler)
    async def wrapped(*args, ctx=None, **kwargs):
        deps = ctx.request_context.lifespan_context
        event_bus = deps.event_bus
        rate_limiter = deps.rate_limiter
        guardrails = deps.guardrails

        # 1. Rate limiting
        if rate_limiter and not rate_limiter.allow(tool_name):
            remaining = rate_limiter.remaining(tool_name)
            await event_bus.emit("rate_limit", {"tool": tool_name})
            return {"error": f"Rate limit exceeded for '{tool_name}'. Try again shortly.", "remaining": remaining}

        # 2. Input guardrails — scan the first string arg (prompt/code/claim/diff)
        prompt_arg = None
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, str) and len(v) > 5:
                prompt_arg = v
                break

        if guardrails and prompt_arg:
            scan = guardrails.scan_input(prompt_arg)
            if not scan.passed:
                await event_bus.emit("guardrail_blocked", {
                    "tool": tool_name, "risk_score": scan.risk_score,
                    "detections": [d["type"] for d in scan.detections],
                })
                return {
                    "error": "Input blocked by guardrails",
                    "risk_score": scan.risk_score,
                    "detections": scan.detections,
                }

        # 3. Emit start event
        start = time.monotonic()
        await event_bus.emit("tool_start", {"tool": tool_name})

        # 4. Execute handler with error handling
        try:
            result = await handler(*args, ctx=ctx, **kwargs)
        except Exception as e:
            elapsed = round((time.monotonic() - start) * 1000, 1)
            await event_bus.emit("tool_error", {
                "tool": tool_name, "error": type(e).__name__,
                "message": str(e)[:200], "latency_ms": elapsed,
            })
            logger.error("%s failed: %s", tool_name, e)
            return {
                "error": f"{tool_name} failed: {type(e).__name__}",
                "message": str(e)[:500],
            }

        # 5. Emit completion event
        elapsed = round((time.monotonic() - start) * 1000, 1)
        await event_bus.emit("tool_complete", {"tool": tool_name, "latency_ms": elapsed})

        return result

    return wrapped
