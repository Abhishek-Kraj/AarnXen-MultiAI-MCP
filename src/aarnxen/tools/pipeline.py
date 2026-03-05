"""Pipeline meta-tool — chain AarnXen tools in sequence."""

import json
import logging

from mcp.server.fastmcp import Context

from aarnxen.core.validation import validate_json_input
from aarnxen.core.tool_middleware import tool_wrapper

logger = logging.getLogger(__name__)

# Map tool names to their (module_path, handler_name) for lazy loading
_TOOL_SOURCES = {
    "chat": ("aarnxen.tools.chat", "chat_handler"),
    "think": ("aarnxen.tools.think", "think_handler"),
    "challenge": ("aarnxen.tools.challenge", "challenge_handler"),
    "codereview": ("aarnxen.tools.codereview", "codereview_handler"),
    "precommit": ("aarnxen.tools.precommit", "precommit_handler"),
    "web_search": ("aarnxen.tools.web_search", "web_search_handler"),
    "web_fetch": ("aarnxen.tools.web_fetch", "web_fetch_handler"),
    "debate": ("aarnxen.tools.debate", "debate_handler"),
    "verify": ("aarnxen.tools.verify", "verify_handler"),
    "jury": ("aarnxen.tools.jury", "jury_handler"),
    "refine": ("aarnxen.tools.refine", "refine_handler"),
}

# Cache of wrapped handlers (loaded on first use)
_wrapped_cache: dict = {}


def _get_wrapped_handler(tool_name: str):
    """Get a middleware-wrapped handler for a tool, caching on first load."""
    if tool_name not in _wrapped_cache:
        module_path, handler_name = _TOOL_SOURCES[tool_name]
        module = __import__(module_path, fromlist=[handler_name])
        handler = getattr(module, handler_name)
        _wrapped_cache[tool_name] = tool_wrapper(handler, f"pipeline.{tool_name}")
    return _wrapped_cache[tool_name]


TOOL_REGISTRY = _TOOL_SOURCES  # Exposed for tests and tool listing


def _extract_text(result: dict) -> str:
    """Extract the main text content from any tool result."""
    for key in ("response", "result", "reasoning", "review", "text"):
        if key in result and isinstance(result[key], str):
            return result[key]
    return json.dumps(result)


async def pipeline_handler(
    steps: str,
    ctx: Context = None,
) -> dict:
    """Execute a pipeline of AarnXen tools. Each step's output feeds the next.

    Args:
        steps: JSON array of tool steps. Each step has "tool" and "args".
               Use "$PREV" in any arg value to reference the previous step's output.
               Example: [
                   {"tool": "think", "args": {"prompt": "Analyze microservices vs monolith", "depth": "deep"}},
                   {"tool": "challenge", "args": {"claim": "$PREV"}}
               ]
    """
    try:
        step_list = validate_json_input(steps)
    except ValueError as exc:
        return {"isError": True, "message": str(exc)}

    results = []
    prev_text = ""

    for i, step in enumerate(step_list):
        tool_name = step.get("tool", "")
        args = step.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            return {
                "isError": True,
                "message": f"Unknown tool '{tool_name}' in step {i+1}",
                "available_tools": list(TOOL_REGISTRY.keys()),
            }

        # Replace $PREV in all string args
        resolved_args = {}
        for k, v in args.items():
            if isinstance(v, str) and "$PREV" in v:
                resolved_args[k] = v.replace("$PREV", prev_text)
            else:
                resolved_args[k] = v

        # Get wrapped handler (with middleware: rate limiting, guardrails, events)
        handler = _get_wrapped_handler(tool_name)

        # Add ctx to args
        resolved_args["ctx"] = ctx

        try:
            result = await handler(**resolved_args)
        except Exception as exc:
            return {
                "isError": True,
                "message": f"Step {i+1} ({tool_name}) failed: {exc}",
                "completed_steps": i,
                "partial_results": results,
            }

        results.append({"step": i + 1, "tool": tool_name, "result": result})
        prev_text = _extract_text(result)

    return {
        "pipeline_results": results,
        "steps_completed": len(results),
        "final_output": prev_text,
    }
