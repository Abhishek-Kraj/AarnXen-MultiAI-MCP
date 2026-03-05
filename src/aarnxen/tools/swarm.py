"""Multi-agent swarm — launch N parallel agents on sub-tasks.

Each agent gets its own prompt and model, runs independently,
and results are gathered for synthesis. Ideal for:
- Breaking a complex problem into sub-tasks
- Getting N independent analyses of the same topic
- Parallel code review across different focus areas
- Brainstorming with diverse model perspectives

Kimi K2 Thinking (256K ctx, 200-300 tool calls) and GLM-5 (744B)
are ideal for complex agent tasks. Ollama Cloud subscription
means launching 100 agents costs $0 in per-token fees.
"""

import asyncio
import json
import logging
import time

from mcp.server.fastmcp import Context

from aarnxen.core.errors import generation_failed

logger = logging.getLogger(__name__)

# Default concurrency limit to avoid overwhelming providers
DEFAULT_CONCURRENCY = 10
MAX_AGENTS = 100


async def swarm_handler(
    tasks: str,
    model: str = "auto",
    concurrency: int = DEFAULT_CONCURRENCY,
    ctx: Context = None,
) -> dict:
    """Launch multiple AI agents in parallel on different sub-tasks.

    Args:
        tasks: JSON array of task objects. Each has "prompt" and optional
               "model", "system_prompt", "label".
               Example: [
                   {"prompt": "Analyze security of auth module", "label": "Security"},
                   {"prompt": "Review performance of DB queries", "label": "Performance"},
                   {"prompt": "Check error handling patterns", "label": "Error Handling"}
               ]
        model: Default model for all agents. Override per-task with task-level "model".
        concurrency: Max agents running simultaneously (default 10, max 100).
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker
    cache = deps.cache

    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError as exc:
        return {"isError": True, "message": f"Invalid JSON: {exc}"}

    if not isinstance(task_list, list) or not task_list:
        return {"isError": True, "message": "Tasks must be a non-empty JSON array"}

    if len(task_list) > MAX_AGENTS:
        return {"isError": True, "message": f"Max {MAX_AGENTS} agents allowed, got {len(task_list)}"}

    concurrency = min(max(1, concurrency), MAX_AGENTS)
    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.monotonic()

    async def run_agent(idx: int, task: dict) -> dict:
        async with semaphore:
            agent_model = task.get("model", model)
            agent_prompt = task.get("prompt", "")
            agent_system = task.get("system_prompt", "")
            label = task.get("label", f"Agent {idx + 1}")

            if not agent_prompt:
                return {"label": label, "isError": True, "message": "Empty prompt"}

            try:
                provider, resolved = registry.resolve(agent_model)
            except Exception as exc:
                return {"label": label, "isError": True, "message": str(exc)}

            # Check cache
            if cache:
                cached = cache.get(
                    provider.provider_name(), resolved,
                    agent_prompt, agent_system, 0.7,
                )
                if cached:
                    cost_tracker.record(
                        provider.provider_name(), resolved,
                        cached.input_tokens, cached.output_tokens, cached=True,
                    )
                    return {
                        "label": label,
                        "model": resolved,
                        "provider": provider.provider_name(),
                        "response": cached.text,
                        "tokens": {"input": cached.input_tokens, "output": cached.output_tokens},
                        "cost_usd": 0.0,
                        "cached": True,
                    }

            try:
                t0 = time.monotonic()
                response = await provider.generate(
                    agent_prompt, resolved,
                    system_prompt=agent_system or None,
                    temperature=0.7,
                )
                elapsed = (time.monotonic() - t0) * 1000
            except Exception as exc:
                return {"label": label, "isError": True, "message": str(exc)}

            if cache:
                cache.put(
                    provider.provider_name(), resolved,
                    agent_prompt, agent_system, 0.7, response,
                )
            cost_entry = cost_tracker.record(
                provider.provider_name(), resolved,
                response.input_tokens, response.output_tokens,
            )

            return {
                "label": label,
                "model": resolved,
                "provider": provider.provider_name(),
                "response": response.text,
                "tokens": {"input": response.input_tokens, "output": response.output_tokens},
                "cost_usd": round(cost_entry.cost_usd, 6),
                "latency_ms": round(elapsed, 1),
                "cached": False,
            }

    # Launch all agents with concurrency control
    total = len(task_list)
    if ctx:
        try:
            await ctx.report_progress(0, total, f"Launching {total} agents (concurrency={concurrency})...")
        except Exception:
            pass

    agent_tasks = [run_agent(i, t) for i, t in enumerate(task_list)]
    results = await asyncio.gather(*agent_tasks, return_exceptions=True)

    if ctx:
        try:
            await ctx.report_progress(total, total, f"All {total} agents completed")
        except Exception:
            pass

    # Separate successes and failures
    responses = []
    errors = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            errors.append({"index": i, "error": str(r)})
        elif isinstance(r, dict) and r.get("isError"):
            errors.append({"index": i, "label": r.get("label"), "error": r.get("message")})
        else:
            responses.append(r)

    total_cost = sum(r.get("cost_usd", 0) for r in responses)
    total_tokens = sum(
        r.get("tokens", {}).get("input", 0) + r.get("tokens", {}).get("output", 0)
        for r in responses
    )
    wall_time = (time.monotonic() - start_time) * 1000

    return {
        "agent_count": total,
        "responses": responses,
        "errors": errors if errors else None,
        "summary": {
            "succeeded": len(responses),
            "failed": len(errors),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "wall_time_ms": round(wall_time, 1),
            "concurrency": concurrency,
        },
        "instruction": (
            "Analyze all agent responses above. Synthesize findings across all agents. "
            "Highlight agreements, disagreements, and unique insights from each agent."
        ),
    }
