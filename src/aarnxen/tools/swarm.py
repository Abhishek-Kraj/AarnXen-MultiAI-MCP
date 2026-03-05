"""Multi-agent swarm — launch N parallel agents on sub-tasks."""

import asyncio
import json
import logging
import time

from mcp.server.fastmcp import Context

from aarnxen.core.errors import generation_failed
from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_json_input, validate_prompt, truncate_response

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 10
MAX_AGENTS = 100


async def swarm_handler(
    tasks: str,
    model: str = "auto",
    concurrency: int = DEFAULT_CONCURRENCY,
    max_budget_usd: float = 0.0,
    ctx: Context = None,
) -> dict:
    """Launch multiple AI agents in parallel on different sub-tasks."""
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker
    cache = deps.cache
    circuit_breaker = deps.circuit_breaker

    try:
        task_list = validate_json_input(tasks, max_items=MAX_AGENTS)
    except ValueError as exc:
        return {"isError": True, "message": str(exc)}

    concurrency = min(max(1, concurrency), MAX_AGENTS)
    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.monotonic()

    budget_enabled = max_budget_usd > 0
    budget_lock = asyncio.Lock()
    budget_spent = {"usd": 0.0, "exceeded": False, "cancelled": 0}

    async def run_agent(idx: int, task: dict) -> dict:
        async with semaphore:
            if budget_enabled and budget_spent["exceeded"]:
                label = task.get("label", f"Agent {idx + 1}")
                async with budget_lock:
                    budget_spent["cancelled"] += 1
                return {"label": label, "isError": True, "message": "Budget exceeded, agent cancelled"}
            agent_model = task.get("model", model)
            agent_prompt = task.get("prompt", "")
            agent_system = task.get("system_prompt", "")
            agent_temp = task.get("temperature", 0.7)
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
                    agent_prompt, agent_system, agent_temp,
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
                        "response": truncate_response(cached.text),
                        "tokens": {"input": cached.input_tokens, "output": cached.output_tokens},
                        "cost_usd": 0.0,
                        "cached": True,
                    }

            try:
                t0 = time.monotonic()
                fallbacks = registry.get_fallbacks(resolved)
                response = await call_with_retry(
                    provider, resolved, agent_prompt,
                    system_prompt=agent_system or None,
                    temperature=agent_temp,
                    fallback_providers=fallbacks,
                    circuit_breaker=circuit_breaker,
                    max_retries=2,
                )
                elapsed = (time.monotonic() - t0) * 1000
            except Exception as exc:
                logger.warning("Agent %s failed: %s", label, exc)
                return {"label": label, "isError": True, "message": type(exc).__name__}

            if cache:
                cache.put(
                    provider.provider_name(), resolved,
                    agent_prompt, agent_system, agent_temp, response,
                )
            cost_entry = cost_tracker.record(
                provider.provider_name(), resolved,
                response.input_tokens, response.output_tokens,
            )

            if budget_enabled:
                async with budget_lock:
                    budget_spent["usd"] += cost_entry.cost_usd
                    if budget_spent["usd"] >= max_budget_usd:
                        budget_spent["exceeded"] = True

            return {
                "label": label,
                "model": resolved,
                "provider": provider.provider_name(),
                "response": truncate_response(response.text),
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

    result = {
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

    if budget_enabled:
        result["budget"] = {
            "max_usd": max_budget_usd,
            "spent_usd": round(budget_spent["usd"], 6),
            "budget_exceeded": budget_spent["exceeded"],
            "agents_cancelled": budget_spent["cancelled"],
        }

    return result
