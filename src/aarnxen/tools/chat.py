"""Single-model chat tool with optional smart routing and cascading."""

import logging

from mcp.server.fastmcp import Context

from aarnxen.core.errors import generation_failed, rate_limit_error
from aarnxen.core.retry import RetryError, call_with_retry
from aarnxen.core.validation import validate_prompt, validate_temperature, sanitize_system_prompt, truncate_response

logger = logging.getLogger("aarnxen")


async def _log(ctx, level, msg):
    if not ctx:
        return
    try:
        fn = getattr(ctx, level, None)
        if fn:
            await fn(msg)
    except Exception:
        pass


async def chat_handler(
    prompt: str,
    model: str = "auto",
    temperature: float = 0.7,
    system_prompt: str = "",
    conversation_id: str = "",
    cascade: bool = False,
    ctx: Context = None,
) -> dict:
    """Chat with any AI model."""
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cache = deps.cache
    cost_tracker = deps.cost_tracker
    memory = deps.memory
    circuit_breaker = deps.circuit_breaker

    prompt = validate_prompt(prompt)
    temperature = validate_temperature(temperature)
    if system_prompt:
        system_prompt = sanitize_system_prompt(system_prompt)

    # Smart cascade mode: classify, route, escalate if needed
    if cascade and (not model or model == "auto"):
        router = deps.router
        try:
            result = await router.cascade(
                prompt, system_prompt=system_prompt or None, temperature=temperature,
            )
        except (RetryError, Exception) as exc:
            logger.warning("Cascade failed: %s", exc)
            error_str = type(exc).__name__
            alts = [m["model"] for m in registry.list_all_models()[:5]]
            if "429" in error_str or "rate" in error_str.lower():
                return rate_limit_error("auto (cascade)", alternatives=alts)
            return generation_failed("auto", "cascade", error_str, alternatives=alts)

        cost_entry = cost_tracker.record(
            result["provider"], result["model"],
            result["input_tokens"], result["output_tokens"],
        )
        # Track the initial (cheap) call's cost when escalation occurred
        initial_cost = 0.0
        if result.get("escalated") and result.get("initial_response"):
            ir = result["initial_response"]
            initial_entry = cost_tracker.record(
                ir.get("provider", "unknown"), ir.get("model", "unknown"),
                ir.get("input_tokens", 0), ir.get("output_tokens", 0),
            )
            initial_cost = initial_entry.cost_usd

        if conversation_id and memory:
            memory.add_message(conversation_id, "user", prompt)
            memory.add_message(conversation_id, "assistant", result["text"], result["model"], result["provider"])

        total_cost = cost_entry.cost_usd + initial_cost
        response = {
            "response": truncate_response(result["text"]),
            "model": result["model"],
            "provider": result["provider"],
            "cached": False,
            "tokens": {"input": result["input_tokens"], "output": result["output_tokens"]},
            "cost_usd": round(total_cost, 6),
            "latency_ms": round(result["latency_ms"], 1),
            "smart_routing": {
                "task_type": result["task_type"],
                "tier": result["tier"],
                "escalated": result["escalated"],
            },
        }
        if result.get("escalation_reason"):
            response["smart_routing"]["escalation_reason"] = result["escalation_reason"]
        if result.get("initial_response"):
            response["smart_routing"]["initial_response"] = result["initial_response"]

        _auto_extract(deps, prompt)
        return response

    provider, resolved_model = registry.resolve(model)

    # Build context from conversation history
    full_prompt = prompt
    if conversation_id and memory:
        history = memory.get_history(conversation_id)
        if history:
            context_lines = [f"[{m['role']}]: {m['content']}" for m in history[-10:]]
            full_prompt = "Previous conversation:\n" + "\n".join(context_lines) + f"\n\nNew message: {prompt}"

    # Check cache
    if cache:
        cached = cache.get(provider.provider_name(), resolved_model, full_prompt, system_prompt, temperature)
        if cached:
            await _log(ctx, "debug", f"Cache hit for {resolved_model}")
            cost_tracker.record(
                provider.provider_name(), resolved_model,
                cached.input_tokens, cached.output_tokens, cached=True,
            )
            return {
                "response": truncate_response(cached.text),
                "model": resolved_model,
                "provider": provider.provider_name(),
                "cached": True,
                "tokens": {"input": cached.input_tokens, "output": cached.output_tokens},
            }

    await _log(ctx, "info", f"Routing to {provider.provider_name()}/{resolved_model}")
    try:
        fallbacks = registry.get_fallbacks(resolved_model)
        response = await call_with_retry(
            provider, resolved_model, full_prompt,
            system_prompt=system_prompt or None,
            temperature=temperature,
            fallback_providers=fallbacks,
            circuit_breaker=circuit_breaker,
            max_retries=2,
        )
    except (RetryError, Exception) as exc:
        logger.warning("Generation failed: %s", exc)
        error_str = type(exc).__name__
        alts = [m["model"] for m in registry.list_all_models()[:5]]
        if "429" in error_str or "rate" in error_str.lower():
            return rate_limit_error(provider.provider_name(), alternatives=alts)
        return generation_failed(
            provider.provider_name(), resolved_model, error_str, alternatives=alts,
        )
    await _log(ctx, "info", f"Response: {len(response.text)} chars, {response.latency_ms:.0f}ms")

    if cache:
        cache.put(provider.provider_name(), resolved_model, full_prompt, system_prompt, temperature, response)
    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    if conversation_id and memory:
        memory.add_message(conversation_id, "user", prompt)
        memory.add_message(conversation_id, "assistant", response.text, resolved_model, provider.provider_name())

    result = {
        "response": truncate_response(response.text),
        "model": resolved_model,
        "provider": provider.provider_name(),
        "cached": False,
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
        "cost_usd": round(cost_entry.cost_usd, 6),
        "latency_ms": round(response.latency_ms, 1),
    }

    _auto_extract(deps, prompt)
    return result


def _auto_extract(deps, prompt: str):
    """Extract entities/relations from the prompt and store in the knowledge base."""
    try:
        extractor = deps.extractor
        if not extractor:
            return
        stats = extractor.extract_and_store(prompt)
        total = stats["entities_added"] + stats["relations_added"]
        if total > 0:
            logger.debug("Auto-extracted: %s", stats)
    except Exception:
        logger.debug("Entity extraction skipped", exc_info=True)
