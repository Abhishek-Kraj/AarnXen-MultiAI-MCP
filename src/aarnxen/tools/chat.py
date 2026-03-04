"""Single-model chat tool with optional smart routing and cascading."""

import logging

from mcp.server.fastmcp import Context

from aarnxen.core.extractor import EntityExtractor
from aarnxen.core.router import SmartRouter

logger = logging.getLogger("aarnxen")


async def chat_handler(
    prompt: str,
    model: str = "auto",
    temperature: float = 0.7,
    system_prompt: str = "",
    conversation_id: str = "",
    cascade: bool = False,
    ctx: Context = None,
) -> dict:
    """Chat with any AI model.

    Args:
        prompt: Your message or question.
        model: Model name, provider name, or "auto". Examples: "gemini-2.5-flash", "groq", "auto".
        temperature: Creativity (0.0=focused, 1.0=creative). Default 0.7.
        system_prompt: Optional system instructions.
        conversation_id: Continue a previous conversation by ID.
        cascade: Smart routing with auto-escalation. Only works when model="auto".
    """
    deps = ctx.request_context.lifespan_context
    registry = deps["registry"]
    cache = deps["cache"]
    cost_tracker = deps["cost_tracker"]
    memory = deps["memory"]

    # Smart cascade mode: classify, route, escalate if needed
    if cascade and (not model or model == "auto"):
        router = SmartRouter(registry)
        result = await router.cascade(
            prompt, system_prompt=system_prompt or None, temperature=temperature,
        )

        cost_entry = cost_tracker.record(
            result["provider"], result["model"],
            result["input_tokens"], result["output_tokens"],
        )

        if conversation_id and memory:
            memory.add_message(conversation_id, "user", prompt)
            memory.add_message(conversation_id, "assistant", result["text"], result["model"], result["provider"])

        response = {
            "response": result["text"],
            "model": result["model"],
            "provider": result["provider"],
            "cached": False,
            "tokens": {"input": result["input_tokens"], "output": result["output_tokens"]},
            "cost_usd": round(cost_entry.cost_usd, 6),
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

    # Check cache
    if cache:
        cached = cache.get(provider.provider_name(), resolved_model, prompt, system_prompt, temperature)
        if cached:
            cost_tracker.record(
                provider.provider_name(), resolved_model,
                cached.input_tokens, cached.output_tokens, cached=True,
            )
            return {
                "response": cached.text,
                "model": resolved_model,
                "provider": provider.provider_name(),
                "cached": True,
                "tokens": {"input": cached.input_tokens, "output": cached.output_tokens},
            }

    # Build context from conversation history
    full_prompt = prompt
    if conversation_id and memory:
        history = memory.get_history(conversation_id)
        if history:
            context_lines = [f"[{m['role']}]: {m['content']}" for m in history[-10:]]
            full_prompt = "Previous conversation:\n" + "\n".join(context_lines) + f"\n\nNew message: {prompt}"

    response = await provider.generate(
        full_prompt, resolved_model,
        system_prompt=system_prompt or None,
        temperature=temperature,
    )

    # Cache and track
    if cache:
        cache.put(provider.provider_name(), resolved_model, prompt, system_prompt, temperature, response)
    cost_entry = cost_tracker.record(
        provider.provider_name(), resolved_model,
        response.input_tokens, response.output_tokens,
    )

    # Save to conversation memory
    if conversation_id and memory:
        memory.add_message(conversation_id, "user", prompt)
        memory.add_message(conversation_id, "assistant", response.text, resolved_model, provider.provider_name())

    result = {
        "response": response.text,
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
        kb = deps.get("knowledge") if isinstance(deps, dict) else getattr(deps, "knowledge", None)
        if not kb:
            return
        extractor = EntityExtractor(knowledge_base=kb)
        stats = extractor.extract_and_store(prompt)
        total = stats["entities_added"] + stats["relations_added"]
        if total > 0:
            logger.debug("Auto-extracted: %s", stats)
    except Exception:
        logger.debug("Entity extraction skipped", exc_info=True)
