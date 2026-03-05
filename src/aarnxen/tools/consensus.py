"""Parallel multi-model consensus tool."""

import asyncio

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt, sanitize_system_prompt, truncate_response


async def consensus_handler(
    prompt: str,
    models: str = "auto",
    temperature: float = 0.7,
    system_prompt: str = "",
    ctx: Context = None,
) -> dict:
    """Query multiple AI models in parallel and get all responses for synthesis.

    Args:
        prompt: Question or task for all models.
        models: Comma-separated models, e.g. "gemini-2.5-flash,groq,openai". Use "auto" for top 3.
        temperature: Creativity level (0.0-1.0).
        system_prompt: Optional system instructions applied to all models.
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cache = deps.cache
    cost_tracker = deps.cost_tracker

    prompt = validate_prompt(prompt)
    if system_prompt:
        system_prompt = sanitize_system_prompt(system_prompt)

    # Resolve models
    if models == "auto":
        model_pairs = registry.get_top_n_models(3)
    else:
        model_pairs = []
        for m in models.split(","):
            m = m.strip()
            provider, resolved = registry.resolve(m)
            model_pairs.append((provider.provider_name(), resolved))

    labels = ["Model A", "Model B", "Model C", "Model D", "Model E"]

    async def query_one(idx: int, provider_name: str, model_id: str) -> dict:
        provider, resolved = registry.resolve(model_id if model_id != "default" else provider_name)

        # Check cache
        if cache:
            cached = cache.get(provider.provider_name(), resolved, prompt, system_prompt, temperature)
            if cached:
                cost_tracker.record(provider.provider_name(), resolved, cached.input_tokens, cached.output_tokens, cached=True)
                return {
                    "label": labels[idx] if idx < len(labels) else f"Model {idx+1}",
                    "model": resolved,
                    "provider": provider.provider_name(),
                    "response": truncate_response(cached.text),
                    "tokens": {"input": cached.input_tokens, "output": cached.output_tokens},
                    "cost_usd": 0.0,
                    "cached": True,
                }

        fallbacks = registry.get_fallbacks(resolved)
        response = await call_with_retry(
            provider, resolved, prompt,
            system_prompt=system_prompt or None,
            temperature=temperature,
            fallback_providers=fallbacks,
            circuit_breaker=deps.circuit_breaker,
            max_retries=2,
        )

        if cache:
            cache.put(provider.provider_name(), resolved, prompt, system_prompt, temperature, response)
        cost_entry = cost_tracker.record(provider.provider_name(), resolved, response.input_tokens, response.output_tokens)

        return {
            "label": labels[idx] if idx < len(labels) else f"Model {idx+1}",
            "model": resolved,
            "provider": provider.provider_name(),
            "response": truncate_response(response.text),
            "tokens": {"input": response.input_tokens, "output": response.output_tokens},
            "cost_usd": round(cost_entry.cost_usd, 6),
            "cached": False,
            "latency_ms": round(response.latency_ms, 1),
        }

    # Fire all queries in parallel
    total = len(model_pairs)
    if ctx:
        await ctx.report_progress(0, total, "Starting consensus...")
    tasks = [query_one(i, pname, mid) for i, (pname, mid) in enumerate(model_pairs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if ctx:
        await ctx.report_progress(total, total, "All models responded")

    responses = []
    errors = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            errors.append({"index": i, "error": str(r)})
        else:
            responses.append(r)

    total_cost = sum(r["cost_usd"] for r in responses)
    total_tokens = sum(r["tokens"]["input"] + r["tokens"]["output"] for r in responses)

    analysis = _build_analysis(responses)

    agreed = len(analysis["agreement_signals"]) > len(analysis["disagreements"])
    agreement_word = "agreed" if agreed else "disagreed"
    model_names = ", ".join(r["label"] for r in responses)

    result = {
        "prompt": prompt,
        "model_count": len(model_pairs),
        "responses": responses,
        "errors": errors if errors else None,
        "analysis": analysis,
        "summary": {
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "succeeded": len(responses),
            "failed": len(errors),
        },
        "instruction": (
            f"Analyze the responses above. The models {agreement_word} on key points. "
            f"Focus on synthesizing the strongest arguments from each model, noting where "
            f"{model_names} differ and why. Also consider the knowledge base context below if available."
        ),
    }

    if deps.knowledge:
        try:
            kb_results = deps.knowledge.search_documents(prompt, limit=3)
            if kb_results:
                result["knowledge_context"] = [
                    {"title": r["title"], "snippet": r["snippet"][:200]}
                    for r in kb_results
                ]
        except Exception:
            pass

    return result


def _build_analysis(responses: list[dict]) -> dict:
    if len(responses) < 2:
        return {
            "agreement_signals": [],
            "disagreements": [],
            "unique_insights": [],
            "response_lengths": [],
        }

    def _extract_key_phrases(text: str) -> set[str]:
        sentences = []
        for s in text.replace("\n", ". ").split(". "):
            s = s.strip().lower()
            if len(s) > 10:
                sentences.append(s)
        return set(sentences)

    all_phrases = []
    for r in responses:
        all_phrases.append(_extract_key_phrases(r.get("response", "")))

    agreement_signals = []
    disagreements = []
    unique_insights = []

    if all_phrases:
        common = all_phrases[0]
        for phrases in all_phrases[1:]:
            common = common & phrases
        agreement_signals = sorted(common)[:10]

        for i, phrases in enumerate(all_phrases):
            others = set()
            for j, other_phrases in enumerate(all_phrases):
                if i != j:
                    others |= other_phrases
            unique = phrases - others
            if unique:
                label = responses[i].get("label", f"Model {i+1}")
                for phrase in sorted(unique)[:3]:
                    unique_insights.append({"model": label, "insight": phrase})

        for i in range(len(all_phrases)):
            for j in range(i + 1, len(all_phrases)):
                only_i = all_phrases[i] - all_phrases[j]
                only_j = all_phrases[j] - all_phrases[i]
                if only_i and only_j:
                    label_i = responses[i].get("label", f"Model {i+1}")
                    label_j = responses[j].get("label", f"Model {j+1}")
                    disagreements.append({
                        "between": [label_i, label_j],
                        "model_a_points": sorted(only_i)[:2],
                        "model_b_points": sorted(only_j)[:2],
                    })

    response_lengths = []
    for r in responses:
        text = r.get("response", "")
        response_lengths.append({
            "label": r.get("label", "Unknown"),
            "char_count": len(text),
            "word_count": len(text.split()),
        })
    response_lengths.sort(key=lambda x: x["char_count"], reverse=True)

    return {
        "agreement_signals": agreement_signals,
        "disagreements": disagreements[:5],
        "unique_insights": unique_insights[:10],
        "response_lengths": response_lengths,
    }
