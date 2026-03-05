"""Debate tool — two AI models argue opposing sides, a judge synthesizes."""

import logging

from mcp.server.fastmcp import Context

from aarnxen.core.retry import call_with_retry
from aarnxen.core.validation import validate_prompt

logger = logging.getLogger(__name__)


async def debate_handler(
    topic: str,
    rounds: int = 3,
    model_a: str = "auto",
    model_b: str = "auto",
    judge: str = "auto",
    ctx: Context = None,
) -> dict:
    """Two AI models debate opposing sides of a topic, then a judge synthesizes.

    Args:
        topic: The topic or question to debate.
        rounds: Number of back-and-forth rounds (1-5).
        model_a: Model arguing FOR the topic.
        model_b: Model arguing AGAINST the topic.
        judge: Model that synthesizes and gives a verdict.
    """
    deps = ctx.request_context.lifespan_context
    registry = deps.registry
    cost_tracker = deps.cost_tracker

    topic = validate_prompt(topic)
    rounds = max(1, min(5, rounds))

    # Resolve 3 models (try to pick different ones)
    models = registry.get_top_n_models(3)
    def resolve(m, idx):
        if m == "auto" and idx < len(models):
            return registry.resolve(models[idx][1])
        return registry.resolve(m)

    prov_a, model_a_resolved = resolve(model_a, 0)
    prov_b, model_b_resolved = resolve(model_b, 1)
    prov_j, model_j_resolved = resolve(judge, 2)

    sys_a = f"You are debating FOR this position. Argue persuasively with evidence and logic. Be concise but thorough."
    sys_b = f"You are debating AGAINST this position. Counter-argue persuasively with evidence and logic. Be concise but thorough."

    transcript = []
    total_cost = 0.0
    total_tokens = 0
    prev_arg = ""

    for r in range(rounds):
        # Side A argues
        prompt_a = f"Topic: {topic}\n\n"
        if prev_arg:
            prompt_a += f"Your opponent's last argument:\n{prev_arg}\n\nRespond and advance your position:"
        else:
            prompt_a += "Make your opening argument FOR this position:"

        fallbacks_a = registry.get_fallbacks(model_a_resolved)
        resp_a = await call_with_retry(
            prov_a, model_a_resolved, prompt_a,
            system_prompt=sys_a, temperature=0.7,
            fallback_providers=fallbacks_a,
            circuit_breaker=deps.circuit_breaker, max_retries=1,
        )
        cost_a = cost_tracker.record(prov_a.provider_name(), model_a_resolved, resp_a.input_tokens, resp_a.output_tokens)
        total_cost += cost_a.cost_usd
        total_tokens += resp_a.input_tokens + resp_a.output_tokens

        transcript.append({"round": r + 1, "side": "FOR", "model": model_a_resolved, "argument": resp_a.text})

        # Side B counters
        prompt_b = f"Topic: {topic}\n\nYour opponent's argument:\n{resp_a.text}\n\nCounter-argue AGAINST this position:"

        fallbacks_b = registry.get_fallbacks(model_b_resolved)
        resp_b = await call_with_retry(
            prov_b, model_b_resolved, prompt_b,
            system_prompt=sys_b, temperature=0.7,
            fallback_providers=fallbacks_b,
            circuit_breaker=deps.circuit_breaker, max_retries=1,
        )
        cost_b = cost_tracker.record(prov_b.provider_name(), model_b_resolved, resp_b.input_tokens, resp_b.output_tokens)
        total_cost += cost_b.cost_usd
        total_tokens += resp_b.input_tokens + resp_b.output_tokens

        transcript.append({"round": r + 1, "side": "AGAINST", "model": model_b_resolved, "argument": resp_b.text})
        prev_arg = resp_b.text

    # Judge synthesizes
    debate_text = "\n\n".join(
        f"[Round {t['round']} - {t['side']} ({t['model']})]:\n{t['argument']}"
        for t in transcript
    )
    judge_prompt = (
        f"You are judging a debate on: {topic}\n\n"
        f"Here is the full debate transcript:\n\n{debate_text}\n\n"
        f"Provide:\n1. A verdict (which side had stronger arguments and why)\n"
        f"2. Key strengths from each side\n3. A balanced synthesis/conclusion"
    )

    fallbacks_j = registry.get_fallbacks(model_j_resolved)
    resp_j = await call_with_retry(
        prov_j, model_j_resolved, judge_prompt,
        system_prompt="You are an impartial judge. Analyze the debate fairly.",
        temperature=0.5,
        fallback_providers=fallbacks_j,
        circuit_breaker=deps.circuit_breaker, max_retries=1,
    )
    cost_j = cost_tracker.record(prov_j.provider_name(), model_j_resolved, resp_j.input_tokens, resp_j.output_tokens)
    total_cost += cost_j.cost_usd
    total_tokens += resp_j.input_tokens + resp_j.output_tokens

    # Store verdict in KB if available
    if deps.knowledge:
        try:
            deps.knowledge.add_observation(
                topic[:100], f"Debate verdict: {resp_j.text[:300]}",
                obs_type="decision",
            )
        except Exception as exc:
            logger.debug("Failed to store debate verdict in KB: %s", exc)

    return {
        "topic": topic,
        "rounds": rounds,
        "models": {"for": model_a_resolved, "against": model_b_resolved, "judge": model_j_resolved},
        "transcript": transcript,
        "verdict": resp_j.text,
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
    }
