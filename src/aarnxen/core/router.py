"""Smart model routing with task classification and cascading."""

from __future__ import annotations

import logging
import re
from typing import Optional

from aarnxen.providers.base import ModelResponse
from aarnxen.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

TIERS = {
    "budget": ["gemini-2.5-flash", "llama3.2", "gemini-flash", "llama-3.3-70b-versatile"],
    "balanced": ["gemini-2.5-pro", "gpt-4o-mini", "mixtral", "llama-3.3-70b-versatile"],
    "premium": ["gemini-2.5-pro", "gpt-4o", "claude-3-opus", "llama-3.3-70b-versatile"],
}

TASK_KEYWORDS = {
    "code": [
        "code", "function", "bug", "error", "debug", "implement", "refactor",
        "class", "api", "sql", "script", "compile", "syntax", "variable",
        "import", "module", "library", "framework", "deploy", "test",
        "regex", "algorithm", "data structure", "git", "commit", "merge",
        "html", "css", "javascript", "python", "typescript", "rust", "java",
        "dockerfile", "yaml", "json", "endpoint", "database", "query",
    ],
    "reasoning": [
        "analyze", "compare", "why", "explain how", "calculate", "prove",
        "reason", "evaluate", "trade-off", "tradeoff", "pros and cons",
        "cause", "effect", "implication", "consequence", "derive",
        "logic", "mathematical", "theorem", "hypothesis", "contradiction",
        "step by step", "break down", "differentiate between",
    ],
    "creative": [
        "write a", "write me", "story", "poem", "creative", "brainstorm", "imagine",
        "compose", "draft a", "fiction", "narrative", "essay", "blog post",
        "slogan", "tagline", "metaphor", "lyric", "screenplay", "dialogue",
    ],
}

# Build compiled regex patterns for word-boundary matching.
# Multi-word phrases use plain `in` (they're specific enough).
# Single words use \b word boundaries to avoid substring false positives.
_TASK_PATTERNS: dict[str, list[tuple[str, re.Pattern | None]]] = {}
for _task, _keywords in TASK_KEYWORDS.items():
    _patterns = []
    for _kw in _keywords:
        if " " in _kw:
            _patterns.append((_kw, None))
        else:
            _patterns.append((_kw, re.compile(r"\b" + re.escape(_kw) + r"\b", re.IGNORECASE)))
    _TASK_PATTERNS[_task] = _patterns

TASK_TO_TIER = {
    "simple": "budget",
    "code": "premium",
    "reasoning": "premium",
    "creative": "balanced",
    "general": "budget",
}

LOW_QUALITY_PHRASES = [
    "i don't know",
    "i'm not sure",
    "i cannot",
    "i can't",
    "as an ai",
    "i don't have enough information",
    "it depends",
    "i'm unable to",
]


class SmartRouter:

    def __init__(self, registry: ProviderRegistry):
        self._registry = registry
        self._available_models = {m["model"] for m in registry.list_all_models()}

    def classify(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return "simple"

        words = len(prompt.split())
        if words < 20:
            lower = prompt.lower().strip()
            # Greetings and trivial
            if lower in ("hi", "hello", "hey", "thanks", "thank you", "ok", "yes", "no"):
                return "simple"
            # Short "what is X" style questions
            if re.match(r"^(what|who|when|where) (is|are|was|were) \w+\??$", lower):
                return "simple"

        lower = prompt.lower()

        scores = {task: 0 for task in TASK_KEYWORDS}
        for task, patterns in _TASK_PATTERNS.items():
            for kw, pattern in patterns:
                if pattern is not None:
                    if pattern.search(lower):
                        scores[task] += 1
                elif kw in lower:
                    scores[task] += 1

        best_task = max(scores, key=scores.get)
        if scores[best_task] > 0:
            return best_task

        if words < 20:
            return "simple"

        return "general"

    def _find_available(self, tier_name: str) -> Optional[str]:
        tier_models = TIERS.get(tier_name, TIERS["budget"])
        for model in tier_models:
            if model in self._available_models:
                return model
            # Fuzzy: check if any available model contains the tier model name
            for avail in self._available_models:
                if model.lower() in avail.lower():
                    return avail
        return None

    def route(self, prompt: str) -> tuple[str, str, str]:
        """Classify and route. Returns (model_name, task_type, tier)."""
        task_type = self.classify(prompt)
        tier = TASK_TO_TIER.get(task_type, "budget")
        model = self._find_available(tier)

        if model is None:
            # Fall back to whatever the registry has
            _, model = self._registry.resolve("auto")
            logger.info("No tier model available, falling back to registry default: %s", model)

        logger.info("Routed task_type=%s tier=%s model=%s", task_type, tier, model)
        return model, task_type, tier

    def _is_low_quality(self, response_text: str, task_type: str) -> bool:
        lower = response_text.lower().strip()

        # Too short for a complex task
        if task_type in ("code", "reasoning") and len(response_text.split()) < 30:
            return True

        # Contains hedging / refusal phrases
        for phrase in LOW_QUALITY_PHRASES:
            if phrase in lower:
                return True

        # Very short response for anything non-simple
        if task_type != "simple" and len(response_text.strip()) < 20:
            return True

        return False

    async def cascade(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> dict:
        """Route, call cheap model, escalate if low quality."""
        model, task_type, tier = self.route(prompt)

        provider, resolved_model = self._registry.resolve(model)
        response = await provider.generate(
            prompt, resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        escalated = False
        escalation_reason = None
        initial_response = None

        if tier != "premium" and self._is_low_quality(response.text, task_type):
            premium_model = self._find_available("premium")
            if premium_model and premium_model != resolved_model:
                logger.info(
                    "Escalating from %s to %s (low quality detected)", resolved_model, premium_model,
                )
                initial_response = {
                    "text": response.text,
                    "model": resolved_model,
                    "provider": provider.provider_name(),
                }
                escalation_reason = "Initial response was low quality; escalated to premium tier."

                provider, resolved_model = self._registry.resolve(premium_model)
                response = await provider.generate(
                    prompt, resolved_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                escalated = True

        result = {
            "text": response.text,
            "model": resolved_model,
            "provider": provider.provider_name(),
            "task_type": task_type,
            "tier": "premium" if escalated else tier,
            "escalated": escalated,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
        }

        if escalated:
            result["escalation_reason"] = escalation_reason
            result["initial_response"] = initial_response

        return result
