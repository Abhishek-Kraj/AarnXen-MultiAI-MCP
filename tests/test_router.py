"""Tests for smart router — classification, routing, and cascading."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aarnxen.core.router import SmartRouter, TIERS, TASK_TO_TIER
from aarnxen.providers.base import ModelResponse


def _make_registry(available_models=None):
    """Build a mock registry with controllable model list."""
    if available_models is None:
        available_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gpt-4o", "llama-3.3-70b-versatile"]

    registry = MagicMock()
    registry.list_all_models.return_value = [{"model": m} for m in available_models]
    registry.get_fallbacks.return_value = []
    registry._model_to_provider = {m: "mock" for m in available_models}

    def resolve_side_effect(model=None):
        if not model or model == "auto":
            m = available_models[0]
        else:
            m = model
        provider = MagicMock()
        provider.provider_name.return_value = "mock"
        return provider, m

    registry.resolve.side_effect = resolve_side_effect
    return registry


def _make_response(text="Hello world", tokens=100):
    return ModelResponse(
        text=text, model="test", provider="mock",
        input_tokens=tokens, output_tokens=tokens, latency_ms=50.0,
    )


# --- Task Classification ---

class TestClassify:

    def test_empty_prompt(self):
        router = SmartRouter(_make_registry())
        assert router.classify("") == "simple"
        assert router.classify("   ") == "simple"

    def test_greeting(self):
        router = SmartRouter(_make_registry())
        assert router.classify("hello") == "simple"
        assert router.classify("Hi") == "simple"
        assert router.classify("thanks") == "simple"

    def test_short_factual(self):
        router = SmartRouter(_make_registry())
        assert router.classify("What is Python?") == "simple"
        assert router.classify("Who is Einstein?") == "simple"

    def test_code_keywords(self):
        router = SmartRouter(_make_registry())
        assert router.classify("Write a Python function to sort a list") == "code"
        assert router.classify("Debug this error in my API endpoint") == "code"
        assert router.classify("Refactor the class to use dependency injection") == "code"
        assert router.classify("Fix the SQL query that returns wrong results from the database") == "code"

    def test_reasoning_keywords(self):
        router = SmartRouter(_make_registry())
        assert router.classify("Analyze the trade-offs between microservices and monoliths") == "reasoning"
        assert router.classify("Compare React and Vue, explain how they differ in state management") == "reasoning"
        assert router.classify("Why does this algorithm have O(n log n) complexity? Prove it step by step") == "reasoning"

    def test_creative_keywords(self):
        router = SmartRouter(_make_registry())
        assert router.classify("Write a short story about a robot discovering emotions") == "creative"
        assert router.classify("Brainstorm ideas for a sci-fi screenplay about time travel") == "creative"
        assert router.classify("Compose a poem about the ocean at sunset") == "creative"

    def test_general_fallback(self):
        router = SmartRouter(_make_registry())
        result = router.classify(
            "Tell me about the history of the Roman Empire and its impact on modern governance "
            "across Europe and the Mediterranean region over several centuries of expansion"
        )
        assert result == "general"

    def test_very_long_prompt_no_keywords(self):
        router = SmartRouter(_make_registry())
        prompt = "something " * 30
        assert router.classify(prompt) == "general"


# --- Routing ---

class TestRoute:

    def test_simple_routes_to_budget(self):
        router = SmartRouter(_make_registry())
        model, task_type, tier = router.route("hello")
        assert task_type == "simple"
        assert tier == "budget"

    def test_code_routes_to_premium(self):
        router = SmartRouter(_make_registry())
        model, task_type, tier = router.route("Write a Python function to parse JSON files")
        assert task_type == "code"
        assert tier == "premium"

    def test_creative_routes_to_balanced(self):
        router = SmartRouter(_make_registry())
        model, task_type, tier = router.route("Write a poem about the stars in the night sky")
        assert task_type == "creative"
        assert tier == "balanced"

    def test_route_returns_available_model(self):
        registry = _make_registry(["gemini-2.5-flash"])
        router = SmartRouter(registry)
        model, _, _ = router.route("hello")
        assert model == "gemini-2.5-flash"

    def test_route_falls_back_when_tier_empty(self):
        registry = _make_registry(["some-unknown-model"])
        router = SmartRouter(registry)
        model, _, _ = router.route("hello")
        # Should fall back to registry default
        assert model == "some-unknown-model"


# --- Cascade ---

class TestCascade:

    @pytest.mark.asyncio
    async def test_cascade_no_escalation(self):
        registry = _make_registry()
        router = SmartRouter(registry)

        good_response = _make_response("Here is a detailed explanation of the concept with examples and context.")
        provider_mock = MagicMock()
        provider_mock.provider_name.return_value = "mock"
        provider_mock.generate = AsyncMock(return_value=good_response)
        registry.resolve.side_effect = lambda m=None: (provider_mock, m or "gemini-2.5-flash")

        result = await router.cascade("hello")
        assert result["escalated"] is False
        assert result["text"] == good_response.text

    @pytest.mark.asyncio
    async def test_cascade_escalates_on_low_quality(self):
        registry = _make_registry()
        router = SmartRouter(registry)

        bad_response = _make_response("I don't know about that.")
        good_response = _make_response("Here is a comprehensive answer with all the details you need about the topic.")

        call_count = 0

        async def mock_generate(prompt, model, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_response
            return good_response

        provider_mock = MagicMock()
        provider_mock.provider_name.return_value = "mock"
        provider_mock.generate = mock_generate
        registry.resolve.side_effect = lambda m=None: (provider_mock, m or "gemini-2.5-flash")

        result = await router.cascade(
            "Tell me about the history of the Roman Empire and how it influenced governance "
            "across several continents over many centuries of expansion and cultural development"
        )
        assert result["escalated"] is True
        assert result["text"] == good_response.text
        assert result["initial_response"] is not None
        assert "escalation_reason" in result

    @pytest.mark.asyncio
    async def test_cascade_no_escalation_for_premium_tier(self):
        registry = _make_registry()
        router = SmartRouter(registry)

        short_response = _make_response("Short.")
        provider_mock = MagicMock()
        provider_mock.provider_name.return_value = "mock"
        provider_mock.generate = AsyncMock(return_value=short_response)
        registry.resolve.side_effect = lambda m=None: (provider_mock, m or "gemini-2.5-pro")

        # Code task routes to premium tier, so no escalation even if response is short
        result = await router.cascade("Write a Python function to sort a list using merge sort algorithm")
        assert result["escalated"] is False


# --- Quality Detection ---

class TestQualityDetection:

    def test_short_code_response_is_low_quality(self):
        router = SmartRouter(_make_registry())
        assert router._is_low_quality("x = 1", "code") is True

    def test_hedging_is_low_quality(self):
        router = SmartRouter(_make_registry())
        assert router._is_low_quality("I don't know the answer to that question.", "general") is True
        assert router._is_low_quality("As an AI, I cannot help with that.", "reasoning") is True

    def test_good_response_is_not_low_quality(self):
        router = SmartRouter(_make_registry())
        long_answer = "Here is a detailed response " * 10
        assert router._is_low_quality(long_answer, "code") is False

    def test_short_simple_is_ok(self):
        router = SmartRouter(_make_registry())
        assert router._is_low_quality("Yes.", "simple") is False

    def test_very_short_non_simple_is_low(self):
        router = SmartRouter(_make_registry())
        assert router._is_low_quality("Ok", "general") is True
