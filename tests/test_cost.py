"""Tests for cost tracker."""

from aarnxen.core.cost import CostTracker


def test_record_cost():
    tracker = CostTracker()
    entry = tracker.record("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
    # gpt-4o: $2.50/M input, $10.00/M output
    assert entry.cost_usd > 0
    assert entry.input_tokens == 1000
    assert entry.output_tokens == 500


def test_cached_zero_cost():
    tracker = CostTracker()
    entry = tracker.record("openai", "gpt-4o", input_tokens=1000, output_tokens=500, cached=True)
    assert entry.cost_usd == 0.0
    assert entry.cached is True


def test_summary():
    tracker = CostTracker()
    tracker.record("openai", "gpt-4o", 1000, 500)
    tracker.record("gemini", "gemini-2.5-flash", 2000, 1000)
    summary = tracker.summary()
    assert summary["total_requests"] == 2
    assert summary["total_input_tokens"] == 3000
    assert summary["total_output_tokens"] == 1500
    assert "openai/gpt-4o" in summary["by_model"]
    assert "gemini/gemini-2.5-flash" in summary["by_model"]


def test_unknown_model_zero_cost():
    tracker = CostTracker()
    entry = tracker.record("unknown", "mystery-model", 1000, 500)
    assert entry.cost_usd == 0.0
