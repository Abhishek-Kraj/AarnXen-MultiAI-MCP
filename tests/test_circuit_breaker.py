"""Tests for circuit breaker pattern."""

import time
from unittest.mock import patch

import pytest

from aarnxen.core.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


def test_initial_state_is_closed():
    cb = CircuitBreaker()
    status = cb.get_status("gemini")
    assert status["state"] == "closed"
    assert status["failures_in_window"] == 0


def test_success_keeps_closed():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_success("gemini")
    cb.record_success("gemini")
    status = cb.get_status("gemini")
    assert status["state"] == "closed"
    assert status["success_count"] == 2


def test_failures_trip_to_open():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=60)
    for _ in range(3):
        cb.record_failure("gemini")
    status = cb.get_status("gemini")
    assert status["state"] == "open"
    assert status["total_failures"] == 3


def test_failures_below_threshold_stay_closed():
    cb = CircuitBreaker(failure_threshold=5)
    for _ in range(4):
        cb.record_failure("gemini")
    assert cb.get_status("gemini")["state"] == "closed"


def test_open_rejects_calls():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=30)
    cb.record_failure("gemini")
    cb.record_failure("gemini")
    assert cb.get_status("gemini")["state"] == "open"
    assert cb.can_execute("gemini") is False


def test_cooldown_transitions_to_half_open():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
    cb.record_failure("gemini")
    cb.record_failure("gemini")
    assert cb.can_execute("gemini") is False
    time.sleep(0.15)
    assert cb.can_execute("gemini") is True
    assert cb.get_status("gemini")["state"] == "half_open"


def test_half_open_success_closes_circuit():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
    cb.record_failure("gemini")
    cb.record_failure("gemini")
    time.sleep(0.15)
    cb.can_execute("gemini")  # triggers HALF_OPEN transition
    cb.record_success("gemini")
    status = cb.get_status("gemini")
    assert status["state"] == "closed"
    assert status["failures_in_window"] == 0


def test_half_open_failure_reopens_circuit():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
    cb.record_failure("gemini")
    cb.record_failure("gemini")
    time.sleep(0.15)
    cb.can_execute("gemini")  # triggers HALF_OPEN transition
    cb.record_failure("gemini")
    status = cb.get_status("gemini")
    assert status["state"] == "open"
    assert status["total_failures"] == 3


def test_multiple_providers_independent():
    cb = CircuitBreaker(failure_threshold=2)
    cb.record_failure("gemini")
    cb.record_failure("gemini")
    cb.record_failure("ollama")

    assert cb.get_status("gemini")["state"] == "open"
    assert cb.get_status("ollama")["state"] == "closed"
    assert cb.can_execute("ollama") is True
    assert cb.can_execute("gemini") is False


def test_sliding_window_clears_old_failures():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=0.1)
    cb.record_failure("gemini")
    cb.record_failure("gemini")
    time.sleep(0.15)
    # Old failures should have expired
    status = cb.get_status("gemini")
    assert status["failures_in_window"] == 0
    assert status["state"] == "closed"
    # New failures shouldn't trip because old ones expired
    cb.record_failure("gemini")
    assert cb.get_status("gemini")["state"] == "closed"


def test_get_all_status_dashboard():
    cb = CircuitBreaker(failure_threshold=2)
    cb.record_success("gemini")
    cb.record_failure("ollama")
    cb.record_failure("ollama")
    cb.record_success("openai")

    dashboard = cb.get_all_status()
    assert set(dashboard.keys()) == {"gemini", "ollama", "openai"}
    assert dashboard["gemini"]["state"] == "closed"
    assert dashboard["ollama"]["state"] == "open"
    assert dashboard["openai"]["state"] == "closed"
    assert "retry_after" in dashboard["ollama"]
    assert "retry_after" not in dashboard["gemini"]


def test_open_status_includes_retry_after():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=30)
    cb.record_failure("gemini")
    cb.record_failure("gemini")
    status = cb.get_status("gemini")
    assert status["state"] == "open"
    assert 0 < status["retry_after"] <= 30.0


def test_circuit_open_error_attributes():
    err = CircuitOpenError("gemini", 15.3)
    assert err.provider == "gemini"
    assert err.retry_after == 15.3
    assert "gemini" in str(err)
    assert "15.3" in str(err)


def test_unknown_provider_defaults_closed():
    cb = CircuitBreaker()
    assert cb.can_execute("never_seen") is True
    status = cb.get_status("never_seen")
    assert status["state"] == "closed"
