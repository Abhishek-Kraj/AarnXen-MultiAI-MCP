"""Tests for sliding window rate limiter."""

from unittest.mock import patch

import pytest

from aarnxen.core.rate_limit import RateLimiter, RateLimitExceeded, check_rate_limit


def test_allows_within_limit():
    rl = RateLimiter(max_calls=5, window_seconds=60.0)
    for _ in range(5):
        assert rl.allow() is True


def test_blocks_over_limit():
    rl = RateLimiter(max_calls=3, window_seconds=60.0)
    for _ in range(3):
        assert rl.allow() is True
    assert rl.allow() is False
    assert rl.allow() is False


def test_window_slides():
    base = 1000.0
    current_time = [base]

    def mock_monotonic():
        return current_time[0]

    with patch("aarnxen.core.rate_limit.time.monotonic", side_effect=mock_monotonic):
        rl = RateLimiter(max_calls=2, window_seconds=10.0)
        assert rl.allow() is True
        current_time[0] = base + 1.0
        assert rl.allow() is True
        current_time[0] = base + 2.0
        assert rl.allow() is False

        # Advance past the window so the first call expires
        current_time[0] = base + 11.0
        assert rl.allow() is True


def test_per_key_isolation():
    rl = RateLimiter(max_calls=2, window_seconds=60.0)
    assert rl.allow("user_a") is True
    assert rl.allow("user_a") is True
    assert rl.allow("user_a") is False
    # Different key should still be allowed
    assert rl.allow("user_b") is True
    assert rl.allow("user_b") is True
    assert rl.allow("user_b") is False


def test_remaining_count():
    rl = RateLimiter(max_calls=5, window_seconds=60.0)
    assert rl.remaining() == 5
    rl.allow()
    assert rl.remaining() == 4
    rl.allow()
    rl.allow()
    assert rl.remaining() == 2


def test_reset_clears_history():
    rl = RateLimiter(max_calls=2, window_seconds=60.0)
    rl.allow()
    rl.allow()
    assert rl.allow() is False
    rl.reset()
    assert rl.remaining() == 2
    assert rl.allow() is True


def test_check_rate_limit_raises():
    rl = RateLimiter(max_calls=1, window_seconds=60.0)
    check_rate_limit(rl)  # first call succeeds
    with pytest.raises(RateLimitExceeded):
        check_rate_limit(rl)


def test_rate_limit_exceeded_retry_after():
    base = 1000.0
    current_time = [base]

    def mock_monotonic():
        return current_time[0]

    with patch("aarnxen.core.rate_limit.time.monotonic", side_effect=mock_monotonic):
        rl = RateLimiter(max_calls=1, window_seconds=30.0)
        check_rate_limit(rl)

        current_time[0] = base + 10.0
        with pytest.raises(RateLimitExceeded) as exc_info:
            check_rate_limit(rl)
        # First call was at base, window is 30s, current time is base+10
        # retry_after = base + 30 - (base + 10) = 20.0
        assert exc_info.value.retry_after == pytest.approx(20.0, abs=0.1)
