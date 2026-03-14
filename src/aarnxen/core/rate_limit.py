"""Sliding window rate limiter for protecting against runaway AI agent loops."""
from __future__ import annotations

import time
from collections import defaultdict


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded, retry after {retry_after:.1f}s")


class RateLimiter:
    def __init__(self, max_calls: int = 60, window_seconds: float = 60.0):
        self._max_calls = max_calls
        self._window = window_seconds
        self._calls: defaultdict[str, list[float]] = defaultdict(list)

    def _purge(self, key: str, now: float) -> None:
        cutoff = now - self._window
        self._calls[key] = [t for t in self._calls[key] if t > cutoff]
        if not self._calls[key]:
            del self._calls[key]

    def allow(self, key: str = "global") -> bool:
        now = time.monotonic()
        self._purge(key, now)
        if len(self._calls[key]) >= self._max_calls:
            return False
        self._calls[key].append(now)
        return True

    def remaining(self, key: str = "global") -> int:
        now = time.monotonic()
        self._purge(key, now)
        return max(self._max_calls - len(self._calls.get(key, [])), 0)

    def reset(self, key: str = "global") -> None:
        self._calls[key].clear()


def check_rate_limit(limiter: RateLimiter, key: str = "global") -> None:
    if not limiter.allow(key):
        now = time.monotonic()
        oldest = min(limiter._calls[key])
        retry_after = oldest + limiter._window - now
        raise RateLimitExceeded(retry_after=max(retry_after, 0.0))
