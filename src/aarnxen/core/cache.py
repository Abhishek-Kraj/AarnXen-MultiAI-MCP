"""TTL + LRU response cache."""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from aarnxen.providers.base import ModelResponse


@dataclass
class CacheEntry:
    response: ModelResponse
    created_at: float
    hit_count: int = 0


class ResponseCache:
    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, provider: str, model: str, prompt: str,
                  system_prompt: str, temperature: float) -> str:
        raw = f"{provider}:{model}:{prompt}:{system_prompt}:{temperature}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, provider: str, model: str, prompt: str,
            system_prompt: str = "", temperature: float = 0.7) -> Optional[ModelResponse]:
        key = self._make_key(provider, model, prompt, system_prompt, temperature)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.monotonic() - entry.created_at > self._ttl:
            del self._store[key]
            self._misses += 1
            return None
        entry.hit_count += 1
        self._hits += 1
        self._store.move_to_end(key)
        resp = entry.response
        resp.cached = True
        return resp

    def put(self, provider: str, model: str, prompt: str,
            system_prompt: str, temperature: float, response: ModelResponse) -> None:
        key = self._make_key(provider, model, prompt, system_prompt, temperature)
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = CacheEntry(response=response, created_at=time.monotonic())

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }
