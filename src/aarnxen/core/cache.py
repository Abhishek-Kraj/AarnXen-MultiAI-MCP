"""TTL + LRU response cache with optional semantic similarity."""
from __future__ import annotations

import copy
import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


from aarnxen.providers.base import ModelResponse


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


@dataclass
class CacheEntry:
    response: ModelResponse
    created_at: float
    hit_count: int = 0
    prompt: str = ""
    embedding: list[float] = field(default_factory=list)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class ResponseCache:
    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600,
                 embedder: Optional[Embedder] = None,
                 similarity_threshold: float = 0.92):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold

    def _make_key(self, provider: str, model: str, prompt: str,
                  system_prompt: str, temperature: float) -> str:
        raw = f"{provider}:{model}:{prompt}:{system_prompt}:{temperature}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _semantic_search(self, provider: str, model: str, prompt: str,
                         system_prompt: str, temperature: float,
                         query_embedding: list[float]) -> Optional[CacheEntry]:
        now = time.monotonic()
        best_entry = None
        best_sim = 0.0
        expired_keys = []
        for key, entry in self._store.items():
            if now - entry.created_at > self._ttl:
                expired_keys.append(key)
                continue
            if not entry.embedding:
                continue
            sim = _cosine_similarity(query_embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
                best_key = key
        for k in expired_keys:
            del self._store[k]
        if best_entry and best_sim >= self._similarity_threshold:
            best_entry.hit_count += 1
            self._store.move_to_end(best_key)
            return best_entry
        return None

    def get(self, provider: str, model: str, prompt: str,
            system_prompt: str = "", temperature: float = 0.7) -> Optional[ModelResponse]:
        key = self._make_key(provider, model, prompt, system_prompt, temperature)
        entry = self._store.get(key)
        if entry is not None:
            if time.monotonic() - entry.created_at > self._ttl:
                del self._store[key]
            else:
                entry.hit_count += 1
                self._hits += 1
                self._store.move_to_end(key)
                resp = copy.copy(entry.response)
                resp.cached = True
                return resp

        if self._embedder is not None:
            query_emb = self._embedder.embed(prompt)
            hit = self._semantic_search(provider, model, prompt,
                                        system_prompt, temperature, query_emb)
            if hit is not None:
                self._hits += 1
                resp = copy.copy(hit.response)
                resp.cached = True
                return resp

        self._misses += 1
        return None

    def put(self, provider: str, model: str, prompt: str,
            system_prompt: str, temperature: float, response: ModelResponse) -> None:
        key = self._make_key(provider, model, prompt, system_prompt, temperature)
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        embedding = self._embedder.embed(prompt) if self._embedder else []
        self._store[key] = CacheEntry(
            response=response, created_at=time.monotonic(),
            prompt=prompt, embedding=embedding,
        )

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "semantic_enabled": self._embedder is not None,
        }
