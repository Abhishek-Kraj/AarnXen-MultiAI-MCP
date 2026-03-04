"""Token counting and cost tracking."""

import time
from dataclasses import dataclass

from aarnxen.pricing.models import get_pricing


@dataclass
class RequestCost:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float
    cached: bool = False


class CostTracker:
    def __init__(self):
        self._requests: list[RequestCost] = []
        self._session_start = time.time()

    def record(
        self, provider: str, model: str,
        input_tokens: int, output_tokens: int,
        cached: bool = False,
    ) -> RequestCost:
        pricing = get_pricing(provider, model)
        cost = 0.0
        if not cached:
            cost = (
                (input_tokens / 1_000_000) * pricing[0]
                + (output_tokens / 1_000_000) * pricing[1]
            )

        entry = RequestCost(
            provider=provider, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost, timestamp=time.time(), cached=cached,
        )
        self._requests.append(entry)
        return entry

    def summary(self) -> dict:
        total_cost = sum(r.cost_usd for r in self._requests)
        total_input = sum(r.input_tokens for r in self._requests)
        total_output = sum(r.output_tokens for r in self._requests)

        by_model: dict[str, dict] = {}
        for r in self._requests:
            key = f"{r.provider}/{r.model}"
            if key not in by_model:
                by_model[key] = {"requests": 0, "cost_usd": 0.0, "tokens": 0, "cached": 0}
            by_model[key]["requests"] += 1
            by_model[key]["cost_usd"] += r.cost_usd
            by_model[key]["tokens"] += r.input_tokens + r.output_tokens
            if r.cached:
                by_model[key]["cached"] += 1

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_requests": len(self._requests),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "by_model": by_model,
            "session_duration_s": round(time.time() - self._session_start, 1),
        }
