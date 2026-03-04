"""Retry with exponential backoff and provider fallback."""

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Optional

from aarnxen.providers.base import BaseProvider, ModelResponse

if TYPE_CHECKING:
    from aarnxen.core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class RetryError(Exception):
    def __init__(self, message: str, attempts: list[Exception]):
        super().__init__(message)
        self.attempts = attempts


async def call_with_retry(
    provider: BaseProvider,
    model: str,
    prompt: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    fallback_providers: Optional[list[tuple[BaseProvider, str]]] = None,
    circuit_breaker: Optional["CircuitBreaker"] = None,
    **kwargs,
) -> ModelResponse:
    """Call provider with exponential backoff, then try fallbacks."""
    errors: list[Exception] = []
    primary_name = provider.provider_name()

    # Check circuit breaker for primary provider
    if circuit_breaker and not circuit_breaker.can_execute(primary_name):
        logger.warning(
            "Circuit open for %s, skipping to fallbacks", primary_name,
        )
    else:
        for attempt in range(max_retries):
            try:
                result = await provider.generate(prompt, model, **kwargs)
                if circuit_breaker:
                    circuit_breaker.record_success(primary_name)
                return result
            except Exception as e:
                errors.append(e)
                if circuit_breaker:
                    circuit_breaker.record_failure(primary_name)
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), max_delay) * (0.5 + random.random())
                    logger.warning(
                        "%s/%s attempt %d/%d failed: %s. Retrying in %.1fs",
                        primary_name, model, attempt + 1, max_retries, e, delay,
                    )
                    await asyncio.sleep(delay)

    for fb_provider, fb_model in (fallback_providers or []):
        fb_name = fb_provider.provider_name()
        if circuit_breaker and not circuit_breaker.can_execute(fb_name):
            logger.warning("Circuit open for %s, skipping fallback", fb_name)
            continue
        try:
            logger.info("Falling back to %s/%s", fb_name, fb_model)
            result = await fb_provider.generate(prompt, fb_model, **kwargs)
            if circuit_breaker:
                circuit_breaker.record_success(fb_name)
            return result
        except Exception as e:
            errors.append(e)
            if circuit_breaker:
                circuit_breaker.record_failure(fb_name)

    raise RetryError(f"All {len(errors)} attempts failed", errors)
