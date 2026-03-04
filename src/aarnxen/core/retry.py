"""Retry with exponential backoff and provider fallback."""

import asyncio
import logging
from typing import Optional

from aarnxen.providers.base import BaseProvider, ModelResponse

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
    **kwargs,
) -> ModelResponse:
    """Call provider with exponential backoff, then try fallbacks."""
    errors: list[Exception] = []

    for attempt in range(max_retries):
        try:
            return await provider.generate(prompt, model, **kwargs)
        except Exception as e:
            errors.append(e)
            if attempt < max_retries - 1:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    "%s/%s attempt %d/%d failed: %s. Retrying in %.1fs",
                    provider.provider_name(), model, attempt + 1, max_retries, e, delay,
                )
                await asyncio.sleep(delay)

    for fb_provider, fb_model in (fallback_providers or []):
        try:
            logger.info("Falling back to %s/%s", fb_provider.provider_name(), fb_model)
            return await fb_provider.generate(prompt, fb_model, **kwargs)
        except Exception as e:
            errors.append(e)

    raise RetryError(f"All {len(errors)} attempts failed", errors)
