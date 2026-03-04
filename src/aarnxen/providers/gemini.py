"""Google Gemini provider using native SDK."""

import time
from typing import Optional

from google import genai
from google.genai import types

from aarnxen.providers.base import BaseProvider, ModelCapability, ModelResponse


class GeminiProvider(BaseProvider):

    def __init__(self, api_key: str, models: list[str] = None, timeout: int = 120):
        self._client = genai.Client(api_key=api_key)
        self._models = models or ["gemini-2.5-flash"]
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        start = time.monotonic()

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 8192,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        elapsed = (time.monotonic() - start) * 1000

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return ModelResponse(
            text=response.text or "",
            model=model,
            provider=self.provider_name(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
        )

    def list_models(self) -> list[ModelCapability]:
        return [
            ModelCapability(
                model_id=m,
                display_name=m,
                max_context=1_048_576 if "pro" in m else 1_048_576,
                input_price_per_m=1.25 if "pro" in m else 0.15,
                output_price_per_m=5.00 if "pro" in m else 0.60,
            )
            for m in self._models
        ]

    def provider_name(self) -> str:
        return "gemini"

    async def health_check(self) -> bool:
        try:
            self._client.models.generate_content(
                model=self._models[0],
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return True
        except Exception:
            return False
