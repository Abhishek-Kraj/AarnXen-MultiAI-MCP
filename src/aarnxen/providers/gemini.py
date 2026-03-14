"""Google Gemini provider using native SDK."""

import base64
import time
from typing import Optional

from google import genai
from google.genai import types

from aarnxen.providers.base import BaseProvider, ModelCapability, ModelResponse


class GeminiProvider(BaseProvider):

    def __init__(self, api_key: str, models: list[str] = None, timeout: int = 120):
        self._client = genai.Client(api_key=api_key)
        self._models = models or ["gemini-2.5-flash", "gemini-2.5-pro"]
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[dict]] = None,
    ) -> ModelResponse:
        start = time.monotonic()

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 8192,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        if images:
            parts = []
            for img in images:
                parts.append(types.Part.from_bytes(
                    data=base64.b64decode(img["data"]),
                    mime_type=img["mime_type"],
                ))
            parts.append(types.Part.from_text(text=prompt))
            contents = parts
        else:
            contents = prompt

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
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

    def _model_pricing(self, m: str) -> tuple[float, float, int]:
        """Return (input_price, output_price, context_window) for a Gemini model."""
        if "3.1-pro" in m:
            return 1.25, 10.00, 1_048_576
        if "3.1-flash-lite" in m or "3-flash" in m:
            return 0.075, 0.30, 1_048_576
        if "2.5-flash-lite" in m:
            return 0.025, 0.10, 1_048_576
        if "pro" in m:
            return 1.25, 10.00, 1_048_576
        return 0.15, 0.60, 1_048_576

    def list_models(self) -> list[ModelCapability]:
        return [
            ModelCapability(
                model_id=m,
                display_name=m,
                max_context=self._model_pricing(m)[2],
                input_price_per_m=self._model_pricing(m)[0],
                output_price_per_m=self._model_pricing(m)[1],
            )
            for m in self._models
        ]

    def provider_name(self) -> str:
        return "gemini"

    async def health_check(self) -> bool:
        """Lightweight health check — counts tokens instead of generating content."""
        try:
            await self._client.aio.models.count_tokens(
                model=self._models[0],
                contents="health check",
            )
            return True
        except Exception:
            return False
