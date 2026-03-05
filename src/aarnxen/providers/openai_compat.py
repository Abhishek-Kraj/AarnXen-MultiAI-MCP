"""OpenAI-compatible provider — works with OpenAI, Groq, OpenRouter, vLLM, LM Studio, etc."""

import time
from typing import Optional

from openai import AsyncOpenAI

from aarnxen.providers.base import BaseProvider, ModelCapability, ModelResponse


class OpenAICompatProvider(BaseProvider):

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        models: list[str] = None,
        timeout: int = 120,
        name: str = "openai",
    ):
        kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**kwargs)
        self._models = models or []
        self._name = name

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

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                if isinstance(img, dict) and img.get("url"):
                    content.append({"type": "image_url", "image_url": {"url": img["url"]}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = await self._client.chat.completions.create(**kwargs)
        elapsed = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        return ModelResponse(
            text=choice.message.content or "",
            model=model,
            provider=self.provider_name(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed,
        )

    def list_models(self) -> list[ModelCapability]:
        return [
            ModelCapability(model_id=m, display_name=m)
            for m in self._models
        ]

    def provider_name(self) -> str:
        return self._name

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
