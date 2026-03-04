"""Ollama provider — supports both local and cloud (ollama.com) via native API."""

import time
from typing import Optional

import httpx

from aarnxen.providers.base import BaseProvider, ModelCapability, ModelResponse


class OllamaProvider(BaseProvider):

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        models: list[str] = None,
        timeout: int = 300,
        name: str = "ollama",
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._models = models or []
        self._timeout = timeout
        self._name = name

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

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

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            body["options"]["num_predict"] = max_tokens

        resp = await self._client.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()

        elapsed = (time.monotonic() - start) * 1000

        return ModelResponse(
            text=data.get("message", {}).get("content", ""),
            model=model,
            provider=self.provider_name(),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=elapsed,
        )

    def list_models(self) -> list[ModelCapability]:
        return [
            ModelCapability(model_id=m, display_name=m, max_context=128_000)
            for m in self._models
        ]

    def provider_name(self) -> str:
        return self._name

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
