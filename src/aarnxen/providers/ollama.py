"""Ollama provider — supports both local and cloud (ollama.com) via native API.

Ollama Cloud (ollama.com) hosts 20+ cloud models including DeepSeek, Qwen,
Kimi, GLM, MiniMax and more. Models are auto-discovered via /api/tags at
startup. Cloud pricing is subscription-based ($0/20/100 per month), NOT
per-token — effectively free per request.
"""

import logging
import time
from typing import Optional

import httpx

from aarnxen.providers.base import BaseProvider, ModelCapability, ModelResponse

logger = logging.getLogger(__name__)

# Well-known Ollama Cloud models with context sizes.
# Used as fallback when /api/tags is unreachable.
CLOUD_MODELS: dict[str, int] = {
    "deepseek-v3.2": 128_000,
    "qwen3.5": 128_000,
    "qwen3-coder-next": 128_000,
    "qwen3-next": 128_000,
    "qwen3-vl": 128_000,
    "kimi-k2.5": 128_000,
    "kimi-k2-thinking": 128_000,
    "glm-5": 128_000,
    "glm-4.7": 128_000,
    "glm-4.6": 128_000,
    "minimax-m2.5": 128_000,
    "minimax-m2": 128_000,
    "minimax-m2.1": 128_000,
    "cogito-2.1": 128_000,
    "devstral-2": 128_000,
    "devstral-small-2": 128_000,
    "nemotron-3-nano": 128_000,
    "rnj-1": 128_000,
    "ministral-3": 128_000,
    "gemini-3-flash-preview": 128_000,
    "llama3.2": 128_000,
}


class OllamaProvider(BaseProvider):

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        models: list[str] = None,
        timeout: int = 300,
        name: str = "ollama",
        auto_discover: bool = True,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._configured_models = models or []
        self._discovered_models: list[dict] = []
        self._timeout = timeout
        self._name = name
        self._auto_discover = auto_discover
        self._discovery_done = False

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

    async def discover_models(self) -> list[dict]:
        """Auto-discover available models via /api/tags endpoint."""
        if self._discovery_done:
            return self._discovered_models
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            self._discovered_models = models
            self._discovery_done = True
            names = [m.get("name", m.get("model", "")) for m in models]
            logger.info("Ollama auto-discovered %d models: %s", len(names), names[:10])
            return models
        except Exception as exc:
            logger.warning("Ollama model discovery failed: %s — using configured models", exc)
            self._discovery_done = True
            return []

    def list_models(self) -> list[ModelCapability]:
        # If discovery ran, use discovered models + configured models
        all_model_ids = set(self._configured_models)

        for m in self._discovered_models:
            name = m.get("name", m.get("model", ""))
            if name:
                all_model_ids.add(name)
                # Also add the base name without tag (e.g., "qwen3.5" from "qwen3.5:latest")
                if ":" in name:
                    all_model_ids.add(name.split(":")[0])

        # If nothing discovered yet, fall back to well-known cloud models for cloud provider
        if not self._discovered_models and not self._configured_models:
            is_cloud = "ollama.com" in self._base_url
            if is_cloud:
                all_model_ids.update(CLOUD_MODELS.keys())

        return [
            ModelCapability(
                model_id=m,
                display_name=m,
                max_context=CLOUD_MODELS.get(m, 128_000),
            )
            for m in sorted(all_model_ids)
        ]

    def provider_name(self) -> str:
        return self._name

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            if resp.status_code == 200:
                # Opportunistically discover models
                if not self._discovery_done:
                    data = resp.json()
                    self._discovered_models = data.get("models", [])
                    self._discovery_done = True
                return True
            return False
        except Exception:
            return False
