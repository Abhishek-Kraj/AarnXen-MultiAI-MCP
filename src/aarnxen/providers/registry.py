"""Provider registry with priority ordering and fallback chains."""

from __future__ import annotations

import logging
from typing import Optional

from aarnxen.config import AarnXenConfig, ProviderConfig
from aarnxen.providers.base import BaseProvider, ModelResponse

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Manages providers with priority ordering and model resolution."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._configs: dict[str, ProviderConfig] = {}
        self._model_to_provider: dict[str, str] = {}

    @classmethod
    def from_config(cls, config: AarnXenConfig) -> ProviderRegistry:
        """Build registry from config, initializing available providers."""
        registry = cls()

        sorted_providers = sorted(config.providers, key=lambda p: p.priority)

        for pc in sorted_providers:
            if not pc.enabled:
                continue

            provider = _create_provider(pc)
            if provider is None:
                logger.warning("Skipping provider %s: no API key or misconfigured", pc.name)
                continue

            registry._providers[pc.name] = provider
            registry._configs[pc.name] = pc

            for model in pc.models:
                if model not in registry._model_to_provider:
                    registry._model_to_provider[model] = pc.name

        logger.info(
            "Registry initialized with %d providers: %s",
            len(registry._providers),
            list(registry._providers.keys()),
        )
        return registry

    def resolve(self, model: Optional[str] = None) -> tuple[BaseProvider, str]:
        """Resolve model request to (provider, model_id).

        If model is None or "auto", uses highest-priority provider's first model.
        If model matches a known model name, routes to its provider.
        If model matches a provider name, uses that provider's first model.
        """
        if not self._providers:
            raise RuntimeError("No providers configured. Check your config.yaml and API keys.")

        if not model or model == "auto":
            name = next(iter(self._providers))
            cfg = self._configs[name]
            resolved_model = cfg.models[0] if cfg.models else "default"
            return self._providers[name], resolved_model

        if model in self._model_to_provider:
            name = self._model_to_provider[model]
            return self._providers[name], model

        if model in self._providers:
            cfg = self._configs[model]
            resolved_model = cfg.models[0] if cfg.models else "default"
            return self._providers[model], resolved_model

        # Fuzzy match: check if any model contains the query
        for m, pname in self._model_to_provider.items():
            if model.lower() in m.lower():
                return self._providers[pname], m

        raise ValueError(
            f"Unknown model '{model}'. Available: {list(self._model_to_provider.keys())} "
            f"or providers: {list(self._providers.keys())}"
        )

    def get_top_n_models(self, n: int = 3) -> list[tuple[str, str]]:
        """Get top N models by provider priority. Returns (provider_name, model_id) pairs."""
        result = []
        for name, cfg in self._configs.items():
            if name in self._providers and cfg.models:
                result.append((name, cfg.models[0]))
                if len(result) >= n:
                    break
        return result

    def get_fallbacks(self, model: str) -> list[tuple[BaseProvider, str]]:
        """Get fallback providers for a model (all other providers in priority order)."""
        primary_provider = self._model_to_provider.get(model)
        fallbacks = []
        for name, cfg in self._configs.items():
            if name != primary_provider and name in self._providers and cfg.models:
                fallbacks.append((self._providers[name], cfg.models[0]))
        return fallbacks

    def list_all_models(self) -> list[dict]:
        """Aggregate models from all providers."""
        result = []
        for name, provider in self._providers.items():
            cfg = self._configs[name]
            for cap in provider.list_models():
                result.append({
                    "provider": name,
                    "model": cap.model_id,
                    "display_name": cap.display_name,
                    "max_context": cap.max_context,
                    "priority": cfg.priority,
                })
        return result


def _create_provider(pc: ProviderConfig) -> Optional[BaseProvider]:
    """Factory: create a provider instance from config."""
    name = pc.name.lower().replace("-", "_")

    if name == "gemini":
        if not pc.api_key:
            return None
        from aarnxen.providers.gemini import GeminiProvider
        return GeminiProvider(api_key=pc.api_key, models=pc.models, timeout=pc.timeout_seconds)

    if name in ("ollama", "ollama_local", "ollama_cloud"):
        from aarnxen.providers.ollama import OllamaProvider
        return OllamaProvider(
            base_url=pc.base_url or "http://localhost:11434",
            api_key=pc.api_key,
            models=pc.models,
            timeout=pc.timeout_seconds,
            name=pc.name,
        )

    if name == "openai":
        if not pc.api_key:
            return None
        from aarnxen.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=pc.api_key,
            models=pc.models,
            timeout=pc.timeout_seconds,
            name="openai",
        )

    if name == "groq":
        if not pc.api_key:
            return None
        from aarnxen.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=pc.api_key,
            base_url="https://api.groq.com/openai/v1",
            models=pc.models,
            timeout=pc.timeout_seconds,
            name="groq",
        )

    if name == "openrouter":
        if not pc.api_key:
            return None
        from aarnxen.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=pc.api_key,
            base_url="https://openrouter.ai/api/v1",
            models=pc.models,
            timeout=pc.timeout_seconds,
            name="openrouter",
        )

    # Generic: any provider with a base_url is treated as OpenAI-compatible
    if pc.base_url:
        from aarnxen.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=pc.api_key or "not-needed",
            base_url=pc.base_url,
            models=pc.models,
            timeout=pc.timeout_seconds,
            name=pc.name,
        )

    return None
