"""Base provider interface and response models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelResponse:
    """Standardized response from any provider."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelCapability:
    """What a specific model can do."""

    model_id: str
    display_name: str
    max_context: int = 128_000
    input_price_per_m: float = 0.0
    output_price_per_m: float = 0.0


class BaseProvider(ABC):
    """Contract every provider must fulfill."""

    @abstractmethod
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
        ...

    @abstractmethod
    def list_models(self) -> list[ModelCapability]:
        ...

    @abstractmethod
    def provider_name(self) -> str:
        ...

    async def health_check(self) -> bool:
        """Check if provider is reachable. Override for custom logic."""
        return True
