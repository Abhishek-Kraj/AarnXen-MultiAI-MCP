"""YAML config loader with Pydantic validation."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    name: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    models: list[str] = Field(default_factory=list)
    priority: int = 100
    enabled: bool = True
    timeout_seconds: int = 120
    max_retries: int = 3


class CacheConfig(BaseModel):
    enabled: bool = True
    max_size: int = 200
    ttl_seconds: int = 3600


class MemoryConfig(BaseModel):
    enabled: bool = True
    path: str = "~/.aarnxen/conversations.db"
    max_turns: int = 50


class RateLimitConfig(BaseModel):
    max_calls: int = 60
    window_seconds: float = 60.0


class AarnXenConfig(BaseModel):
    providers: list[ProviderConfig] = Field(default_factory=list)
    default_model: str = "auto"
    default_temperature: float = 0.7
    cache: CacheConfig = CacheConfig()
    memory: MemoryConfig = MemoryConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    cost_tracking: bool = True
    log_level: str = "INFO"


CONFIG_SEARCH_PATHS = [
    Path("config.yaml"),
    Path("aarnxen.yaml"),
    Path.home() / ".aarnxen" / "config.yaml",
]


def load_config(path: Optional[str] = None) -> AarnXenConfig:
    """Load config from YAML file, resolving env var API keys."""
    if path:
        config_path = Path(path)
    else:
        config_path = None
        for p in CONFIG_SEARCH_PATHS:
            if p.exists():
                config_path = p
                break

    if config_path and config_path.exists():
        raw = yaml.safe_load(config_path.read_text())
        cfg = AarnXenConfig(**(raw or {}))
    else:
        cfg = AarnXenConfig()

    _resolve_env_keys(cfg)
    _add_env_providers(cfg)
    return cfg


def _resolve_env_keys(cfg: AarnXenConfig) -> None:
    """Replace api_key_env references with actual env values."""
    for p in cfg.providers:
        if p.api_key_env and not p.api_key:
            p.api_key = os.environ.get(p.api_key_env)


def _add_env_providers(cfg: AarnXenConfig) -> None:
    """Auto-detect providers from env vars if no config file exists."""
    if cfg.providers:
        return

    env_providers = [
        ("gemini", "GEMINI_API_KEY", 1, ["gemini-2.5-flash", "gemini-2.5-pro"]),
        ("openai", "OPENAI_API_KEY", 2, ["gpt-4o"]),
        ("groq", "GROQ_API_KEY", 3, ["llama-3.3-70b-versatile"]),
        ("openrouter", "OPENROUTER_API_KEY", 4, []),
    ]

    for name, env_key, priority, models in env_providers:
        if os.environ.get(env_key):
            cfg.providers.append(
                ProviderConfig(
                    name=name,
                    api_key=os.environ[env_key],
                    priority=priority,
                    models=models,
                )
            )
