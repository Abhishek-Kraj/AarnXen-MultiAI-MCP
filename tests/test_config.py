"""Tests for config loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from aarnxen.config import AarnXenConfig, ProviderConfig, load_config


def test_default_config():
    cfg = AarnXenConfig()
    assert cfg.default_model == "auto"
    assert cfg.default_temperature == 0.7
    assert cfg.cache.enabled is True
    assert cfg.memory.enabled is True


def test_provider_config():
    pc = ProviderConfig(name="test", api_key="key123", priority=1, models=["model-1"])
    assert pc.name == "test"
    assert pc.api_key == "key123"
    assert pc.timeout_seconds == 120
    assert pc.max_retries == 3


def test_load_config_from_yaml(tmp_path):
    config_data = {
        "default_model": "gemini-2.5-flash",
        "providers": [
            {"name": "gemini", "api_key": "test-key", "priority": 1, "models": ["gemini-2.5-flash"]},
        ],
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))

    cfg = load_config(str(config_file))
    assert cfg.default_model == "gemini-2.5-flash"
    assert len(cfg.providers) == 1
    assert cfg.providers[0].name == "gemini"
    assert cfg.providers[0].api_key == "test-key"


def test_env_key_resolution():
    os.environ["TEST_API_KEY"] = "env-key-value"
    try:
        cfg = AarnXenConfig(
            providers=[ProviderConfig(name="test", api_key_env="TEST_API_KEY", priority=1)]
        )
        from aarnxen.config import _resolve_env_keys
        _resolve_env_keys(cfg)
        assert cfg.providers[0].api_key == "env-key-value"
    finally:
        del os.environ["TEST_API_KEY"]
