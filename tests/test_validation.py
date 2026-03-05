"""Tests for input validation and sanitization."""

import json

import pytest

from aarnxen.core.validation import (
    sanitize_system_prompt,
    truncate_response,
    validate_json_input,
    validate_prompt,
    validate_temperature,
)


def test_validate_prompt_strips_whitespace():
    assert validate_prompt("  hello world  ") == "hello world"


def test_validate_prompt_rejects_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_prompt("   ")


def test_validate_prompt_rejects_too_long():
    with pytest.raises(ValueError, match="exceeds maximum length"):
        validate_prompt("a" * 101, max_length=100)


def test_validate_temperature_clamps():
    assert validate_temperature(-0.5) == 0.0
    assert validate_temperature(3.0) == 2.0
    assert validate_temperature(1.0) == 1.0


def test_validate_json_input_valid():
    result = validate_json_input('[{"task": "a"}, {"task": "b"}]')
    assert len(result) == 2
    assert result[0]["task"] == "a"


def test_validate_json_input_rejects_too_many():
    data = json.dumps(list(range(10)))
    with pytest.raises(ValueError, match="exceeds maximum"):
        validate_json_input(data, max_items=5)


def test_validate_json_input_rejects_invalid_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        validate_json_input("not json at all")


def test_sanitize_system_prompt_truncates():
    result = sanitize_system_prompt("a" * 200, max_length=100)
    assert len(result) == 100


def test_truncate_response_under_limit():
    text = "short response"
    assert truncate_response(text, max_chars=1000) == text


def test_truncate_response_over_limit():
    text = "a" * 150
    result = truncate_response(text, max_chars=100)
    assert result.startswith("a" * 100)
    assert "[Response truncated at 100 characters]" in result
