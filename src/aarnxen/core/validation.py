"""Input validation and sanitization for MCP server tools."""

import json


def validate_prompt(prompt: str, max_length: int = 100_000) -> str:
    cleaned = prompt.strip()
    if not cleaned:
        raise ValueError("Prompt cannot be empty")
    if len(cleaned) > max_length:
        raise ValueError(f"Prompt exceeds maximum length of {max_length} characters")
    return cleaned


def validate_temperature(temp: float) -> float:
    return max(0.0, min(2.0, temp))


def validate_json_input(raw: str, max_items: int = 100) -> list:
    if len(raw) > 500_000:
        raise ValueError(f"JSON input too large: {len(raw)} bytes (max 500000)")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError("JSON input must be an array")
    if not parsed:
        raise ValueError("JSON array cannot be empty")
    if len(parsed) > max_items:
        raise ValueError(f"JSON array exceeds maximum of {max_items} items")
    return parsed


def sanitize_system_prompt(prompt: str, max_length: int = 10_000) -> str:
    cleaned = prompt.strip()
    return cleaned[:max_length]


def truncate_response(text: str, max_chars: int = 50_000) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[Response truncated at {max_chars} characters]"
    return text
