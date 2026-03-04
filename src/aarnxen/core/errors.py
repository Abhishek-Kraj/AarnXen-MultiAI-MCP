"""Structured error responses for graceful LLM recovery."""


def rate_limit_error(provider: str, retry_after: int = 30, alternatives: list = None):
    return {
        "isError": True,
        "error_type": "rate_limited",
        "message": f"Provider {provider} rate limited.",
        "recovery_hint": f"Retry after {retry_after}s, or use model='auto' for a different provider.",
        "retry_after_seconds": retry_after,
        "alternative_models": alternatives or [],
    }


def provider_unavailable(provider: str, alternatives: list = None):
    return {
        "isError": True,
        "error_type": "provider_unavailable",
        "message": f"Provider {provider} is temporarily unavailable.",
        "recovery_hint": "Use model='auto' to route to an available provider.",
        "available_alternatives": alternatives or [],
    }


def model_not_found(model: str, available: list = None):
    return {
        "isError": True,
        "error_type": "model_not_found",
        "message": f"Model '{model}' not found.",
        "recovery_hint": f"Available models: {', '.join(available[:5]) if available else 'use model=auto'}",
        "available_models": available or [],
    }


def generation_failed(provider: str, model: str, error: str, alternatives: list = None):
    return {
        "isError": True,
        "error_type": "generation_failed",
        "message": f"Failed to generate response from {provider}/{model}: {error}",
        "recovery_hint": "Try again with model='auto' or specify a different model.",
        "available_alternatives": alternatives or [],
    }
