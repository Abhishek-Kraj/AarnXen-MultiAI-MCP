"""Per-model pricing data (USD per 1M tokens)."""

# (input_price_per_million, output_price_per_million)
PRICING: dict[tuple[str, str], tuple[float, float]] = {
    # Gemini
    ("gemini", "gemini-2.5-pro"): (1.25, 10.00),
    ("gemini", "gemini-2.5-flash"): (0.15, 0.60),
    ("gemini", "gemini-2.0-flash"): (0.10, 0.40),
    # OpenAI
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "o3"): (10.00, 40.00),
    ("openai", "o3-mini"): (1.10, 4.40),
    # Groq
    ("groq", "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("groq", "mixtral-8x7b-32768"): (0.24, 0.24),
    ("groq", "llama-3.1-8b-instant"): (0.05, 0.08),
    # OpenRouter (varies by model — common ones)
    ("openrouter", "anthropic/claude-sonnet-4"): (3.00, 15.00),
    ("openrouter", "google/gemini-2.5-pro"): (1.25, 10.00),
    ("openrouter", "deepseek/deepseek-r1"): (0.55, 2.19),
    # Ollama (local = free, cloud = subscription-based, $0 per token)
    ("ollama", "llama3.2"): (0.0, 0.0),
    ("ollama-local", "llama3.2"): (0.0, 0.0),
    ("ollama-cloud", "llama3.2"): (0.0, 0.0),
    ("ollama-cloud", "deepseek-v3.2"): (0.0, 0.0),
    ("ollama-cloud", "qwen3.5"): (0.0, 0.0),
    ("ollama-cloud", "qwen3-coder-next"): (0.0, 0.0),
    ("ollama-cloud", "qwen3-next"): (0.0, 0.0),
    ("ollama-cloud", "kimi-k2.5"): (0.0, 0.0),
    ("ollama-cloud", "kimi-k2-thinking"): (0.0, 0.0),
    ("ollama-cloud", "glm-5"): (0.0, 0.0),
    ("ollama-cloud", "glm-4.7"): (0.0, 0.0),
    ("ollama-cloud", "minimax-m2.5"): (0.0, 0.0),
    ("ollama-cloud", "cogito-2.1"): (0.0, 0.0),
    ("ollama-cloud", "devstral-2"): (0.0, 0.0),
    ("ollama-cloud", "devstral-small-2"): (0.0, 0.0),
}


def get_pricing(provider: str, model: str) -> tuple[float, float]:
    """Get (input_price_per_M, output_price_per_M) for a model."""
    key = (provider, model)
    if key in PRICING:
        return PRICING[key]
    # Try partial match
    for (p, m), price in PRICING.items():
        if p == provider and model.startswith(m.split("-")[0]):
            return price
    return (0.0, 0.0)
