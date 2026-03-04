# AarnXen-MultiAI-MCP

A lean, fast multi-AI MCP server with parallel consensus, response caching, cost tracking, and persistent memory.

Built as a better alternative to PAL MCP Server — fewer tools, less context bloat, faster consensus, and real cost visibility.

## Features

- **7 Tools** — chat, consensus, compare, think, codereview, costs, list_models
- **Parallel Consensus** — query 3+ models simultaneously via `asyncio.gather` (not sequential like PAL)
- **6 Providers** — Gemini, OpenAI, Ollama (local + cloud), Groq, OpenRouter, any OpenAI-compatible endpoint
- **Response Cache** — TTL + LRU cache saves cost on repeated queries
- **Cost Tracking** — per-request and cumulative USD cost with per-model breakdown
- **Retry + Fallback** — exponential backoff with automatic provider fallback
- **Persistent Memory** — SQLite-backed conversation threading that survives restarts
- **YAML Config** — single file instead of 200+ env vars

## Quick Start

```bash
# Clone
git clone https://github.com/AarnXen/aarnxen-multiai-mcp.git
cd aarnxen-multiai-mcp

# Install
uv sync

# Configure
mkdir -p ~/.aarnxen
cp config.example.yaml ~/.aarnxen/config.yaml
# Edit ~/.aarnxen/config.yaml with your API keys

# Test
uv run pytest tests/ -v
```

## Add to Claude Code

```bash
claude mcp add aarnxen -s user -- uv --directory /path/to/aarnxen-multiai-mcp run aarnxen
```

Or add to `~/.claude/settings.json` manually:

```json
{
  "mcpServers": {
    "aarnxen": {
      "command": "uv",
      "args": ["--directory", "/path/to/aarnxen-multiai-mcp", "run", "aarnxen"],
      "env": {
        "GEMINI_API_KEY": "your-key",
        "OLLAMA_CLOUD_KEY": "your-key"
      }
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `chat` | Chat with any AI model |
| `consensus` | Query multiple models in parallel, get all responses |
| `compare` | Side-by-side comparison of two models |
| `think` | Deep step-by-step reasoning (light/medium/deep) |
| `codereview` | Code review with focus options (general/security/performance/bugs) |
| `costs` | Session cost summary and cache stats |
| `list_models` | List all available models across providers |

## Usage Examples

```
# Chat with Gemini
"Use aarnxen chat to ask gemini-2.5-flash about Python async patterns"

# Get consensus from 3 models
"Use aarnxen consensus to ask about microservices vs monolith"

# Compare two models
"Use aarnxen compare with model_a=gemini-2.5-flash and model_b=groq to evaluate this code"

# Deep reasoning
"Use aarnxen think with depth=deep about this race condition"

# Code review
"Use aarnxen codereview with focus=security on this authentication code"

# Check spending
"Use aarnxen costs"
```

## Configuration

See `config.example.yaml` for full reference. Key sections:

```yaml
providers:
  - name: gemini
    api_key_env: GEMINI_API_KEY    # Read from env var
    priority: 1                     # Lower = higher priority
    models: [gemini-2.5-pro]

cache:
  enabled: true
  ttl_seconds: 3600                # 1 hour cache

memory:
  enabled: true
  path: "~/.aarnxen/conversations.db"

cost_tracking: true
```

## Supported Providers

| Provider | Auth | Notes |
|----------|------|-------|
| Gemini | API key | Native SDK, 1M+ token context |
| OpenAI | API key | GPT-4o, o3, etc. |
| Ollama Local | None | `http://localhost:11434` |
| Ollama Cloud | API key | `https://ollama.com` |
| Groq | API key | Fast inference (Llama, Mixtral) |
| OpenRouter | API key | 100+ models via single key |
| Custom | Optional | Any OpenAI-compatible endpoint |

## vs PAL MCP Server

| | PAL | AarnXen |
|---|---|---|
| Context footprint | 90KB+ (16 tools) | ~15KB (7 tools) |
| Consensus | Sequential (N round-trips) | Parallel (1 round-trip) |
| Caching | None | TTL + LRU |
| Cost tracking | None | Per-request + cumulative |
| Retry/fallback | None | Exponential backoff + chain |
| Config | 200+ env vars | Single YAML |
| Memory | In-memory (lost on restart) | SQLite persistent |
| Ollama | OpenAI shim | Native API |

## License

MIT
