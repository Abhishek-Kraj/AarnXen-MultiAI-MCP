"""AarnXen Multi-AI MCP Server — entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from mcp.server.fastmcp import Context, FastMCP

from aarnxen.config import AarnXenConfig, load_config
from aarnxen.core.cache import ResponseCache
from aarnxen.core.circuit_breaker import CircuitBreaker
from aarnxen.core.conversation import ConversationMemory
from aarnxen.core.cost import CostTracker
from aarnxen.core.extractor import EntityExtractor
from aarnxen.core.knowledge import KnowledgeBase
from aarnxen.providers.registry import ProviderRegistry

logger = logging.getLogger("aarnxen")


@dataclass
class AppContext:
    registry: ProviderRegistry
    cache: Optional[ResponseCache]
    cost_tracker: CostTracker
    memory: Optional[ConversationMemory]
    knowledge: Optional[KnowledgeBase]
    circuit_breaker: CircuitBreaker
    extractor: EntityExtractor
    config: AarnXenConfig


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    cfg = load_config()

    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
    logger.info("AarnXen Multi-AI MCP Server starting...")

    registry = ProviderRegistry.from_config(cfg)

    cache = None
    if cfg.cache.enabled:
        cache = ResponseCache(max_size=cfg.cache.max_size, ttl_seconds=cfg.cache.ttl_seconds)

    cost_tracker = CostTracker()

    memory = None
    if cfg.memory.enabled:
        memory = ConversationMemory(persist_path=cfg.memory.path)

    knowledge = KnowledgeBase()
    circuit_breaker = CircuitBreaker()
    extractor = EntityExtractor(knowledge_base=knowledge)

    try:
        yield AppContext(
            registry=registry,
            cache=cache,
            cost_tracker=cost_tracker,
            memory=memory,
            knowledge=knowledge,
            circuit_breaker=circuit_breaker,
            extractor=extractor,
            config=cfg,
        )
    finally:
        if memory:
            memory.close()
        knowledge.close()
        logger.info("AarnXen shut down.")


mcp = FastMCP(
    "aarnxen",
    lifespan=app_lifespan,
    instructions=(
        "Multi-model AI orchestration server with 20+ models (Gemini, DeepSeek, Qwen, "
        "Kimi, GLM, MiniMax, Ollama). Use for: getting second opinions from external AI "
        "models (chat, consensus, compare), deep reasoning with step-by-step analysis "
        "(think), code review before committing (precommit, codereview), challenging your "
        "own reasoning (challenge), and persistent knowledge storage (kb_store, kb_search, "
        "kb_recall). Smart routing auto-picks the best model per task. Set cascade=True "
        "on chat for automatic cheap-to-premium escalation."
    ),
)


# --- Tool registration ---

from aarnxen.tools.chat import chat_handler  # noqa: E402
from aarnxen.tools.codereview import codereview_handler  # noqa: E402
from aarnxen.tools.compare import compare_handler  # noqa: E402
from aarnxen.tools.consensus import consensus_handler  # noqa: E402
from aarnxen.tools.costs import costs_handler  # noqa: E402
from aarnxen.tools.knowledge import (  # noqa: E402
    kb_get_handler,
    kb_recall_handler,
    kb_relate_handler,
    kb_remember_handler,
    kb_search_handler,
    kb_stats_handler,
    kb_store_handler,
)
from aarnxen.tools.challenge import challenge_handler  # noqa: E402
from aarnxen.tools.pipeline import pipeline_handler  # noqa: E402
from aarnxen.tools.precommit import precommit_handler  # noqa: E402
from aarnxen.tools.think import think_handler  # noqa: E402


@mcp.tool()
async def chat(
    prompt: str,
    model: str = "auto",
    temperature: float = 0.7,
    system_prompt: str = "",
    conversation_id: str = "",
    cascade: bool = False,
    ctx: Context = None,
) -> dict:
    """Chat with any AI model (GPT, Gemini, Groq, Ollama). Use when you need a second opinion, want to delegate a question to a different model, or need to query an external AI. Set cascade=True for smart routing (tries cheap model first, escalates if needed). Returns the model's response with cost and token usage."""
    return await chat_handler(prompt, model, temperature, system_prompt, conversation_id, cascade, ctx)


@mcp.tool()
async def consensus(
    prompt: str,
    models: str = "auto",
    temperature: float = 0.7,
    system_prompt: str = "",
    ctx: Context = None,
) -> dict:
    """Query multiple AI models in parallel and synthesize responses. Use when you need high-confidence answers, want to validate reasoning across models, or need diverse perspectives on a technical question. Returns all model responses for comparison."""
    return await consensus_handler(prompt, models, temperature, system_prompt, ctx)


@mcp.tool()
async def compare(
    prompt: str,
    model_a: str = "",
    model_b: str = "",
    temperature: float = 0.7,
    system_prompt: str = "",
    ctx: Context = None,
) -> dict:
    """Compare two AI models side-by-side. Use when you want to see how different models handle the same prompt, or benchmark model quality for a specific task."""
    return await compare_handler(prompt, model_a, model_b, temperature, system_prompt, ctx)


@mcp.tool()
async def think(
    prompt: str,
    model: str = "auto",
    depth: str = "medium",
    ctx: Context = None,
) -> dict:
    """Deep reasoning — delegates complex analysis to an external AI model for step-by-step thinking. Use for architecture decisions, debugging strategies, trade-off analysis, or any problem that benefits from extended reasoning. Depth: light/medium/deep."""
    return await think_handler(prompt, model, depth, ctx)


@mcp.tool()
async def codereview(
    code: str,
    language: str = "auto",
    model: str = "auto",
    focus: str = "general",
    ctx: Context = None,
) -> dict:
    """Review code for bugs, style, performance, and security using an external AI model. Use before committing code, during pull request review, or when you want a second opinion on code quality. Supports focus areas: general, security, performance, bugs."""
    return await codereview_handler(code, language, model, focus, ctx)


@mcp.tool()
async def precommit(
    diff: str,
    model: str = "auto",
    ctx: Context = None,
) -> dict:
    """Validate code changes before committing. Pass a git diff to catch bugs, security vulnerabilities, performance issues, and style problems. Returns a PASS/FAIL verdict with specific issues found. Use this before every git commit."""
    return await precommit_handler(diff, model, ctx)


@mcp.tool()
async def challenge(
    claim: str,
    evidence: str = "",
    model: str = "auto",
    ctx: Context = None,
) -> dict:
    """Challenge a claim or approach with critical analysis from another AI model. Use when you want to stress-test your reasoning, find flaws in a plan, or get a devil's advocate perspective before proceeding. Returns STRONG/MODERATE/WEAK verdict."""
    return await challenge_handler(claim, evidence, model, ctx)


@mcp.tool()
async def pipeline(
    steps: str,
    ctx: Context = None,
) -> dict:
    """Execute a pipeline of AarnXen tools in sequence. Each step's output feeds the next via $PREV. Example: [{"tool": "think", "args": {"prompt": "analyze X", "depth": "deep"}}, {"tool": "challenge", "args": {"claim": "$PREV"}}]. Available tools: chat, think, challenge, codereview, precommit."""
    return await pipeline_handler(steps, ctx)


@mcp.tool()
async def costs(ctx: Context = None) -> dict:
    """Show session cost summary and cache stats. Use to check how much you've spent on AI API calls and how effective the cache is."""
    return await costs_handler(ctx)


@mcp.tool()
async def list_models(ctx: Context = None) -> dict:
    """List all available models across all providers."""
    deps = ctx.request_context.lifespan_context
    return {"models": deps.registry.list_all_models()}


@mcp.tool()
async def provider_health(ctx: Context = None) -> dict:
    """Show circuit breaker health status for all AI providers. Use to check which providers are healthy, degraded, or down."""
    deps = ctx.request_context.lifespan_context
    return deps.circuit_breaker.get_all_status()


@mcp.tool()
async def kb_consolidate(ctx: Context = None) -> dict:
    """Run memory maintenance: deduplicate, decay old relations, prune stale data."""
    deps = ctx.request_context.lifespan_context
    return deps.knowledge.consolidate()


# --- Knowledge Base Tools ---


@mcp.tool()
async def kb_store(
    title: str,
    content: str,
    doc_type: str = "note",
    tags: str = "",
    source: str = "",
    ctx: Context = None,
) -> dict:
    """Store a document/note/fact in the knowledge base. Use this to remember important information, project decisions, or facts for future reference."""
    return await kb_store_handler(title, content, doc_type, tags, source, ctx)


@mcp.tool()
async def kb_search(query: str, limit: int = 5, ctx: Context = None) -> dict:
    """Search the knowledge base. Use to find previously stored notes, facts, or documents by keyword or topic."""
    return await kb_search_handler(query, limit, ctx)


@mcp.tool()
async def kb_get(doc_id: str, ctx: Context = None) -> dict:
    """Retrieve a specific document by ID."""
    return await kb_get_handler(doc_id, ctx)


@mcp.tool()
async def kb_remember(
    entity: str,
    fact: str,
    entity_type: str = "concept",
    ctx: Context = None,
) -> dict:
    """Remember a fact about an entity (person, project, concept)."""
    return await kb_remember_handler(entity, fact, entity_type, ctx)


@mcp.tool()
async def kb_recall(query: str, ctx: Context = None) -> dict:
    """Recall everything known about an entity or topic. Use to retrieve all stored facts, relationships, and observations about a person, project, or concept."""
    return await kb_recall_handler(query, ctx)


@mcp.tool()
async def kb_relate(
    from_entity: str,
    to_entity: str,
    relation: str,
    ctx: Context = None,
) -> dict:
    """Create a relationship between two entities."""
    return await kb_relate_handler(from_entity, to_entity, relation, ctx)


@mcp.tool()
async def kb_stats(ctx: Context = None) -> dict:
    """Show knowledge base statistics."""
    return await kb_stats_handler(ctx)


# --- MCP Resources (read-only context) ---


@mcp.resource("aarnxen://models")
def models_resource() -> str:
    """List of all configured AI models and their providers."""
    from aarnxen.providers.ollama import CLOUD_MODELS
    import json
    models = {
        "gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "ollama-cloud": list(CLOUD_MODELS.keys()),
    }
    return json.dumps(models, indent=2)


@mcp.resource("aarnxen://config")
def config_resource() -> str:
    """Current AarnXen configuration (providers, cache, memory settings)."""
    from pathlib import Path
    config_path = Path.home() / ".aarnxen" / "config.yaml"
    if config_path.exists():
        return config_path.read_text()
    return "No config found at ~/.aarnxen/config.yaml"


@mcp.resource("aarnxen://tiers")
def tiers_resource() -> str:
    """Smart routing tier definitions -- which models are used for which task types."""
    import json
    from aarnxen.core.router import TIERS, TASK_TO_TIER
    return json.dumps({"tiers": TIERS, "task_routing": TASK_TO_TIER}, indent=2)


# --- MCP Prompts (workflow templates) ---


@mcp.prompt()
def pre_commit_review(diff: str) -> str:
    """Review code changes before committing. Paste your git diff."""
    return (
        "Review this git diff for bugs, security issues, and code quality. "
        "Give a PASS/FAIL verdict with specific issues found.\n\n"
        f"```diff\n{diff}\n```"
    )


@mcp.prompt()
def architecture_decision(question: str, context: str = "") -> str:
    """Get deep analysis for an architecture decision."""
    prompt = f"Analyze this architecture decision with trade-offs, pros/cons, and a recommendation:\n\nQuestion: {question}"
    if context:
        prompt += f"\n\nContext: {context}"
    return prompt


@mcp.prompt()
def second_opinion(topic: str, your_analysis: str = "") -> str:
    """Get a second opinion from another AI model on any topic."""
    prompt = f"Provide an independent analysis of: {topic}"
    if your_analysis:
        prompt += f"\n\nPrevious analysis to compare against:\n{your_analysis}"
    return prompt


@mcp.prompt()
def debug_error(error_message: str, code_context: str = "") -> str:
    """Get help debugging an error from another AI model."""
    prompt = f"Debug this error. Explain the root cause and suggest fixes.\n\nError:\n```\n{error_message}\n```"
    if code_context:
        prompt += f"\n\nRelevant code:\n```\n{code_context}\n```"
    return prompt


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
