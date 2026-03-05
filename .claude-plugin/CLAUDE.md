# AarnXen Auto-Memory

This plugin automatically captures session observations into AarnXen's knowledge base.

## How It Works

1. **SessionStart**: Searches KB for relevant past context about the current project
2. **Stop**: Summarizes session work and stores key observations as KB documents
3. **PreCompact**: Saves important context before conversation compaction

## Knowledge Base Tools Available

- `mcp__aarnxen__kb_store` — Store documents/observations
- `mcp__aarnxen__kb_search` — Search past observations
- `mcp__aarnxen__kb_recall` — Recall everything about an entity
- `mcp__aarnxen__kb_remember` — Remember facts about entities
- `mcp__aarnxen__kb_relate` — Create entity relationships

## Observation Types

Use these doc_type values when storing:
- `bugfix` — Bug fixes with root cause
- `feature` — New features built
- `discovery` — Important codebase/API findings
- `decision` — Architectural/design decisions
- `change` — Configuration or infrastructure changes
- `refactor` — Code restructuring

## Naming Convention

Titles should follow: `[project-name] Concise description`
Tags should include: project name, type, date (YYYY-MM-DD)
Source should be: `auto-memory` or `auto-memory-precompact`
