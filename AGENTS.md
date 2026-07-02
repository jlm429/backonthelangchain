# Project agent memory

This file is the project's durable agent guide. Keep it focused on build,
test, architecture, release, and sharp-edge notes that should travel with the
code.

Add project-specific notes only when they are discovered through real work and
are likely to remain useful.

## Project purpose

`backonthelangchain` is a teaching and experimentation repository for building
LLM-powered applications with LangGraph, RAG, routing, tools, safety gates, and
related agentic AI patterns.

Prefer clear examples over abstract frameworks. The repository should help
readers understand practical patterns they can adapt.

## Repository map

- `src/backonthelangchain/agents/` contains LangGraph routing, nodes, prompts,
  tool definitions, schemas, and service adapters.
- `src/backonthelangchain/rag/` contains loading, chunking, embedding,
  retrieval, reranking, metadata, prompts, and RAG pipelines.
- `src/backonthelangchain/utils/` contains shared environment and tracing
  helpers.
- `examples/` contains runnable teaching scripts. Prefer adding or updating
  these when behavior is meant to be demonstrated.
- `tests/` contains automated tests. Add focused tests when behavior changes.
- `skills/` contains agent-facing workflow guidance for this repository.

## Security and secrets

Agents must treat local credentials as off-limits.

Do not read, print, summarize, modify, copy, or commit:

- `.env`
- `.env.*`
- API keys
- tokens
- credentials files
- service account files
- LangSmith keys
- OpenAI keys
- Voyage keys
- GitHub tokens

Use `.env.example` for documenting required environment variables.

If a task requires a new secret, update `.env.example` with the variable name
only, never the value.

Never include secret values in:

- source code
- examples
- tests
- docs
- command output
- commit messages
- PR summaries

If a secret is accidentally exposed, stop and alert the human reviewer.

## Environment configuration

This project uses local environment variables for provider credentials.

Expected local-only files:

- `.env`
- `.env.local`

Expected committed file:

- `.env.example`

Agents may inspect `.env.example` but must not inspect `.env`.

## Development rules

- Keep changes small and reviewable.
- Prefer explicit, readable code over clever abstractions.
- Do not change public APIs unless explicitly requested.
- Do not rewrite unrelated modules.
- Add or update tests when behavior changes.
- Update documentation when user-facing behavior changes.
- Prefer runnable scripts/examples over notebook-only workflows.
- Keep examples focused on one concept at a time.
- Keep optional provider integrations optional unless the task is specifically
  about that provider.
- Do not modify application code when the request is limited to agent harness
  files such as `AGENTS.md`, `skills/`, or `scripts/`.

## Documentation

- Keep CHANGELOG.md up to date for meaningful repository changes.
- Record features, workflow improvements, and behavior changes—not minor edits or formatting changes.

## Dependency notes

- The project uses Poetry.
- Base install: `poetry install`
- RAG examples need optional dependencies: `poetry install -E rag`
- Notebook work needs optional dependencies: `poetry install -E notebooks`
- Runtime credentials belong in local-only `.env` or `.env.local` files.

## Validation

Before proposing changes, run the full repository check when applicable:

```bash
./scripts/check.sh
```

The check script runs Ruff and pytest through Poetry.

For narrowly scoped documentation or agent-harness-only changes, the full check
may still be useful but is not always required. If it is skipped or cannot run,
state the reason clearly in the final response.

For targeted iteration, useful commands include:

```bash
poetry run ruff check .
poetry run pytest
```

