---
name: backonthelangchain
description: General development workflow for contributing to the backonthelangchain repository. Use when editing Python source, examples, tests, docs, scripts, AGENTS.md, or repository skills unless a more specific skill applies.
user-invocable: false
tags:
  - langgraph
  - rag
  - agents
  - examples
  - python
---

# backonthelangchain Development

Use this skill for repository work in `backonthelangchain`.

## Working Principles

- Treat the repo as an educational codebase: optimize for examples that are
  runnable, readable, and easy to adapt.
- Prefer local patterns in `src/backonthelangchain/` over new abstractions.
- Keep changes scoped to the requested behavior.
- Do not change public APIs unless the user explicitly asks for it.
- Do not read local secret files. Use `.env.example` for documenting variable
  names only.

## Repository Orientation

- `src/backonthelangchain/agents/`: LangGraph workflows, nodes, schemas,
  prompts, tools, and domain service adapters.
- `src/backonthelangchain/rag/`: document loading, chunking, embeddings,
  retrieval, reranking, metadata, prompts, and RAG pipelines.
- `examples/`: runnable demonstrations for the concepts in the package.
- `tests/`: pytest coverage for behavior and import contracts.
- `AGENTS.md`: durable project instructions for agents.

## Workflow

1. Read `AGENTS.md` before making changes.
2. Inspect the relevant source, example, or test files before editing.
3. Keep edits small and avoid unrelated cleanup.
4. Add or update tests when behavior changes.
5. Update README or examples when user-facing behavior changes.
6. Run validation before finishing when practical.
7. In the final response, report changed files, checks run, and any remaining
   risk.

## Implementation Guidance

- For routing and agent behavior, keep graph transitions and schemas explicit.
- For RAG behavior, keep retrieval, reranking, and generation steps separable
  enough for readers to inspect.
- For examples, prefer command-line runnable scripts over notebook-only flows.
- For provider integrations, keep tracing and optional services opt-in unless
  the task is specifically about that provider.
- For harness-only tasks, do not modify application code under `src/`,
  `examples/`, or `tests/` unless the user expands the request.

## Validation

Run the full check when applicable:

```bash
./scripts/check.sh
```

The full check runs:

```bash
poetry run ruff check .
poetry run pytest
```

For documentation-only or agent-harness-only changes, the full check can be
skipped if it is not useful or cannot run. State that clearly in the final
response.
