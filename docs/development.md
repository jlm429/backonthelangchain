# Development

## Requirements

- Python 3.10 through 3.13
- Poetry
- Node.js 20.9 or newer and npm for the frontend

The package uses optional dependency groups so focused examples do not require
every provider integration.

| Purpose | Install command | Required variables |
| --- | --- | --- |
| Base Python routers | `poetry install` | `OPENAI_API_KEY` |
| Tests and lint | `poetry install -E dev` | none |
| Jev routing and web backend | `poetry install -E jev` | `OPENAI_API_KEY`, `TYPESAFE_API_KEY` |
| FAQ retrieval | `poetry install -E rag` | `OPENAI_API_KEY` |
| Standalone Voyage-reranked RAG example | `poetry install -E rag` | `OPENAI_API_KEY`, `VOYAGE_API_KEY` |
| Notebooks | `poetry install -E notebooks` | depends on the notebook |

Extras can be combined, for example:

```bash
poetry install -E dev -E jev -E rag
```

Copy `.env.example` to `.env` and populate only the variables needed for the
workflow you are running. Never commit local credential files.

## Python examples

All scripts run from the repository root and accept an optional query. Without
one, they prompt interactively.

| Script | Demonstrates | Install |
| --- | --- | --- |
| `examples/run_support_router.py` | basic OpenAI routing and responses, without moderation | base |
| `examples/run_safe_support_router.py` | OpenAI Moderation followed by routing | base |
| `examples/run_safe_jev_support_router.py` | moderation, Jev routing, fallback, and escalation | `jev` |
| `examples/run_safe_rag_support_router.py` | moderation, routing, FAISS retrieval, and Voyage reranking | `rag` |

Example:

```bash
poetry run python examples/run_safe_jev_support_router.py \
  "I was charged twice for my subscription."
```

These scripts are focused teaching tools. The web application uses the unified
graph described in [Architecture](architecture.md).

## Repository map

- `src/backonthelangchain/agents/` contains graphs, nodes, schemas, prompts,
  tools, and provider-facing services.
- `src/backonthelangchain/api/` contains FastAPI routes, schemas, configuration,
  and rate limiting.
- `src/backonthelangchain/examples/` contains reusable execution and registry
  adapters shared by scripts and HTTP routes.
- `src/backonthelangchain/rag/` contains loading, chunking, embeddings,
  retrieval, reranking, metadata, prompts, and pipelines.
- `examples/` contains runnable command-line demonstrations.
- `web/` contains the Next.js graph viewer and stream consumer.
- `tests/` contains fake-provider Python coverage.

## Validation

Install development dependencies, then run the Python checks:

```bash
poetry install -E dev
./scripts/check.sh
```

The check script runs Ruff and pytest. Tests use fake providers and do not need
local credentials or make paid calls.

Run the frontend checks after installing its locked dependencies:

```bash
cd web
npm ci
npm test
npm run lint
npm run typecheck
npm run build
```

## Extending examples

Keep new teaching examples focused on one concept and prefer runnable scripts
over notebook-only workflows. Provider and business logic belongs under
`src/`; command-line scripts should remain thin adapters.

For a compatibility API example, expose an async handler that accepts a query
and returns a JSON-compatible dictionary, then register one
`ExampleDefinition`. Move blocking provider work to a worker thread, list
required configuration by variable name, and add fake-provider tests for both
the service and route.

For unified graph changes, keep the executable graph builder authoritative.
Derive graph descriptions from the compiled graph, preserve stable node ids
across backend events and frontend state, and allowlist only safe application
outputs in streamed events.
