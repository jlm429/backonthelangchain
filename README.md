# backonthelangchain

An interactive workshop for a production-minded support system built with
Python, LangGraph, FastAPI, Next.js, OpenAI, and Jev.

The web application centers one end-to-end support graph. It shows the real
backend graph before a query runs, then updates node states from LangGraph's
event stream. The interface exposes application-level decisions and structured
outputs only. It never displays model chain-of-thought or hidden reasoning.

## Unified support graph

```text
START / user query
  |
  v
OpenAI Moderation (required and authoritative)
  | flagged                         | allowed
  v                                 v
blocked response             simulated status evidence
  |                                 |
  |                                 v
  |                         Jev support routing
  |                    with existing OpenAI fallback
  |                         /        |        \
  |                        /         |         \
  |                escalation    billing    technical
  |                    |             |           |
  |                    |             |     optional FAQ retrieval
  |                    |             |           |
  |                    v             v           v
  +------------------------------ response / handoff
                                      |
                                      v
                                     END
```

The graph reuses the repository's existing components:

- `OpenAIModerationSafetyService` for the mandatory first gate
- `JevSupportRouterService` and fixed `jev-1.13.0` questions
- the existing confidence-based OpenAI fallback router
- deterministic human escalation
- existing technical and structured billing response services
- `TechSupportRAGPipeline` for the optional technical FAQ branch

`GET /api/support/graph` compiles the same Python graph builder used for
execution and serializes its nodes and edges. Shared node metadata supplies
readable labels. A disabled optional node remains in the description with
`enabled: false`, while executable edges come from the compiled graph and
bypass that node.

The legacy focused Python examples remain available for teaching individual
concepts, but the web interface no longer presents an example catalog.

## Live execution

`POST /api/support/run` returns Server-Sent Events. FastAPI consumes
LangGraph's `astream_events(..., version="v2")` stream and emits safe lifecycle
events for application nodes:

- `run_started`
- `node_started`
- `node_completed`
- `node_failed`
- `run_completed`
- `run_failed`

Event node identifiers are the same identifiers returned by the graph
description. The browser represents waiting, running, completed, failed, and
disabled states and preserves the actual executed path after completion.
Provider internals, prompts, raw SDK responses, stack traces, and hidden model
reasoning are not streamed.

## Simulated system status

The UI includes independently selectable demo state for Authentication,
Billing, Checkout, and API. Each can be Operational, Degraded, or Outage.

These values are simulated evidence. They do not come from monitoring systems
and must not be interpreted as real service health. The browser sends them as a
structured request object. A required Python node compares the user's report
with the selected demo state and distinguishes:

- a user report with no component-specific signal
- a report corroborated by simulated outage state
- a report partially corroborated by simulated degraded state
- a report not corroborated by simulated operational state

The structured evidence is passed to Jev and the OpenAI fallback router as
context, not as a hard-coded routing or escalation result. It is also returned
in the normalized application result.

A future integration can replace the simulated evidence builder with an
authenticated monitoring or status-page adapter while retaining the same
graph state contract. Real integrations should add time stamps, source
identity, freshness checks, authorization, failure handling, and audit logs.

## Optional FAQ retrieval

Tier 1 FAQ retrieval is the only caller-configurable graph stage. When enabled,
the backend compiles a technical branch that runs the existing OpenAI embedding
and FAISS pipeline before technical response generation. When disabled, the
compiled branch connects Jev routing directly to technical response and the
retrieval node does not execute.

OpenAI Moderation and every required node cannot be disabled through the API.
Unknown option fields are rejected. RAG execution requires the optional RAG
dependencies installed with `poetry install -E rag`.

## Architecture

```text
Browser at localhost:3000
  |
  | same-origin /backend requests
  v
Next.js server rewrite
  |
  v
FastAPI at 127.0.0.1:8000
  |
  +-- graph description from compiled Python LangGraph
  |
  +-- streamed LangGraph application events
  |
  v
OpenAI Moderation, Jev, OpenAI models, and optional RAG
```

All provider calls stay in Python. Next.js contains presentation and stream
consumption only. It does not reproduce the routing graph or provider logic.

## Local setup

Requirements:

- Python 3.10 through 3.13
- Poetry
- Node.js 20.9 or newer
- npm

Install the base application, development tools, and Jev integration:

```bash
poetry install -E dev -E jev
```

To enable the optional FAQ stage, also install RAG dependencies:

```bash
poetry install -E rag
```

Create a local environment file from the committed template:

```bash
cp .env.example .env
```

Set these provider values locally:

```text
OPENAI_API_KEY
TYPESAFE_API_KEY
```

Never commit the populated file. Start FastAPI from the repository root:

```bash
poetry run uvicorn backonthelangchain.api.app:app \
  --env-file .env --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. Next.js proxies `/backend/*` to
`http://127.0.0.1:8000/api/*`. Set the server-only `BACKEND_API_URL` to use a
different backend address. Do not prefix it with `NEXT_PUBLIC_`.

## API

- `GET /api/support/graph?faq_retrieval=false` returns the browser-safe graph.
- `POST /api/support/run` accepts a query, graph options, and simulated status,
  then streams `text/event-stream` execution events.
- Legacy `GET /api/examples` and `POST /api/examples/{id}/run` routes remain for
  compatibility with the existing focused example adapter.

Queries must contain non-whitespace text and are limited to 2,000 characters.
The API uses generic error envelopes and never returns credentials,
configuration values, headers, stack traces, or raw provider errors.

## Security model

`OPENAI_API_KEY` and `TYPESAFE_API_KEY` remain backend-only. They are read from
server environment variables and never included in graph metadata, events,
API results, browser bundles, or `NEXT_PUBLIC_` variables. Real `.env` files
remain ignored. `.env.example` contains variable names only.

Other protections include:

- mandatory OpenAI Moderation before status, Jev, retrieval, or response nodes
- exact-origin CORS with local development defaults
- strict request schemas that reject unknown graph controls
- a 2,000-character query limit and 16 KiB streaming body limit
- ten runs per client IP per minute with bounded in-memory tracking
- no-store and browser hardening headers
- opaque validation and provider failure responses

Before public deployment, add authentication, user-scoped quotas, a shared
rate limiter, HTTPS, network controls, abuse monitoring, and alerting. Multiple
API processes do not share the current in-memory rate limiter.

## Validation

Run Python checks:

```bash
./scripts/check.sh
```

Run frontend checks:

```bash
cd web
npm test
npm run lint
npm run typecheck
npm run build
```

Automated tests use fake providers and make no paid calls.
