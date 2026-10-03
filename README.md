![Python](https://img.shields.io/badge/python-3.10%20to%203.13-blue.svg)
![Next.js](https://img.shields.io/badge/frontend-Next.js-black.svg)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-05998B.svg)
![Workflow](https://img.shields.io/badge/workflows-LangGraph-1B1B1B.svg)

# backonthelangchain

An interactive workshop for production-minded LLM application patterns. The web
application lets you choose an example from a backend catalog, run a query, and
inspect normalized safety, escalation, routing, and response data.

The first registered workflow is the Jev support router. OpenAI Moderation is
the authoritative first gate. Safe requests continue to the existing fixed Jev
`jev-1.13.0` decision service, then human escalation or support routing. Low
confidence Jev decisions and Jev failures use the existing OpenAI router.

## Architecture

```text
Browser
  |
  | same-origin /backend requests
  v
Next.js web server
  |
  | server-side rewrite
  v
FastAPI example registry
  |
  | async handler contract, blocking work moved to a worker thread
  v
Reusable Python Jev runner
  |
  v
Existing LangGraph workflow
  |
  +-- OpenAI Moderation
  +-- Jev escalation and route decision
  +-- human escalation, tech support, or billing
```

The Python runner in
`src/backonthelangchain/examples/jev_support.py` is shared by FastAPI and
`examples/run_safe_jev_support_router.py`. FastAPI never shells out to Python or
Poetry. The catalog in `src/backonthelangchain/examples/registry.py` is the only
example list. The frontend always fetches its selector, descriptions, defaults,
and samples from that catalog.

## Local setup

Requirements:

- Python 3.10 through 3.13
- Poetry
- Node.js 20.9 or newer
- npm

Install the Python application, test tools, and Jev integration:

```bash
poetry install -E dev -E jev
```

Create a local environment file from the committed template:

```bash
cp .env.example .env
```

Set these required values locally:

```text
OPENAI_API_KEY
TYPESAFE_API_KEY
```

Never commit the populated file. Start the backend from the repository root:

```bash
poetry run uvicorn backonthelangchain.api.app:app \
  --env-file .env --reload --host 127.0.0.1 --port 8000
```

In a second terminal, install and start the frontend:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. The Next.js server proxies browser requests to
`http://127.0.0.1:8000` by default. To use another backend address, set the
server-only `BACKEND_API_URL` before starting or building Next.js. It is not a
browser secret and is not prefixed with `NEXT_PUBLIC_`.

The reusable CLI adapter remains available:

```bash
poetry run python examples/run_safe_jev_support_router.py \
  "I cannot log in after enabling MFA."
```

## API

- `GET /api/examples` returns browser-safe registry metadata.
- `POST /api/examples/{id}/run` accepts `{ "query": "..." }` and returns a
  normalized result.

Queries must contain non-whitespace text and are limited to 2,000 characters.
The API returns stable error codes and generic messages. It never returns raw
provider errors, credentials, headers, stack traces, or configuration values.

## Security model

Provider calls and both required API keys stay in Python. Keys are read only
from server environment variables and are never included in catalog metadata,
application logs, browser bundles, `NEXT_PUBLIC_` variables, or API responses.
Real `.env` files remain ignored. `.env.example` contains variable names only.

The initial API also provides:

- OpenAI Moderation before Jev or support routing
- exact-origin CORS with local development origins by default
- a 2,000-character query limit
- a 16 KiB request body limit enforced while streaming, including chunked requests
- ten runs per client IP per minute
- a bounded in-process limiter that tracks at most 10,000 clients
- no-store and basic browser hardening headers
- safe request-validation and provider-failure responses

Set `BACKONTHELANGCHAIN_CORS_ORIGINS` to a comma-separated list of exact origins
when the frontend is hosted separately. A wildcard production origin is not
enabled.

The limiter is intentionally suitable only for this single-instance slice.
Before public deployment, add authentication or user-scoped quotas, a shared
rate limiter, a matching reverse-proxy body limit, HTTPS,
a restrictive network policy, abuse monitoring, and alerting.
Multiple API processes do not share the current in-memory counters.

## Add an example

1. Put provider and business logic in a reusable Python service under `src/`.
2. Expose one async handler that accepts a query and returns a JSON-compatible
   dictionary. Move blocking SDK calls to `asyncio.to_thread` when needed.
3. Create an `ExampleDefinition` in `build_example_registry` with its id,
   display name, description, default prompt, samples, handler, and required
   configuration variable names.
4. Add fake-provider tests for the service and API path. Tests must never make
   paid provider calls.

The frontend needs no new example list or selector code. It renders metadata
from the backend registry and displays the normalized result plus raw JSON.

The next straightforward additions are the existing safety-gated support
router, followed by the basic support router. The RAG support router is also a
good fit after its larger FAISS and Voyage dependency footprint is made an
explicit runtime option.

## Validation

Run the Python checks:

```bash
./scripts/check.sh
```

Run the frontend checks:

```bash
cd web
npm run lint
npm run typecheck
npm run build
npm audit
```

All provider tests use injected fakes. No local credentials are required for
the test suite.

## Other Python examples

The existing focused teaching scripts remain under `examples/`, including the
basic support router, safety-gated router, and RAG support router. Optional RAG
dependencies can be installed with:

```bash
poetry install -E rag
```
