<p align="center">
  <img src="docs/assets/backonthelangchain_banner.png" alt="backonthelangchain" width="100%">
</p>

# backonthelangchain

`backonthelangchain` is a teaching repository for practical LLM application
patterns. It combines LangGraph workflows, safety gates, model-based routing,
optional RAG, a FastAPI streaming API, and a Next.js graph viewer in one
support-system example.

Use the focused Python scripts to learn one pattern at a time, or run the web
application to watch the unified support graph execute. This is an
experimentation project, not a production-ready support service.

## Python-only quick start

Yes, the project works without the GUI, Node.js, or FastAPI. The shortest path
runs the safety-gated support router from a terminal.

Requirements: Python 3.10 through 3.13, Poetry, and an OpenAI API key.

```bash
poetry install
cp .env.example .env
```

Set `OPENAI_API_KEY` in the local `.env` file, then run:

```bash
poetry run python examples/run_safe_support_router.py \
  "I cannot log in after enabling MFA."
```

The script runs OpenAI Moderation first, routes an allowed request to technical
or billing support, and prints the answer and safety result. Provider calls may
incur usage charges. Never commit the populated `.env` file.

Other terminal examples add Jev routing or RAG. See
[Python examples and dependencies](docs/development.md#python-examples).

## Web frontend quick start

The web application visualizes the full support graph and streams live node
status from the Python backend. In addition to the Python requirements, install
Node.js 20.9 or newer and npm.

Install the backend with the Jev integration:

```bash
poetry install -E jev
cp .env.example .env
```

Set these variables in `.env`:

```text
OPENAI_API_KEY
TYPESAFE_API_KEY
```

Start the backend from the repository root:

```bash
poetry run uvicorn backonthelangchain.api.app:app \
  --env-file .env --reload --host 127.0.0.1 --port 8000
```

In another terminal, start the frontend:

```bash
cd web
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The Next.js server proxies
`/backend/*` to `http://127.0.0.1:8000/api/*` by default.

To use the optional FAQ retrieval stage, install both runtime extras before
starting the backend:

```bash
poetry install -E jev -E rag
```

The FAQ stage uses OpenAI embeddings and runs only for technical requests when
the option is enabled in the interface.

## What the unified example demonstrates

- mandatory OpenAI Moderation before provider-backed routing or responses
- simulated system-status evidence for authentication, billing, checkout, and
  API incidents
- Jev routing and human-escalation scoring with an OpenAI fallback
- technical and structured billing responses
- optional OpenAI embedding and FAISS retrieval over a bundled Tier 1 FAQ
- a graph description generated from the executable LangGraph definition
- safe Server-Sent Events that expose application state, not hidden reasoning

The focused command-line examples remain independent teaching paths. The web
application centers the unified graph and does not present an example catalog.

## Documentation

- [Architecture](docs/architecture.md): graph flow, live events, status
  evidence, optional retrieval, and frontend/backend boundaries
- [API reference](docs/api.md): endpoints, request shapes, streaming events,
  limits, and compatibility routes
- [Security and deployment](docs/security-and-deployment.md): current
  protections, safety limitations, and production requirements
- [Development](docs/development.md): example matrix, optional dependencies,
  project layout, and validation commands

## License

No license file is currently included. Treat the repository as source-available
for inspection until the project owner adds explicit license terms.
