<p align="center">
  <img src="docs/assets/backonthelangchain_banner.png" alt="backonthelangchain" width="100%">
</p>

# backonthelangchain

`backonthelangchain` is an observable prototype of a combined AI support flow.
It connects mandatory safety, simulated system evidence, fast classification,
retrieval, response generation, escalation, and LangGraph orchestration in one
inspectable application.

The repository remains useful for learning System 1-style classifiers and
routers, System 2-style response generation, and RAG. The unified application
is organized like a real support workflow so a run can be tested and debugged
from its structured inputs, decisions, evidence, route, and output. It is a
prototype, not a production-ready support service.

## Python-only quick start

Yes, the project works without the GUI or a running FastAPI server, and the
command-line examples do not require Node.js. The shortest path runs the
safety-gated support router from a terminal.

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

The web application visualizes the full support graph, streams live node state,
and lets you select each completed node to inspect its application-level
contribution. In addition to the Python requirements, install Node.js 20.9 or
newer and npm.

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

The retrieval stage uses OpenAI embeddings and runs only for technical requests
when the option is enabled in the interface. Its bundled documents are clearly
labeled fictional demo knowledge.

## Compare observable scenarios

The interface includes scenarios for ordinary MFA support, repeated failed
attempts, business-critical checkout impact, an explicit human request, a
duplicate billing charge, and a reported checkout outage. The two checkout
samples configure the same report against Operational and Outage demo state so
the system evidence and Jev score can be compared without forcing a result.
Set a relevant service to Operational, Degraded, and Outage in turn to compare
contradicted, partially corroborated, and corroborated evidence.

The accounting workstation sample is a direct RAG A/B comparison. Run it with
RAG off to give response generation only generic support context. Run it with
RAG on to retrieve the fictional **Accounting Workstation Recovery** document,
which supplies the organization-specific Acme Report Writer procedure. The
procedure is not embedded in the response prompt or application routing logic.

For escalation, compare ordinary failure, repeated failure, critical business
impact, and the explicit human request. The UI shows Jev's escalation
probability, the configured `0.80` threshold, whether it was met, and whether
the application detected explicit human-request language. Detection is
observational and does not override Jev.

## What the unified example demonstrates

- **OpenAI Moderation** is mandatory and authoritative before routing or
  response generation.
- **Simulated system and tool evidence** records configured service state and
  whether the user's report is corroborated, contradicted, partially
  corroborated, or not applicable. It is demo evidence, not monitoring data.
- **Jev** is the routing and escalation classifier. The application exposes its
  route confidence, route probabilities, escalation probability, thresholds,
  fallback use, and threshold results without substituting a desired outcome.
- **RAG** optionally retrieves from distinct fictional organization procedures
  for accounting workstations, VPN certificates, warehouse scanners, and
  meeting room displays. Ranked documents, scores, snippets, and the exact
  supplied context are visible.
- **The response-generation LLM** receives allowlisted application context:
  the query, selected route, system evidence, retrieved knowledge, and
  escalation state. Its generated response is recorded as the stage output.
- **LangGraph** remains the orchestrator and the executable source of truth for
  the graph description, execution path, and streamed node ids.
- **Human escalation** occurs only when Jev's escalation score meets the
  configured threshold. Repeated failure or critical impact may still score
  below it, and that limitation remains visible.

Every completed run includes a deterministic "What happened?" summary and
final-result provenance. Both are derived from structured stage evidence. No
model is asked to explain its hidden reasoning, and the API never exposes
prompts, chain-of-thought, raw provider objects, credentials, request headers,
private reasoning tokens, stack traces, or sensitive internal errors.

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
