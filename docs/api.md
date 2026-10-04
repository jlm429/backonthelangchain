# API reference

Start the backend from the repository root:

```bash
poetry run uvicorn backonthelangchain.api.app:app \
  --env-file .env --reload --host 127.0.0.1 --port 8000
```

Provider-backed routes require `OPENAI_API_KEY` and `TYPESAFE_API_KEY`. The
graph-description route does not call providers.

## Describe the support graph

```http
GET /api/support/graph?faq_retrieval=false
```

The response includes:

- `graph_id`
- the resolved `options`
- nodes with `id`, label, description, kind, required state, enabled state, and
  stage
- executable edges with source, target, conditional state, and branch label

The description comes from the same graph builder used for execution. Set
`faq_retrieval=true` to include the optional retrieval node in the executable
path.

## Run the support graph

```http
POST /api/support/run
Content-Type: application/json
```

All four simulated component states are required:

```json
{
  "query": "Authentication is down and I cannot log in.",
  "options": {
    "faq_retrieval": false
  },
  "simulated_status": {
    "authentication": "outage",
    "billing": "operational",
    "checkout": "operational",
    "api": "operational"
  }
}
```

Valid status values are `operational`, `degraded`, and `outage`. FAQ retrieval
is the only optional stage. Unknown request and option fields are rejected.

The response uses `text/event-stream`. A local request can be inspected with:

```bash
curl -N http://127.0.0.1:8000/api/support/run \
  -H 'Content-Type: application/json' \
  --data '{
    "query": "Authentication is down and I cannot log in.",
    "options": {"faq_retrieval": false},
    "simulated_status": {
      "authentication": "outage",
      "billing": "operational",
      "checkout": "operational",
      "api": "operational"
    }
  }'
```

Each SSE frame uses the event name `execution` and contains one JSON object.
Possible object types are:

- `run_started`, with a request-scoped run id and resolved options
- `node_started`, with a graph node id
- `node_completed`, with a node id and allowlisted application output
- `node_failed`, with a node id and generic message
- `run_completed`, with the normalized result and executed path
- `run_failed`, with a generic message

The completed result separates safety, simulated system status, routing, Jev
observations, optional retrieval, the answer, and the executed graph path.

## Compatibility routes

The focused Jev example remains available through the original registry API:

```http
GET /api/examples
POST /api/examples/jev-support-router/run
```

The run request is `{ "query": "..." }`. It returns a JSON envelope with the
example id and normalized result. These routes are retained for compatibility;
the current browser interface uses the unified support routes.

## Validation, limits, and errors

- Queries are trimmed, must contain non-whitespace text, and are limited to
  2,000 characters.
- Request bodies are limited to 16 KiB, including streamed or chunked bodies.
- Provider-backed runs are limited to ten requests per client IP per minute by
  a bounded in-memory limiter.
- CORS allows exact configured origins. Localhost port 3000 is allowed by
  default.
- Responses use `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`,
  and `Referrer-Policy: no-referrer`.

Errors use a stable generic envelope:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The request contains invalid or missing fields."
  }
}
```

The API does not return configuration values, request headers, stack traces,
raw provider errors, or provider response objects.
