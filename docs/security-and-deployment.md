# Security and deployment

The repository demonstrates production-minded boundaries, but it is not a
production-ready service.

## Credential boundary

Provider credentials stay in the Python backend. The supported local variable
names are documented in `.env.example`:

- `OPENAI_API_KEY`
- `TYPESAFE_API_KEY`
- `VOYAGE_API_KEY` for the standalone Voyage-reranked RAG example
- `LANGSMITH_API_KEY` for optional tracing

Real `.env` and `.env.local` files are ignored and must never be committed.
Credentials are not included in graph metadata, application events, normalized
results, browser bundles, or `NEXT_PUBLIC_` variables.

`BACKEND_API_URL` is a server-only Next.js proxy setting. Use
`BACKONTHELANGCHAIN_CORS_ORIGINS` for a comma-separated list of exact browser
origins when the frontend is hosted separately.

## Current protections

- OpenAI Moderation is mandatory before status evaluation, Jev, retrieval, or
  response generation in the unified graph.
- Required graph stages cannot be disabled through request options.
- Request schemas reject unknown fields.
- Queries are limited to 2,000 characters and bodies to 16 KiB.
- Provider-backed runs are limited to ten requests per client IP per minute.
- The in-memory rate limiter stores a bounded number of client entries.
- CORS uses exact origins with safe local-development defaults.
- Responses set no-store and basic browser hardening headers.
- Validation and provider failures use opaque error messages.
- Streamed events validate typed, application-owned stage evidence and do not
  pass through provider objects. They exclude prompts, raw SDK responses,
  configuration, authorization information, request headers, stack traces,
  hidden model reasoning, and private reasoning tokens.
- Execution summaries are deterministic transformations of stage evidence, not
  model-generated explanations.
- Automated tests use injected fake providers and make no paid calls.

The simulated component status selected in the UI is demo input. It is not
real monitoring data and must not be represented as actual service health.

## Before public deployment

Add and verify at least:

- authentication and authorization
- user-scoped quotas and abuse controls
- a shared rate limiter for multiple API processes
- matching request limits at the reverse proxy
- HTTPS and restrictive network controls
- managed secret storage and rotation
- production-safe CORS origins
- observability, audit logs, abuse monitoring, and alerting
- provider timeout, retry, and availability policies
- retention and privacy policies for user inputs and outputs

The current per-process limiter is not shared across workers or hosts. Do not
rely on it as the only production quota or denial-of-service control.

A real system-status adapter also needs authenticated sources, source identity,
timestamps, freshness validation, failure handling, and an audit trail. The
demo's simulated evidence is intentionally insufficient for that role.
