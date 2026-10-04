# Architecture

The repository contains focused Python examples and one unified support graph.
The web application and primary support API both use the unified graph. They do
not duplicate its routing logic.

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

The graph composes these existing components:

- `OpenAIModerationSafetyService` as the mandatory first gate
- `JevSupportRouterService` with fixed `jev-1.13.0` questions
- the confidence-based OpenAI fallback router
- deterministic human escalation
- technical and structured billing response services
- `TechSupportRAGPipeline` for optional technical FAQ retrieval

`build_unified_support_graph` is the executable source of truth. The graph
description API compiles that same builder and serializes its nodes and edges.
Shared node metadata adds readable labels. A disabled optional node remains in
the description with `enabled: false`, while executable edges bypass it.

## Live execution

`POST /api/support/run` returns Server-Sent Events. FastAPI consumes LangGraph's
`astream_events(..., version="v2")` stream and emits only these
application-level lifecycle events:

- `run_started`
- `node_started`
- `node_completed`
- `node_failed`
- `run_completed`
- `run_failed`

Event node identifiers match the graph-description identifiers. The browser
uses them to render waiting, running, completed, failed, and disabled states and
to preserve the executed path after completion.

The event adapter allowlists useful application fields. It does not stream
prompts, raw SDK responses, provider internals, stack traces, or model hidden
reasoning.

Each web request receives a newly compiled, checkpoint-free graph. This keeps
request state isolated. The focused teaching graphs may use in-memory
checkpointers for their command-line demonstrations.

## Simulated system status

The unified workflow requires an explicit demo state for Authentication,
Billing, Checkout, and API. Each component can be `operational`, `degraded`, or
`outage`.

These values are simulated evidence, not monitoring data. A required graph node
compares the user's report with the selected state and distinguishes:

- no component-specific signal in the report
- a report corroborated by simulated outage state
- a report partially corroborated by simulated degraded state
- a report not corroborated by simulated operational state

The resulting context informs Jev and the OpenAI fallback. It does not directly
force a route or escalation. The normalized result returns both the selected
demo state and the derived evidence with a simulation notice.

A real status integration should replace the evidence builder with an
authenticated adapter while preserving the graph-state contract. It should add
source identity, timestamps, freshness checks, authorization, failure handling,
and audit logs.

## Optional FAQ retrieval

FAQ retrieval is the only caller-configurable graph stage. When enabled, a
technical route runs the bundled FAQ through OpenAI embeddings and a FAISS
index before technical response generation. The unified graph uses the no-op
reranker, so it does not require Voyage.

When disabled, the compiled technical branch connects routing directly to the
technical response. Moderation, status evaluation, routing, and response stages
cannot be disabled through the API. Unknown option fields are rejected.

## Application boundary

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
  +-- streamed application events from LangGraph execution
  |
  v
OpenAI Moderation, Jev, OpenAI models, and optional RAG
```

All provider calls remain in Python. Next.js owns presentation and stream
consumption only. `BACKEND_API_URL` can change the server-side proxy target; it
must not be exposed as a `NEXT_PUBLIC_` credential or used to move provider
calls into the browser.
