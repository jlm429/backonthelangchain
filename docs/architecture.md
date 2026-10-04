# Architecture

The repository contains focused Python examples and one observable prototype of
a combined support flow. The web application and primary support API both use
the unified graph. They do not duplicate its routing logic.

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
uses them to render waiting, running, completed, failed, and disabled states,
preserve the executed path, and attach backend-authored evidence to each
completed node. Completed nodes are selectable.

Each completed stage uses one typed evidence envelope with `stage_id`, `label`,
`summary`, `inputs`, and `outputs`. Nodes construct the allowlisted
application-level contribution, and the event adapter validates it before
serialization. The adapter does not stream prompts, raw SDK responses,
provider internals, configuration, request headers, stack traces, private
reasoning tokens, or model hidden reasoning.

The inspectable nodes expose:

| Node | Application evidence |
| --- | --- |
| User Query | validated query, selected simulated states, and RAG option |
| OpenAI Moderation | allowed or blocked, flagged state, model, and normalized result |
| Simulated Status | configured state, relevant services, per-service evidence relations, and an overall assessment including mixed evidence |
| Jev Support Routing | classified and selected routes, route confidence and probabilities, route and escalation thresholds, threshold results, and fallback use |
| Tier 1 FAQ Retrieval | retrieval query, result count, ranked demo documents, document ids, scores when available, snippets, and exact response context |
| Technical Response | query, route, structured system evidence, retrieved knowledge supplied, escalation state, production type, and generated response |
| Billing Response | query, route, system evidence visible to the application, escalation state, production type, and structured response |
| Human Escalation | the Jev-derived escalation state and deterministic handoff response |
| Blocked Response | the moderation decision and deterministic safe response |
| Final Result | outcome, answer, deterministic execution summary, and provenance |

When a provider decision is unavailable, the evidence says `unknown` and
records fallback use rather than inventing a reason. Jev remains authoritative
for both routing and escalation classification.

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
- a report contradicted by simulated operational state

The resulting context informs Jev and the OpenAI fallback. It does not directly
force a route or escalation. The normalized result returns both the selected
demo state and the derived evidence with a simulation notice.

A real status integration should replace the evidence builder with an
authenticated adapter while preserving the graph-state contract. It should add
source identity, timestamps, freshness checks, authorization, failure handling,
and audit logs.

## Optional demo knowledge retrieval

Retrieval is the only caller-configurable graph stage. When enabled, a technical
route runs the bundled fictional demo knowledge through OpenAI embeddings and a
FAISS index before technical response generation. The unified graph uses the
no-op reranker, so it does not require Voyage.

The demo knowledge base contains separate procedures for Accounting Workstation
Recovery, Field VPN Certificate Recovery, Warehouse Scanner Synchronization,
and Meeting Room Display Recovery. The accounting procedure is the only source
of its Acme Report Writer instructions. Those instructions are not present in
the response prompt or routing logic.

When disabled, the compiled technical branch connects routing directly to the
technical response. Moderation, status evaluation, routing, and response stages
cannot be disabled through the API. Unknown option fields are rejected.

## Execution summary and provenance

After a completed run, the backend derives a concise execution summary from the
same validated stage evidence returned by the graph. It reports moderation,
system evidence, Jev scores and threshold results, retrieval, response context,
and outcome. This is deterministic application narration, not generated
chain-of-thought.

The final result also records the user query, selected simulated state, status
assessment, Jev classified route, retrieved document names, escalation
outcome, and response production type. Changing a query, status selection, or
RAG setting therefore changes both the executed path and its visible
provenance.

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
