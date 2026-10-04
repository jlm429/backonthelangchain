# Changelog

## Unreleased

### Features

- Added backend-authoritative stage evidence for every executed unified-graph
  stage, selectable node inspection, deterministic execution summaries, and
  final-result provenance.
- Added visible Jev route and escalation thresholds with threshold results.
- Added fictional demo knowledge for accounting workstation recovery, VPN
  certificate recovery, warehouse scanner synchronization, and meeting room
  display recovery, plus paired RAG and system-evidence scenarios.
- Unified the web application around one support graph: mandatory OpenAI
  Moderation, simulated status evidence, Jev with the existing fallback,
  optional FAQ retrieval, and response or escalation.
- Added a graph-description API generated from the compiled Python LangGraph
  and a graph-first interface with readable nodes, edges, conditional paths,
  required and optional stages, and disabled-stage styling.
- Added live Server-Sent Events backed by LangGraph execution events for
  waiting, running, completed, failed, and disabled node states.
- Added independently selectable simulated Authentication, Billing, Checkout,
  and API state, plus structured comparison of reported and corroborating
  outage evidence supplied to Jev and routing context. All four selections are
  required at the API boundary.
- Preserved the FastAPI example registry and reusable Jev runner as legacy
  teaching interfaces alongside the unified web application.
- Added bounded request rate limiting, exact-origin CORS, server-side
  configuration validation, query and streaming request-body limits, and safe
  API error envelopes.
- Added a standalone, safety-gated Jev support-router example with confidence-based
  fallback to the existing router and deterministic human escalation.

### Tests

- Added deterministic fake-provider coverage for ordinary, repeated, critical,
  explicit-human, billing, contradictory-status, and accounting RAG scenarios,
  including evidence propagation, summary accuracy, and private-field omission.
- Added fake-provider behavioral coverage for graph serialization, live events,
  executed branches, optional retrieval, non-disableable required nodes, all
  simulated status levels, Jev context, and outage corroboration semantics.
- Added frontend tests for streamed event parsing and execution-state updates.
- Added fake-provider coverage for registry behavior, request validation,
  chunked request limits, Jev execution and fallback output, invalid ids,
  missing configuration, rate limiting, and secret-safe provider failures.

### Documentation

- Repositioned the unified application as an observable support-flow prototype
  and documented node evidence, scenario comparisons, demo knowledge,
  deterministic summaries, provenance, and hidden-reasoning boundaries.
- Added the project banner, concise Python-only and web quick starts, and
  focused architecture, API, security, deployment, and development guides.
- Updated the contributor harness with unified-graph invariants, current
  dependency boundaries, and frontend validation commands.
- Documented the unified graph, live visualization, optional FAQ branch,
  simulated status evidence, routing context, and a path to real monitoring
  integrations.
- Documented changelog guidance for documentation-only changes.
- Created changelog to track project changes going forward.
