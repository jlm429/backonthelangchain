# Changelog

## Unreleased

### Features

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
  outage evidence supplied to Jev and routing context.
- Added a Next.js and TypeScript interface backed by a FastAPI example registry.
- Added structured browser results that preserve Jev provider observations
  separately from fallback routing controls, and shared the reusable Python
  runner between the API and CLI.
- Added bounded request rate limiting, exact-origin CORS, server-side
  configuration validation, query and streaming request-body limits, and safe
  API error envelopes.
- Added a standalone, safety-gated Jev support-router example with confidence-based
  fallback to the existing router and deterministic human escalation.

### Tests

- Added fake-provider behavioral coverage for graph serialization, live events,
  executed branches, optional retrieval, non-disableable required nodes, all
  simulated status levels, Jev context, and outage corroboration semantics.
- Added frontend tests for streamed event parsing and execution-state updates.
- Added fake-provider coverage for registry behavior, request validation,
  chunked request limits, Jev execution and fallback output, invalid ids,
  missing configuration, rate limiting, and secret-safe provider failures.

### Documentation

- Documented the unified graph, live visualization, optional FAQ branch,
  simulated status evidence, routing context, and a path to real monitoring
  integrations.
- Revised the README positioning and badges.
- Documented the web architecture, local backend and frontend setup, security
  model, deployment limitations, and example registration workflow.
- Documented changelog guidance for documentation-only changes.
- Created changelog to track project changes going forward.
