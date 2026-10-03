# Changelog

## Unreleased

### Features

- Added a Next.js and TypeScript interface backed by a FastAPI example registry.
- Added structured browser results for the existing safety-gated Jev support
  router and shared its reusable Python runner between the API and CLI.
- Added bounded request rate limiting, exact-origin CORS, server-side
  configuration validation, query limits, and safe API error envelopes.
- Added a standalone, safety-gated Jev support-router example with confidence-based
  fallback to the existing router and deterministic human escalation.

### Tests

- Added fake-provider coverage for registry behavior, request validation, Jev
  execution, invalid ids, missing configuration, rate limiting, and secret-safe
  provider failures.

### Documentation

- Revised the README positioning and badges.
- Documented the web architecture, local backend and frontend setup, security
  model, deployment limitations, and example registration workflow.
- Documented changelog guidance for documentation-only changes.
- Created changelog to track project changes going forward.
