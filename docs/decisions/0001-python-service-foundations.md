# ADR 0001: Python service foundations

Status: Accepted  
Date: 2026-07-13

## Context

The backend needs safe, testable foundations before unresolved database, queue, model,
and public-source decisions are implemented. It must remain portable across managed
container platforms and follow the authoritative sibling architecture.

## Decision

- Use Python 3.12, `uv`, a committed lockfile, and a `src` package layout.
- Use a FastAPI application factory with an async lifespan and app-scoped explicit
  dependency container.
- Use Pydantic v2 settings from process environment only; represent credentials with
  secret types and never emit settings wholesale.
- Use structlog with JSON as the default and mandatory recursive redaction.
- Keep liveness process-only. Readiness runs bounded checks registered by enabled
  integrations.
- Use Ruff, strict mypy, pytest/pytest-asyncio, and repository-local pre-commit commands
  backed by the lockfile.

## Consequences

Integrations are injected and testable without global clients. The API can start before
later-phase dependencies are selected. Each integration must explicitly add lifecycle
management and a readiness probe; a configured value alone does not imply availability.

This record does not select or implement a database access library, migration strategy,
durable queue, parser, worker, model provider, or telemetry exporter.
