# Backend bootstrap

Status: Implemented  
Scope: repository and runtime foundations only

## Included

- Python 3.12 project using `uv`, a lockfile, and a `src` package layout.
- Immutable Pydantic v2 settings loaded from process environment.
- Structured JSON logging with recursive redaction.
- FastAPI application factory, lifespan, typed errors, request context, and explicit
  dependency container.
- Process-only liveness and dependency-aware readiness endpoints.
- Ruff, mypy strict mode, pytest, pytest-asyncio, and local pre-commit hooks.

## Intentionally deferred

The authoritative implementation plan leaves provider, vector dimension, durable queue,
retention, and confidence thresholds unresolved. This bootstrap therefore does not add
database models/migrations, queue consumers, parsers, normalization, RAG, LangGraph,
MCP, authentication, chat APIs, or external telemetry exporters.

Future integrations register a `NamedReadinessProbe` in `AppDependencies`; liveness
must remain free of external calls. Long-running ingestion must use the selected durable
queue and worker, never FastAPI `BackgroundTasks`.

## Configuration policy

Deployment configuration owns environment values. Secret-bearing settings are typed as
`SecretStr`; code must not serialize settings into logs, responses, traces, or exception
messages. Add new configuration names to the root README without publishing values or
creating dotenv templates.

## Authoritative references

Architecture changes belong in the sibling `auction-lens-ai/docs` documents linked from
`docs/README.md`. Local records describe implementation choices only and do not
supersede those contracts.
