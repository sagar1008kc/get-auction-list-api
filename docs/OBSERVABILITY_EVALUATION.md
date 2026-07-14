# Observability and evaluation

All exporters are optional. Prometheus remains local, and OpenTelemetry/Langfuse setup or
export failures are logged and cannot fail a request. Telemetry records identifiers,
bounded enums, counts, timings, tokens, and costs; it does not record prompts, response
bodies, raw tool payloads, authorization headers, or SQL arguments. The common sanitizer
masks secret keys, bearer/JWT values, email addresses, phone numbers, and street addresses.

## Configuration

Settings use the `GET_AUCTION_LIST_` prefix:

- `OTEL_ENABLED` (default `true`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` (optional OTLP/HTTP endpoint)
- `LANGFUSE_ENABLED` (default `false`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `LANGFUSE_SAMPLE_RATE` (`0..1`)

Install Langfuse only where it is enabled:

```sh
uv sync --extra langfuse
```

HTTPX and asyncpg are automatically instrumented. FastAPI extracts/injects W3C
`traceparent`; manual spans cover graph nodes, retrieval, MCP tools, and ingestion.
`GET /metrics` exposes API, stream/TTFT, graph, retrieval, auction-search, tool/source,
ingestion, DB-pool, token/cost, and feedback series. Label values are allowlisted in
`api/metrics.py`; unknown values collapse to `other`.

Import `docs/grafana/get-auction-list-overview.json` into Grafana and select the
Prometheus data source. It includes API, graph, retrieval, tool, ingestion, source-health,
database-pool, token, cost, and feedback panels.

## Evaluation commands

The committed v1 JSONL dataset is sanitized and versioned. The deterministic gate uses
the real graph and mocked service boundaries; it has no network, model, Langfuse, or Ragas
dependency:

```sh
uv run auction-eval --threshold 1.0
uv run pytest -m "not live_contract and not scheduled_eval"
```

Separate suites:

```sh
uv run pytest -m integration_mock
uv run pytest -m live_contract                 # explicit approved-source access
uv sync --extra evaluation --extra langfuse
uv run pytest -m scheduled_eval                # scheduled/vendor-backed quality job
```

The deterministic report covers router, filter extraction, structured search, retrieval,
grounding, relevance, citations, no-answer, tool selection, arguments, success, and
disclaimer metrics. `evaluation/adapters.py` provides lazy Ragas rows and Langfuse score
publishing without adding live or costly dependencies to CI.
