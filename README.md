# GetAuctionList API

Python 3.12/FastAPI foundations for the GetAuctionList AI Front Door. The controlling
architecture and API contracts live in the sibling `auction-lens-ai/docs` directory;
`docs/README.md` links to them.

## Bootstrap

Install the locked development environment:

```sh
uv sync --frozen --dev
```

Run the API:

```sh
uv run uvicorn get_auction_list_api.main:app --reload
```

Verify the project:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run pre-commit run --all-files
```

The process exposes `GET /health/live`, `GET /health/ready`, and internal `GET /metrics`.
Liveness never calls external services. Readiness executes only checks registered for
enabled integrations.

## Configuration

Settings are read from process environment with the `GET_AUCTION_LIST_` prefix.

### Required for authenticated local / staging chat

These three values must point at the **same Supabase project** as `auction-lens-ai` so
BFF-forwarded user access tokens validate against this API's JWKS:

- `GET_AUCTION_LIST_SUPABASE_URL` — HTTPS project URL; derives JWKS
  `{url}/auth/v1/.well-known/jwks.json` and issuer `{url}/auth/v1` unless overridden
- `GET_AUCTION_LIST_DATABASE_URL` — PostgreSQL pool URL that can call RPCs gated to
  `service_role` / `is_ops_jwt()` (migration
  `20260714040000_ai_backend_foundations.sql`)
- `GET_AUCTION_LIST_OPENAI_API_KEY` — classifier, synthesizer, and embeddings

Tokens without `app_metadata.role` / `roles` default to role `user` with
`auction:read`, `document:read`, and `tool:execute`. Ops/admin roles keep write/audit
boundaries.

Optional JWT overrides: `GET_AUCTION_LIST_JWT_ISSUER`, `GET_AUCTION_LIST_JWT_AUDIENCE`
(default `authenticated`), `GET_AUCTION_LIST_JWKS_CACHE_TTL_SECONDS`.

### Other supported names

- `GET_AUCTION_LIST_SERVICE_NAME`
- `GET_AUCTION_LIST_SERVICE_VERSION`
- `GET_AUCTION_LIST_ENVIRONMENT`
- `GET_AUCTION_LIST_LOG_LEVEL`
- `GET_AUCTION_LIST_LOG_FORMAT`
- `GET_AUCTION_LIST_SUPABASE_SERVICE_ROLE_KEY`
- `GET_AUCTION_LIST_INTERNAL_MCP_TOKEN`
- `GET_AUCTION_LIST_OPENAI_BASE_URL`
- `GET_AUCTION_LIST_OPENAI_CHAT_MODEL`
- `GET_AUCTION_LIST_OPENAI_EMBEDDING_MODEL`
- `GET_AUCTION_LIST_OPENAI_EMBEDDING_DIMENSIONS`
- `GET_AUCTION_LIST_CHECKPOINT_ENABLED`
- `GET_AUCTION_LIST_CHECKPOINT_SETUP_ON_START`

Provide sensitive values through the deployment platform's secret manager. The service
does not implicitly read dotenv files, and this repository intentionally has no
`.env.example`.

See `docs/BOOTSTRAP.md` for scope and integration guidance, and `docs/DEPLOYMENT.md`
for data readiness (auction row ingest vs document chunks).

When checkpointing is enabled, the database migration must include the isolated
`langgraph` schema. The service uses a dedicated PostgreSQL pool, runs the official
checkpointer setup on startup by default, and enforces user ownership for every thread.

## Observability and evaluation

Prometheus metrics and W3C OpenTelemetry propagation are built in. OTLP and Langfuse
export are optional and failure-isolated; install the latter with
`uv sync --extra langfuse`. Run the deterministic, network-free quality gate with:

```sh
uv run auction-eval --threshold 1.0
```

See `docs/OBSERVABILITY_EVALUATION.md` for configuration, privacy controls, test markers,
optional Ragas/Langfuse adapters, and the importable Grafana dashboard.

## Ingestion, search, and retrieval

The framework-independent packages under `ingestion`, `parsers`, `normalization`,
`tools`, `rag`, and `llm` implement the approved source registry and worker boundaries.
The built-in worker handler is
`get_auction_list_api.ingestion.handler:handle_ingestion_job`. It downloads
`supabase://auction_files/...` objects with the service role (never signed browser
URLs), publishes spreadsheet rows into `auction_records`, and ingests approved
policy HTML into `document_chunks` with embeddings.

Network fetching for parsers themselves stays injectable so normal tests never contact
live policy or Williamson County sources.

The PostgreSQL queue uses atomic `FOR UPDATE SKIP LOCKED` claims, owner-checked
heartbeats/completion, stale lease recovery, bounded retry, and dead-letter transitions.
Workers must generate an unpredictable lease token for each claim cycle and must only
publish through the transactional publisher boundary.

Structured auction filters remain parameterized PostgreSQL calls; exact auction filters
never use vector retrieval. County schedule/calendar questions route to MCP public-record
tools instead of the auction SQL index. Policy retrieval independently queries FTS and
pgvector, applies reciprocal-rank fusion and budgets, removes duplicate evidence, marks
all source content untrusted, and constructs citations before any synthesis caller
receives context. Answers without evidence do not invent auction rows, schedules, or
policy text; CTA is emitted only when indexed `auction_results` are non-empty.

## Chat orchestration and public records

Authenticated clients use `POST /v1/chat` or `POST /v1/chat/stream`. Both execute the
same bounded, typed LangGraph workflow. SSE emits only public lifecycle events and
display text; prompts, hidden reasoning, and raw tool responses are never streamed.
Auction and public-record answers always state that official records control.

The internal `/mcp` Streamable HTTP mount uses the official Python MCP SDK/FastMCP and
exposes exactly six typed, read-only tools:

- `county.discover_trustee_sale_sources`
- `county.search_foreclosure_records`
- `county.get_foreclosure_notice`
- `wcad.search_property`
- `wcad.get_property_details`
- `property.correlate_records`

The direct adapters accept no arbitrary destination URL. HTTPS destinations and every
redirect are revalidated against the configured host allowlist and public IP space.
Requests have bounded timeouts, retries, redirects, response sizes, and a short TTL
cache. Tool results include parser version, retrieval time, trace ID, and audit ID.
