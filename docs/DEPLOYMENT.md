# Deployment and operations

## Runtime units

Build one OCI image and override its command for each independently scaled unit. The
image has no platform-specific entrypoint and runs as UID/GID 10001.

```sh
docker build -t get-auction-list-api:local .
docker run --rm -p 8000:8000 get-auction-list-api:local
```

- API: `uv run auction-api` (the image default is `auction-api`).
- Ingestion worker: `uv run auction-ingestion-worker`.
- Optional local MCP server: `uv run auction-mcp` (stdio only).

The API launcher runs one Uvicorn process, honors SIGTERM, stops accepting work, and
allows the configured graceful-shutdown interval for active requests. Scale replicas in
the container platform instead of creating child workers in the container. The API also
mounts Streamable HTTP MCP at `/mcp`; production requires its bearer token. Prefer the
stdio command when MCP is consumed on the same host.

The ingestion command requires both a database URL and
`GET_AUCTION_LIST_INGESTION_HANDLER`. The latter names a trusted async callable as
`package.module:async_callable`. It receives an `IngestionJob` and returns a JSON-safe
mapping. The command validates this contract before connecting or claiming work. This
repository intentionally does not pretend that the parser/publisher abstractions are a
complete production source handler.

## Deployment configuration

Set values in the platform's configuration and secret stores. Names supported by the
service are:

- Identity/logging: `GET_AUCTION_LIST_SERVICE_NAME`,
  `GET_AUCTION_LIST_SERVICE_VERSION`, `GET_AUCTION_LIST_ENVIRONMENT`,
  `GET_AUCTION_LIST_LOG_LEVEL`, `GET_AUCTION_LIST_LOG_FORMAT`.
- Listener: `GET_AUCTION_LIST_BIND_HOST`, `GET_AUCTION_LIST_PORT`,
  `GET_AUCTION_LIST_GRACEFUL_SHUTDOWN_SECONDS`,
  `GET_AUCTION_LIST_KEEP_ALIVE_SECONDS`.
- Supabase/auth: `GET_AUCTION_LIST_SUPABASE_URL` (HTTPS; **must be the same Supabase
  project as auction-lens-ai** so JWKS validates BFF-forwarded access tokens),
  `GET_AUCTION_LIST_SUPABASE_SERVICE_ROLE_KEY`, `GET_AUCTION_LIST_JWT_ISSUER`,
  `GET_AUCTION_LIST_JWT_AUDIENCE`, `GET_AUCTION_LIST_JWKS_CACHE_TTL_SECONDS`.
- Database: `GET_AUCTION_LIST_DATABASE_URL` (pool identity must satisfy RPC gates
  `auth.role() = 'service_role'` or `is_ops_jwt()` from migration
  `20260714040000_ai_backend_foundations.sql`),
  `GET_AUCTION_LIST_DATABASE_POOL_MIN_SIZE`,
  `GET_AUCTION_LIST_DATABASE_POOL_MAX_SIZE`,
  `GET_AUCTION_LIST_DATABASE_COMMAND_TIMEOUT_SECONDS`.
- OpenAI: `GET_AUCTION_LIST_OPENAI_API_KEY` (required for live chat classify/synthesize/
  embed), `GET_AUCTION_LIST_OPENAI_BASE_URL`, `GET_AUCTION_LIST_OPENAI_CHAT_MODEL`,
  `GET_AUCTION_LIST_OPENAI_EMBEDDING_MODEL`,
  `GET_AUCTION_LIST_OPENAI_EMBEDDING_DIMENSIONS`,
  `GET_AUCTION_LIST_OPENAI_TIMEOUT_SECONDS`, `GET_AUCTION_LIST_OPENAI_MAX_RETRIES`.
- Checkpoint: `GET_AUCTION_LIST_CHECKPOINT_ENABLED`,
  `GET_AUCTION_LIST_CHECKPOINT_SETUP_ON_START`.
- HTTP security: `GET_AUCTION_LIST_CORS_ORIGINS`,
  `GET_AUCTION_LIST_TRUSTED_HOSTS`, `GET_AUCTION_LIST_MAX_REQUEST_BODY_BYTES`,
  `GET_AUCTION_LIST_RATE_LIMIT_REQUESTS`,
  `GET_AUCTION_LIST_RATE_LIMIT_WINDOW_SECONDS`,
  `GET_AUCTION_LIST_CONCURRENCY_LIMIT`, `GET_AUCTION_LIST_REQUEST_TIMEOUT_SECONDS`.
- Public records: `GET_AUCTION_LIST_APPROVED_SOURCE_HOSTS`,
  `GET_AUCTION_LIST_PUBLIC_HTTP_TIMEOUT_SECONDS`,
  `GET_AUCTION_LIST_PUBLIC_HTTP_MAX_ATTEMPTS`,
  `GET_AUCTION_LIST_PUBLIC_HTTP_MAX_RESPONSE_BYTES`,
  `GET_AUCTION_LIST_PUBLIC_HTTP_CACHE_TTL_SECONDS`,
  `GET_AUCTION_LIST_INTERNAL_MCP_TOKEN`.
- Worker: `GET_AUCTION_LIST_INGESTION_HANDLER`,
  `GET_AUCTION_LIST_INGESTION_POLL_SECONDS`,
  `GET_AUCTION_LIST_INGESTION_HEARTBEAT_SECONDS`,
  `GET_AUCTION_LIST_INGESTION_STALE_SECONDS`,
  `GET_AUCTION_LIST_INGESTION_MAX_ATTEMPTS`,
  `GET_AUCTION_LIST_INGESTION_RETRY_SECONDS`.
- Telemetry: `GET_AUCTION_LIST_OTEL_ENABLED`,
  `GET_AUCTION_LIST_OTEL_EXPORTER_OTLP_ENDPOINT`,
  `GET_AUCTION_LIST_LANGFUSE_ENABLED`, `GET_AUCTION_LIST_LANGFUSE_PUBLIC_KEY`,
  `GET_AUCTION_LIST_LANGFUSE_SECRET_KEY`, `GET_AUCTION_LIST_LANGFUSE_HOST`,
  `GET_AUCTION_LIST_LANGFUSE_SAMPLE_RATE`.

Default JWT role when `app_metadata` has no roles is `user`
(`auction:read` + `document:read` + `tool:execute`). Ops/admin retain
`ingestion:write` / `audit:read`. Browser clients never call MCP or OpenAI directly;
auction-lens-ai proxies `/api/ai/{chat,chat/stream,feedback}` to `/v1/*` with
`Authorization: Bearer <supabase access token>`, `X-Request-ID`, and
`X-Correlation-ID`.

Tuple settings use the serialization accepted by Pydantic Settings. Never put secrets
in image layers, command arguments, logs, health responses, or repository files.

## Health and rollout

- Liveness: `GET /health/live`; use for process restart decisions.
- Readiness: `GET /health/ready`; use for traffic admission.
- Metrics: `GET /metrics`; expose only to the internal scraper network.

Roll out the API and worker independently. Start with no public traffic, verify readiness
and metrics, run authenticated JSON and SSE probes, then canary traffic. Stop worker
claims before database maintenance. The worker finishes its current handler on SIGTERM;
the lease becomes recoverable if the platform exceeds the termination grace period.

## Database migration

The additive migration is
`auction-lens-ai/supabase/migrations/20260714040000_ai_backend_foundations.sql`.
It depends on earlier sibling migrations plus Supabase Auth schemas and is not a
standalone PostgreSQL bootstrap script.

From `auction-lens-ai`, review and apply with the Supabase CLI version approved by the
deployment environment:

```sh
supabase db push --linked --dry-run
supabase db push --linked
```

For a disposable local Supabase stack, run `supabase start` followed by
`supabase db reset`. For an existing staging project, restore a production-like backup,
apply all pending migrations, and run RLS/role smoke tests before production.

There is intentionally no destructive down migration. Forward rollback is:

1. Disable API/BFF traffic and pause ingestion consumers.
2. Roll the API and worker images back independently.
3. Add a reviewed forward migration to restore changed grants, policies, views, or
   functions.
4. Retain additive tables, immutable document versions, audit data, and lineage until
   retention policy permits removal.

Do not drop columns, evidence, or indexes during an incident. Take and verify a database
backup before applying the migration and record its identifier in the change ticket.

The local `compose.yaml` supplies pgvector PostgreSQL for adapter development only. It
binds to loopback and requires an explicitly supplied password. It cannot validate the
full Supabase migration because plain PostgreSQL lacks the Auth roles and schemas:

```sh
POSTGRES_PASSWORD="$(openssl rand -hex 24)" docker compose up -d postgres
```

## Incident runbooks

- API not ready: inspect dependency component status and database pool metrics; remove
  the replica from traffic. Do not convert readiness failures into liveness failures.
- Source outage: leave unrelated tools enabled, open/disable only the affected source,
  preserve typed partial results, and do not bypass host or public-IP validation.
- Queue backlog: check oldest available job, retry/dead-letter counts, worker heartbeat,
  and database saturation. Scale consumers only within database connection headroom.
- Stale workers: stop the old replica, wait beyond the stale lease interval, and let one
  healthy worker recover leases. Never edit lease tokens manually.
- Compromised credential: revoke and rotate it in the platform secret manager, restart
  affected units, and inspect redacted audit/trace IDs. Never print the old value.
- Bad release: stop canary traffic, pause workers, deploy the prior immutable image, and
  use a reviewed forward database migration if database behavior must change.

## Known source and runtime limitations

- Public-record automation remains subject to owner approval, terms, robots policy,
  rate limits, HTML/PDF drift, CAPTCHA behavior, and source availability.
- The allowlist includes only configured HTTPS hosts; tool callers cannot supply an
  arbitrary destination. DNS and every redirect are revalidated.
- XLSX and PDF are supported parsing boundaries; legacy binary XLS is not implemented.
- The worker loads
  `get_auction_list_api.ingestion.handler:handle_ingestion_job` (or another trusted
  callable) for Storage downloads and approved policy URL publication.
- The default graph is deterministic scaffolding, not a configured model-provider
  deployment. Live-provider, representative-load, restore, failover, and source-policy
  approval remain release gates.

## Data readiness for E2E trustee search

After applying `20260714040000_ai_backend_foundations.sql`:

- `search_auction_records` and `hybrid_retrieve_document_chunks` are callable by the API
  pool when the DB role is `service_role` (or JWT context passes `is_ops_jwt()`).
- Trustee / mortgagor / address auction search reads **`auction_records`** (and related
  mortgagor tables), not document chunks. Chunk ingestion alone does not populate
  trustee results.
- Populate `auction_records` from the approved Storage spreadsheet using the built-in
  worker handler:

  ```sh
  export GET_AUCTION_LIST_INGESTION_HANDLER=get_auction_list_api.ingestion.handler:handle_ingestion_job
  uv run auction-ingestion-worker
  ```

  Enqueue (ops JWT) the July 2026 index:

  ```json
  {
    "source_type": "supabase_storage",
    "storage_bucket": "auction_files",
    "storage_path": "williamson_county/getAuctionList_July_2026.xlsx",
    "document_type": "auction_spreadsheet"
  }
  ```

  The handler downloads with the service role (never signed browser URLs), writes
  `documents` / `document_versions`, and upserts normalized `auction_records` /
  `auction_record_mortgagors` with `report_year=2026`, `report_month=7`, `stable_key`,
  and sheet/row lineage. Verify with `search_auction_records` filtered to that period.
- Sync approved policy HTML into RAG chunks:

  ```json
  {
    "source_ids": ["getauctionlist-privacy", "getauctionlist-disclaimer"],
    "mode": "incremental"
  }
  ```

  via `POST /v1/ingestion/sources/sync`, or enqueue `registered_url` jobs with those
  `source_id` registry keys.
- An empty index must return the standard no-match answer with `cta=null` (never 500).
  Emit `FinalResponse.cta` only when indexed search returns non-empty
  `auction_results`, copying filters from the matched entities.
- County trustee-sale **schedule/calendar** questions route to MCP public-record tools
  (`county.discover_trustee_sale_sources`), not SQL auction search. Missing schedules
  or MCP timeouts surface as `unavailable_sources` without inventing dates.
