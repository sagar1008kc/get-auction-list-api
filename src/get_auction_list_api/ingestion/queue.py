"""Durable PostgreSQL queue with owner-checked leases and bounded retries."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from get_auction_list_api.database import QueryExecutor

_ENQUEUE = """
insert into public.ingestion_jobs
  (requested_by, document_id, idempotency_key, source_uri, input)
values ($1, $2, $3, $4, $5::jsonb)
on conflict (
  coalesce(requested_by, '00000000-0000-0000-0000-000000000000'::uuid),
  idempotency_key
) do nothing
returning *
"""

_CLAIM = """
with candidate as (
  select id
  from public.ingestion_jobs
  where state in ('queued', 'retry') and available_at <= now()
  order by available_at, created_at, id
  for update skip locked
  limit $1
)
update public.ingestion_jobs j
set state = 'running',
    attempt_count = attempt_count + 1,
    started_at = coalesce(started_at, now()),
    heartbeat_at = now(),
    output = j.output || jsonb_build_object('lease_token', $2::text),
    updated_at = now()
from candidate
where j.id = candidate.id
returning j.*
"""

_HEARTBEAT = """
update public.ingestion_jobs
set heartbeat_at = now(), updated_at = now()
where id = $1 and state = 'running' and output->>'lease_token' = $2
returning id
"""

_SUCCEED = """
update public.ingestion_jobs
set state = 'succeeded', completed_at = now(), heartbeat_at = null,
    output = $3::jsonb, updated_at = now()
where id = $1 and state = 'running' and output->>'lease_token' = $2
returning *
"""

_FAIL = """
update public.ingestion_jobs
set state = case when attempt_count >= $3 then 'dead_letter' else 'retry' end,
    available_at = case when attempt_count >= $3 then available_at
                        else now() + ($4::interval * (0.75 + random() * 0.5)) end,
    completed_at = case when attempt_count >= $3 then now() else null end,
    heartbeat_at = null, error_code = $5, error_detail_redacted = $6,
    output = output - 'lease_token', updated_at = now()
where id = $1 and state = 'running' and output->>'lease_token' = $2
returning *
"""

_RECOVER = """
update public.ingestion_jobs
set state = case when attempt_count >= $2 then 'dead_letter' else 'retry' end,
    available_at = case when attempt_count >= $2 then available_at
                        else now() + ($3::interval * (0.75 + random() * 0.5)) end,
    completed_at = case when attempt_count >= $2 then now() else null end,
    heartbeat_at = null, error_code = 'stale_lease',
    error_detail_redacted = 'Worker lease expired.',
    output = output - 'lease_token', updated_at = now()
where state = 'running' and heartbeat_at < now() - $1::interval
returning *
"""


class IngestionJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: UUID
    requested_by: UUID | None = None
    document_id: UUID | None = None
    idempotency_key: str
    source_uri: str
    state: str
    attempt_count: int = Field(ge=0)
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


def _interval(value: timedelta) -> str:
    seconds = value.total_seconds()
    if seconds <= 0:
        raise ValueError("Interval must be positive.")
    return f"{seconds:f} seconds"


class PostgresJobQueue:
    def __init__(self, executor: QueryExecutor) -> None:
        self._executor = executor

    async def enqueue(
        self,
        *,
        requested_by: UUID | None,
        document_id: UUID | None,
        idempotency_key: str,
        source_uri: str,
        payload: Mapping[str, object],
    ) -> IngestionJob | None:
        row = await self._executor.fetchrow(
            _ENQUEUE,
            requested_by,
            document_id,
            idempotency_key,
            source_uri,
            dict(payload),
        )
        return IngestionJob.model_validate(dict(row)) if row else None

    async def claim(self, *, lease_token: str, limit: int = 1) -> Sequence[IngestionJob]:
        if not lease_token or not 1 <= limit <= 20:
            raise ValueError("A lease token and a limit from 1 to 20 are required.")
        rows = await self._executor.fetch(_CLAIM, limit, lease_token)
        return [IngestionJob.model_validate(dict(row)) for row in rows]

    async def heartbeat(self, job_id: UUID, *, lease_token: str) -> bool:
        return await self._executor.fetchrow(_HEARTBEAT, job_id, lease_token) is not None

    async def succeed(
        self,
        job_id: UUID,
        *,
        lease_token: str,
        output: Mapping[str, object],
    ) -> IngestionJob | None:
        row = await self._executor.fetchrow(_SUCCEED, job_id, lease_token, dict(output))
        return IngestionJob.model_validate(dict(row)) if row else None

    async def fail(
        self,
        job_id: UUID,
        *,
        lease_token: str,
        max_attempts: int,
        retry_delay: timedelta,
        error_code: str,
        redacted_detail: str,
    ) -> IngestionJob | None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        row = await self._executor.fetchrow(
            _FAIL,
            job_id,
            lease_token,
            max_attempts,
            _interval(retry_delay),
            error_code[:80],
            redacted_detail[:500],
        )
        return IngestionJob.model_validate(dict(row)) if row else None

    async def recover_stale(
        self,
        *,
        stale_after: timedelta,
        max_attempts: int,
        retry_delay: timedelta,
    ) -> Sequence[IngestionJob]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        rows = await self._executor.fetch(
            _RECOVER,
            _interval(stale_after),
            max_attempts,
            _interval(retry_delay),
        )
        return [IngestionJob.model_validate(dict(row)) for row in rows]
