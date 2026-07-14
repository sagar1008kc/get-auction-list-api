"""Production adapter backing ops routes with the existing durable queue."""

import json
from typing import cast
from uuid import UUID, uuid4

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.database import QueryExecutor
from get_auction_list_api.domain import stable_idempotency_key
from get_auction_list_api.ingestion.queue import PostgresJobQueue

_STATUS = """
select id, state, attempt_count, source_uri, document_id, output,
       error_code, error_detail_redacted, created_at, started_at, completed_at
from public.ingestion_jobs
where id = $1
"""


class PostgresIngestionOperations:
    def __init__(self, executor: QueryExecutor) -> None:
        self._executor = executor
        self._queue = PostgresJobQueue(executor)

    async def enqueue(
        self,
        principal: Principal,
        payload: dict[str, object],
    ) -> dict[str, object]:
        principal.require(Permission.INGESTION_WRITE)
        source_type = str(payload["source_type"])
        if source_type == "supabase_storage":
            source_uri = f"supabase://{payload['storage_bucket']}/{payload['storage_path']}"
        else:
            source_uri = f"registered://{payload['source_id']}"
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        key = stable_idempotency_key(
            operation="ingestion_job",
            principal_scope=str(principal.user_id),
            canonical_payload=canonical,
        )
        job = await self._queue.enqueue(
            requested_by=principal.user_id,
            document_id=None,
            idempotency_key=key,
            source_uri=source_uri,
            payload=payload,
        )
        if job is None:
            row = await self._executor.fetchrow(
                "select id, state from public.ingestion_jobs "
                "where requested_by = $1 and idempotency_key = $2",
                principal.user_id,
                key,
            )
            if row is None:
                raise RuntimeError("Idempotent ingestion job was not found.")
            return {"job_id": str(row["id"]), "status": str(row["state"])}
        return {"job_id": str(job.id), "status": job.state}

    async def status(
        self,
        principal: Principal,
        job_id: UUID,
    ) -> dict[str, object] | None:
        principal.require(Permission.INGESTION_WRITE)
        row = await self._executor.fetchrow(_STATUS, job_id)
        if row is None:
            return None
        allowed = {
            "id",
            "state",
            "attempt_count",
            "source_uri",
            "document_id",
            "output",
            "error_code",
            "error_detail_redacted",
            "created_at",
            "started_at",
            "completed_at",
        }
        return {key: value for key, value in row.items() if key in allowed}

    async def sync(
        self,
        principal: Principal,
        payload: dict[str, object],
    ) -> dict[str, object]:
        principal.require(Permission.INGESTION_WRITE)
        run_id = uuid4()
        for source_id in cast(list[object], payload["source_ids"]):
            child = {
                "source_type": "registered_url",
                "source_id": str(source_id),
                "document_type": "registered_source_sync",
                "sync_mode": payload["mode"],
                "source_sync_run_id": str(run_id),
            }
            await self.enqueue(principal, child)
        return {"source_sync_run_id": str(run_id), "status": "queued"}
