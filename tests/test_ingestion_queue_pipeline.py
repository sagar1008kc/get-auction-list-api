from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from get_auction_list_api.database import DatabaseRow
from get_auction_list_api.ingestion.chunking import DocumentChunk
from get_auction_list_api.ingestion.pipeline import IngestionPipeline, PublishedVersion
from get_auction_list_api.ingestion.queue import PostgresJobQueue
from get_auction_list_api.ingestion.validation import ValidatedFile
from get_auction_list_api.llm.embeddings import EmbeddingBatch
from get_auction_list_api.parsers.html import HtmlParser


class Executor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_rows: Sequence[DatabaseRow] = ()
        self.fetchrow_value: DatabaseRow | None = None

    async def fetch(self, query: str, *args: object) -> Sequence[DatabaseRow]:
        self.calls.append((query, args))
        return self.fetch_rows

    async def fetchrow(self, query: str, *args: object) -> DatabaseRow | None:
        self.calls.append((query, args))
        return self.fetchrow_value

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        return "OK"


def _job(job_id: UUID, *, state: str = "running") -> dict[str, Any]:
    return {
        "id": job_id,
        "requested_by": None,
        "document_id": None,
        "idempotency_key": "same-request",
        "source_uri": "https://getauctionlist.com/privacy",
        "state": state,
        "attempt_count": 1,
        "input": {},
        "output": {"lease_token": "worker-1"},
    }


@pytest.mark.asyncio
async def test_queue_claim_uses_skip_locked_and_parameterized_lease() -> None:
    executor = Executor()
    executor.fetch_rows = (_job(uuid4()),)
    lease = str(uuid4())
    jobs = await PostgresJobQueue(executor).claim(lease_token=lease, limit=2)
    query, args = executor.calls[0]
    assert "for update skip locked" in query.lower()
    assert lease not in query
    assert args == (2, lease)
    assert jobs[0].attempt_count == 1


@pytest.mark.asyncio
async def test_queue_heartbeat_and_failure_are_owner_checked() -> None:
    executor = Executor()
    executor.fetchrow_value = _job(uuid4())
    queue = PostgresJobQueue(executor)
    job_id = UUID(str(executor.fetchrow_value["id"]))
    lease = str(uuid4())
    assert await queue.heartbeat(job_id, lease_token=lease)
    failed = await queue.fail(
        job_id,
        lease_token=lease,
        max_attempts=3,
        retry_delay=timedelta(seconds=5),
        error_code="transient",
        redacted_detail="safe detail",
    )
    assert failed is not None
    assert all(lease not in query for query, _ in executor.calls)
    assert any("dead_letter" in query and "retry" in query for query, _ in executor.calls)


@pytest.mark.asyncio
async def test_stale_recovery_is_bounded() -> None:
    executor = Executor()
    executor.fetch_rows = (_job(uuid4(), state="retry"),)
    rows = await PostgresJobQueue(executor).recover_stale(
        stale_after=timedelta(minutes=2),
        max_attempts=5,
        retry_delay=timedelta(seconds=10),
    )
    assert rows[0].state == "retry"
    assert "heartbeat_at <" in executor.calls[0][0]


class Embeddings:
    calls = 0

    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls += 1
        return EmbeddingBatch(
            model="test-embedding",
            dimensions=1536,
            vectors=tuple((0.0,) * 1536 for _ in texts),
        )


class Publisher:
    def __init__(self, existing: PublishedVersion | None = None) -> None:
        self.existing = existing
        self.published: Mapping[str, object] | None = None

    async def find_version(self, document_id: UUID, sha256: str) -> PublishedVersion | None:
        return self.existing

    async def publish(
        self,
        *,
        document_id: UUID,
        validated: ValidatedFile,
        parser_name: str,
        parser_version: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
        rejected: int,
    ) -> PublishedVersion:
        self.published = {
            "document_id": document_id,
            "validated": validated,
            "parser_name": parser_name,
            "parser_version": parser_version,
            "chunks": chunks,
            "embeddings": embeddings,
            "embedding_model": embedding_model,
            "rejected": rejected,
        }
        return PublishedVersion(
            document_id=document_id,
            document_version_id=uuid4(),
            created=True,
            chunk_count=len(chunks),
            rejected_count=rejected,
        )


@pytest.mark.asyncio
async def test_pipeline_publishes_one_checksum_generation() -> None:
    publisher = Publisher()
    embeddings = Embeddings()
    result = await IngestionPipeline(
        publisher=publisher,
        embedding_provider=embeddings,
    ).run(
        document_id=uuid4(),
        content=b"<html><body><h1>Privacy</h1><p>We protect data.</p></body></html>",
        declared_media_type="text/html",
        parser=HtmlParser(),
    )
    assert result.created
    assert publisher.published is not None
    assert embeddings.calls == 1


@pytest.mark.asyncio
async def test_pipeline_short_circuits_identical_checksum() -> None:
    existing = PublishedVersion(
        document_id=uuid4(),
        document_version_id=uuid4(),
        created=False,
        chunk_count=1,
        rejected_count=0,
    )
    embeddings = Embeddings()
    result = await IngestionPipeline(
        publisher=Publisher(existing),
        embedding_provider=embeddings,
    ).run(
        document_id=existing.document_id,
        content=b"<html><body><p>Privacy policy.</p></body></html>",
        declared_media_type="text/html",
        parser=HtmlParser(),
    )
    assert result == existing
    assert embeddings.calls == 0
