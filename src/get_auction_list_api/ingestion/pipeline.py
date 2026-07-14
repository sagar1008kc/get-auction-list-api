"""Checksum-idempotent ingestion orchestration with atomic publication."""

import time
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from get_auction_list_api.api.metrics import INGESTION_DURATION, INGESTION_JOBS
from get_auction_list_api.ingestion.chunking import DocumentChunk, chunk_units
from get_auction_list_api.ingestion.validation import ValidatedFile, validate_file
from get_auction_list_api.llm.embeddings import EmbeddingProvider
from get_auction_list_api.observability.telemetry import tracer
from get_auction_list_api.parsers.models import ParseResult


class Parser(Protocol):
    name: str
    version: str

    def parse(self, content: bytes) -> ParseResult: ...


class PublishedVersion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: UUID
    document_version_id: UUID
    created: bool
    chunk_count: int
    rejected_count: int


class TransactionalPublisher(Protocol):
    async def find_version(self, document_id: UUID, sha256: str) -> PublishedVersion | None: ...

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
        """Atomically insert immutable version/chunks and repoint current_version_id."""
        ...


class IngestionPipeline:
    def __init__(
        self,
        *,
        publisher: TransactionalPublisher,
        embedding_provider: EmbeddingProvider,
        expected_embedding_dimensions: int = 1536,
    ) -> None:
        self._publisher = publisher
        self._embeddings = embedding_provider
        self._dimensions = expected_embedding_dimensions

    async def run(
        self,
        *,
        document_id: UUID,
        content: bytes,
        declared_media_type: str | None,
        parser: Parser,
    ) -> PublishedVersion:
        started = time.perf_counter()
        outcome = "success"
        try:
            with tracer().start_as_current_span("ingestion.pipeline") as span:
                span.set_attribute("ingestion.parser", parser.name)
                validated = validate_file(content, declared_media_type=declared_media_type)
                existing = await self._publisher.find_version(document_id, validated.sha256)
                if existing is not None:
                    outcome = "unchanged"
                    return existing
                parsed = parser.parse(validated.content)
                chunks = chunk_units(parsed.units)
                if not chunks:
                    raise ValueError("Parser produced no publishable evidence.")
                batch = await self._embeddings.embed([chunk.content for chunk in chunks])
                if batch.dimensions != self._dimensions:
                    raise ValueError("Embedding dimensions do not match the database contract.")
                return await self._publisher.publish(
                    document_id=document_id,
                    validated=validated,
                    parser_name=parser.name,
                    parser_version=parser.version,
                    chunks=chunks,
                    embeddings=batch.vectors,
                    embedding_model=batch.model,
                    rejected=len(parsed.errors),
                )
        except Exception:
            outcome = "error"
            raise
        finally:
            INGESTION_JOBS.labels("pipeline", outcome).inc()
            INGESTION_DURATION.labels(outcome).observe(time.perf_counter() - started)
