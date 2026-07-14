"""Transactional publishers for document chunks and normalized auction rows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from get_auction_list_api.database import AsyncDatabase, QueryExecutor
from get_auction_list_api.ingestion.auction_rows import PublishableAuctionRow
from get_auction_list_api.ingestion.chunking import DocumentChunk
from get_auction_list_api.ingestion.pipeline import PublishedVersion, TransactionalPublisher
from get_auction_list_api.ingestion.validation import ValidatedFile

_FIND_VERSION = """
select dv.id as document_version_id, dv.document_id,
       (select count(*)::int from public.document_chunks c
        where c.document_version_id = dv.id) as chunk_count
from public.document_versions dv
where dv.document_id = $1 and dv.sha256 = $2
"""

_ENSURE_DOCUMENT = """
insert into public.documents (
  id, source_type, source_uri, canonical_source_key, title, status
) values ($1, $2, $3, $4, $5, 'active')
on conflict (canonical_source_key) do update
  set source_uri = excluded.source_uri,
      title = coalesce(excluded.title, public.documents.title),
      updated_at = now()
returning id
"""

_INSERT_VERSION = """
insert into public.document_versions (
  id, document_id, sha256, media_type, byte_size, storage_path,
  parser_version, embedding_model, embedding_dimensions, metadata
) values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
on conflict (document_id, sha256) do update
  set media_type = excluded.media_type
returning id, (xmax = 0) as inserted
"""

_SET_CURRENT = """
update public.documents
set current_version_id = $2, updated_at = now()
where id = $1
"""

_INSERT_CHUNK = """
insert into public.document_chunks (
  document_version_id, chunk_index, content, source_coordinates,
  content_sha256, embedding, embedding_model, token_count
) values (
  $1, $2, $3, $4::jsonb, $5, $6::extensions.vector, $7, $8
)
on conflict (document_version_id, chunk_index) do nothing
"""

_UPSERT_AUCTION = """
insert into public.auction_records (
  county, auction_type, sale_date, property_address, city, state, zip_code,
  trustee_name, opening_bid, source_url, source_name, raw_metadata,
  rid, stable_key, document_version_id, source_coordinates,
  normalized_trustee_name, normalized_address, report_year, report_month,
  loan_type, equity, margin, normalization_version, ingested_at, status
) values (
  $1, 'trustee_sale', $2, $3, $4, $5, $6,
  $7, $8, $9, $10, $11::jsonb,
  $12, $13, $14, $15::jsonb,
  $16, $17, $18, $19,
  $20, $21, $22, $23, $24, 'new'
)
on conflict (stable_key) where (stable_key is not null) do update set
  sale_date = excluded.sale_date,
  property_address = excluded.property_address,
  city = excluded.city,
  zip_code = excluded.zip_code,
  trustee_name = excluded.trustee_name,
  opening_bid = excluded.opening_bid,
  document_version_id = excluded.document_version_id,
  source_coordinates = excluded.source_coordinates,
  normalized_trustee_name = excluded.normalized_trustee_name,
  normalized_address = excluded.normalized_address,
  report_year = excluded.report_year,
  report_month = excluded.report_month,
  loan_type = excluded.loan_type,
  equity = excluded.equity,
  margin = excluded.margin,
  normalization_version = excluded.normalization_version,
  ingested_at = excluded.ingested_at,
  updated_at = now()
returning id
"""

_DELETE_MORTGAGORS = """
delete from public.auction_record_mortgagors where auction_record_id = $1
"""

_INSERT_MORTGAGOR = """
insert into public.auction_record_mortgagors (
  auction_record_id, ordinal, display_name, normalized_full_name,
  normalized_first_name, normalized_last_name, source_coordinates
) values ($1, $2, $3, $4, $5, $6, $7::jsonb)
"""


def _coords_json(coordinates: object) -> str:
    if hasattr(coordinates, "model_dump"):
        return json.dumps(coordinates.model_dump(mode="json"), separators=(",", ":"))
    return json.dumps(coordinates if isinstance(coordinates, dict) else {}, separators=(",", ":"))


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


class PostgresDocumentPublisher(TransactionalPublisher):
    def __init__(self, database: AsyncDatabase) -> None:
        self._database = database

    async def ensure_document(
        self,
        *,
        canonical_source_key: str,
        source_type: str,
        source_uri: str,
        title: str | None,
        document_id: UUID | None = None,
    ) -> UUID:
        row = await self._database.fetchrow(
            _ENSURE_DOCUMENT,
            document_id or uuid4(),
            source_type,
            source_uri,
            canonical_source_key,
            title,
        )
        if row is None:
            raise RuntimeError("Failed to ensure document row.")
        return UUID(str(row["id"]))

    async def find_version(self, document_id: UUID, sha256: str) -> PublishedVersion | None:
        row = await self._database.fetchrow(_FIND_VERSION, document_id, sha256)
        if row is None:
            return None
        return PublishedVersion(
            document_id=UUID(str(row["document_id"])),
            document_version_id=UUID(str(row["document_version_id"])),
            created=False,
            chunk_count=int(row["chunk_count"]),
            rejected_count=0,
        )

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
        storage_path: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PublishedVersion:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")
        async with self._database.transaction() as tx:
            return await self._publish_tx(
                tx,
                document_id=document_id,
                validated=validated,
                parser_name=parser_name,
                parser_version=parser_version,
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=embedding_model or None,
                rejected=rejected,
                storage_path=storage_path,
                metadata=metadata or {},
            )

    async def _publish_tx(
        self,
        tx: QueryExecutor,
        *,
        document_id: UUID,
        validated: ValidatedFile,
        parser_name: str,
        parser_version: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str | None,
        rejected: int,
        storage_path: str | None,
        metadata: dict[str, object],
    ) -> PublishedVersion:
        version_id = uuid4()
        dimensions = len(embeddings[0]) if embeddings else None
        version_row = await tx.fetchrow(
            _INSERT_VERSION,
            version_id,
            document_id,
            validated.sha256,
            validated.media_type,
            validated.byte_size,
            storage_path,
            f"{parser_name}/{parser_version}",
            embedding_model if embeddings else None,
            dimensions,
            json.dumps(metadata, separators=(",", ":")),
        )
        if version_row is None:
            raise RuntimeError("Failed to insert document version.")
        created = bool(version_row["inserted"])
        resolved_version = UUID(str(version_row["id"]))
        if created:
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True)):
                await tx.execute(
                    _INSERT_CHUNK,
                    resolved_version,
                    index,
                    chunk.content,
                    _coords_json(chunk.coordinates),
                    chunk.content_sha256,
                    _vector_literal(vector),
                    embedding_model,
                    chunk.token_count,
                )
        await tx.execute(_SET_CURRENT, document_id, resolved_version)
        return PublishedVersion(
            document_id=document_id,
            document_version_id=resolved_version,
            created=created,
            chunk_count=len(chunks) if created else 0,
            rejected_count=rejected,
        )

    async def publish_auction_rows(
        self,
        *,
        document_version_id: UUID,
        source_url: str,
        source_name: str,
        rows: Sequence[PublishableAuctionRow],
    ) -> int:
        now = datetime.now(UTC)
        written = 0
        async with self._database.transaction() as tx:
            for row in rows:
                record_id = await self._upsert_auction_row(
                    tx,
                    document_version_id=document_version_id,
                    source_url=source_url,
                    source_name=source_name,
                    row=row,
                    ingested_at=now,
                )
                await tx.execute(_DELETE_MORTGAGORS, record_id)
                for ordinal, mortgagor in enumerate(row.normalized.mortgagors):
                    await tx.execute(
                        _INSERT_MORTGAGOR,
                        record_id,
                        ordinal,
                        mortgagor.display_name,
                        mortgagor.full_name,
                        mortgagor.first_name,
                        mortgagor.last_name,
                        _coords_json(row.normalized.coordinates),
                    )
                written += 1
        return written

    async def _upsert_auction_row(
        self,
        tx: QueryExecutor,
        *,
        document_version_id: UUID,
        source_url: str,
        source_name: str,
        row: PublishableAuctionRow,
        ingested_at: datetime,
    ) -> UUID:
        normalized = row.normalized
        trustee_name = normalized.trustee.display_name if normalized.trustee else None
        normalized_trustee = normalized.trustee.full_name if normalized.trustee else None
        normalized_address = normalized.address.normalized if normalized.address else None
        metadata: dict[str, Any] = {
            "report_year": row.report_year,
            "report_month": row.report_month,
        }
        coordinates = {
            **(
                normalized.coordinates.model_dump(mode="json")
                if hasattr(normalized.coordinates, "model_dump")
                else {}
            ),
            "report_year": row.report_year,
            "report_month": row.report_month,
        }
        result = await tx.fetchrow(
            _UPSERT_AUCTION,
            row.county,
            row.sale_date,
            row.property_address,
            row.city,
            row.state or "TX",
            row.zip_code,
            trustee_name,
            normalized.amounts.opening_bid,
            source_url,
            source_name,
            json.dumps(metadata, separators=(",", ":")),
            row.rid,
            normalized.stable_key,
            document_version_id,
            json.dumps(coordinates, separators=(",", ":")),
            normalized_trustee,
            normalized_address,
            row.report_year,
            row.report_month,
            row.loan_type,
            normalized.amounts.estimated_equity,
            normalized.amounts.estimated_margin,
            normalized.normalization_version,
            ingested_at,
        )
        if result is None:
            raise RuntimeError("Failed to upsert auction record.")
        return UUID(str(result["id"]))
