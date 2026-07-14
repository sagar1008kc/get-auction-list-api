"""Production ingestion handler for Storage spreadsheets and approved policy URLs.

Configure the worker with::

    GET_AUCTION_LIST_INGESTION_HANDLER=get_auction_list_api.ingestion.handler:handle_ingestion_job
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import httpx

from get_auction_list_api.config import Settings, get_settings
from get_auction_list_api.database import AsyncDatabase
from get_auction_list_api.ingestion.auction_rows import build_publishable_rows
from get_auction_list_api.ingestion.chunking import chunk_units
from get_auction_list_api.ingestion.path_meta import parse_auction_storage_path
from get_auction_list_api.ingestion.publisher import PostgresDocumentPublisher
from get_auction_list_api.ingestion.queue import IngestionJob
from get_auction_list_api.ingestion.sources import APPROVED_SOURCES, SourceRegistry
from get_auction_list_api.ingestion.storage import (
    download_supabase_object,
    parse_supabase_uri,
)
from get_auction_list_api.ingestion.validation import validate_file
from get_auction_list_api.llm import (
    OpenAIEmbeddingProvider,
    create_embedding_transport,
    create_openai_client,
)
from get_auction_list_api.parsers.html import HtmlParser
from get_auction_list_api.parsers.xlsx import XlsxParser

_REGISTRY = SourceRegistry()
_POLICY_KEYS = {source.key for source in APPROVED_SOURCES if source.kind.value == "policy_html"}


class IngestionHandlerError(RuntimeError):
    pass


async def handle_ingestion_job(job: IngestionJob) -> Mapping[str, object]:
    """Worker entrypoint: download, validate, publish, and never invent source bytes."""

    settings = get_settings()
    if settings.database_url is None:
        raise IngestionHandlerError("GET_AUCTION_LIST_DATABASE_URL is required for ingestion.")
    database = AsyncDatabase(
        settings.database_url.get_secret_value(),
        min_size=1,
        max_size=2,
        command_timeout=settings.database_command_timeout_seconds,
    )
    await database.connect()
    try:
        publisher = PostgresDocumentPublisher(database)
        payload = dict(job.input)
        if job.source_uri.startswith("supabase://"):
            return await _handle_storage(
                job, settings=settings, publisher=publisher, payload=payload
            )
        if job.source_uri.startswith("registered://"):
            return await _handle_registered(
                job, settings=settings, publisher=publisher, payload=payload
            )
        raise IngestionHandlerError(f"Unsupported source URI scheme for job {job.id}.")
    finally:
        await database.close()


async def _handle_storage(
    job: IngestionJob,
    *,
    settings: Settings,
    publisher: PostgresDocumentPublisher,
    payload: dict[str, Any],
) -> dict[str, object]:
    if settings.supabase_url is None or settings.supabase_service_role_key is None:
        raise IngestionHandlerError(
            "Storage ingestion requires GET_AUCTION_LIST_SUPABASE_URL and "
            "GET_AUCTION_LIST_SUPABASE_SERVICE_ROLE_KEY."
        )
    bucket, path = parse_supabase_uri(job.source_uri)
    if payload.get("storage_bucket") and str(payload["storage_bucket"]) != bucket:
        raise IngestionHandlerError("Payload storage_bucket does not match source_uri.")
    if payload.get("storage_path") and str(payload["storage_path"]) != path:
        raise IngestionHandlerError("Payload storage_path does not match source_uri.")

    downloaded = await download_supabase_object(
        supabase_url=settings.supabase_url,
        service_role_key=settings.supabase_service_role_key.get_secret_value(),
        bucket=bucket,
        path=path,
        timeout_seconds=settings.openai_timeout_seconds * 3,
    )
    validated = validate_file(
        downloaded.content,
        declared_media_type=downloaded.media_type
        or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    document_type = str(payload.get("document_type") or "auction_spreadsheet")
    meta = parse_auction_storage_path(path)
    document_id = await publisher.ensure_document(
        canonical_source_key=job.source_uri,
        source_type="supabase_storage",
        source_uri=job.source_uri,
        title=f"{meta.county} auction list {meta.report_period}",
        document_id=job.document_id,
    )
    existing = await publisher.find_version(document_id, validated.sha256)
    force = bool(payload.get("force_new_version"))
    if existing is not None and not force:
        # Still ensure auction rows exist for this checksum/version (idempotent upsert).
        parsed = XlsxParser().parse(validated.content)
        rows = build_publishable_rows(parsed.units, meta=meta)
        written = await publisher.publish_auction_rows(
            document_version_id=existing.document_version_id,
            source_url=job.source_uri,
            source_name=f"{meta.county} indexed auction list",
            rows=rows,
        )
        return {
            "document_id": str(document_id),
            "document_version_id": str(existing.document_version_id),
            "unchanged": True,
            "auction_rows": written,
            "report_year": meta.report_year,
            "report_month": meta.report_month,
            "document_type": document_type,
        }

    parsed = XlsxParser().parse(validated.content)
    rows = build_publishable_rows(parsed.units, meta=meta)
    if not rows:
        raise IngestionHandlerError("Spreadsheet produced no publishable auction rows.")
    version = await publisher.publish(
        document_id=document_id,
        validated=validated,
        parser_name=XlsxParser.name,
        parser_version=XlsxParser().version,
        chunks=(),
        embeddings=(),
        embedding_model="none",
        rejected=len(parsed.errors),
        storage_path=path,
        metadata={
            "document_type": document_type,
            "report_year": meta.report_year,
            "report_month": meta.report_month,
            "county": meta.county,
        },
    )
    written = await publisher.publish_auction_rows(
        document_version_id=version.document_version_id,
        source_url=job.source_uri,
        source_name=f"{meta.county} indexed auction list",
        rows=rows,
    )
    return {
        "document_id": str(document_id),
        "document_version_id": str(version.document_version_id),
        "unchanged": False,
        "auction_rows": written,
        "rejected_rows": len(parsed.errors),
        "report_year": meta.report_year,
        "report_month": meta.report_month,
        "document_type": document_type,
    }


async def _handle_registered(
    job: IngestionJob,
    *,
    settings: Settings,
    publisher: PostgresDocumentPublisher,
    payload: dict[str, Any],
) -> dict[str, object]:
    source_key = str(payload.get("source_id") or job.source_uri.removeprefix("registered://"))
    if source_key not in _POLICY_KEYS:
        # Allow lookup by ApprovedSource.key even when callers pass the registry key.
        try:
            source = _REGISTRY.get(source_key)
        except ValueError as error:
            raise IngestionHandlerError(
                "Registered ingestion currently supports approved policy HTML sources only."
            ) from error
    else:
        source = _REGISTRY.get(source_key)
    if source.key not in _POLICY_KEYS:
        raise IngestionHandlerError("Only approved policy HTML sources are ingested into RAG.")
    if settings.openai_api_key is None:
        raise IngestionHandlerError(
            "Policy RAG ingestion requires GET_AUCTION_LIST_OPENAI_API_KEY."
        )

    url = str(source.base_url)
    async with httpx.AsyncClient(
        timeout=settings.public_http_timeout_seconds,
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
    if response.status_code >= 400:
        raise IngestionHandlerError(f"Policy source fetch failed with HTTP {response.status_code}.")
    validated = validate_file(response.content, declared_media_type="text/html")
    parsed = HtmlParser().parse(validated.content, source_url=url)
    chunks = chunk_units(parsed.units)
    if not chunks:
        raise IngestionHandlerError("Policy HTML produced no publishable chunks.")

    openai_client = create_openai_client(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
    try:
        embeddings = OpenAIEmbeddingProvider(
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            transport=create_embedding_transport(openai_client),
        )
        batch = await embeddings.embed([chunk.content for chunk in chunks])
    finally:
        await openai_client.close()

    document_id = await publisher.ensure_document(
        canonical_source_key=source.key,
        source_type="registered_url",
        source_uri=url,
        title=source.title,
        document_id=job.document_id or uuid4(),
    )
    existing = await publisher.find_version(document_id, validated.sha256)
    if existing is not None and not bool(payload.get("force_new_version")):
        return {
            "document_id": str(document_id),
            "document_version_id": str(existing.document_version_id),
            "unchanged": True,
            "chunk_count": existing.chunk_count,
            "source_key": source.key,
        }
    version = await publisher.publish(
        document_id=document_id,
        validated=validated,
        parser_name=HtmlParser.name,
        parser_version=HtmlParser.version,
        chunks=chunks,
        embeddings=batch.vectors,
        embedding_model=batch.model,
        rejected=len(parsed.errors),
        storage_path=None,
        metadata={"source_key": source.key, "official": source.official},
    )
    return {
        "document_id": str(document_id),
        "document_version_id": str(version.document_version_id),
        "unchanged": not version.created,
        "chunk_count": version.chunk_count,
        "source_key": source.key,
    }
