"""Repository boundaries and PostgreSQL implementations."""

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.database import QueryExecutor
from get_auction_list_api.domain import AuctionRecord, AuctionSearchFilters, RetrievalMatch

_SEARCH_SQL = """
select *
from public.search_auction_records(
  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
)
"""

_HYBRID_SQL = """
select *
from public.hybrid_retrieve_document_chunks($1, $2::extensions.vector, $3, $4::uuid[])
"""

_CREATE_THREAD_SQL = """
insert into public.conversation_threads (id, user_id, title)
values ($1, $2, $3)
on conflict (id) do update
set title = excluded.title, updated_at = now()
where conversation_threads.user_id = excluded.user_id
returning id
"""


class AuctionRepository(Protocol):
    async def search(
        self,
        principal: Principal,
        filters: AuctionSearchFilters,
    ) -> Sequence[AuctionRecord]: ...


class RetrievalRepository(Protocol):
    async def hybrid(
        self,
        principal: Principal,
        *,
        query: str,
        embedding: Sequence[float],
        limit: int = 12,
        document_ids: Sequence[UUID] | None = None,
    ) -> Sequence[RetrievalMatch]: ...


class ConversationRepository(Protocol):
    async def create_thread(
        self,
        principal: Principal,
        *,
        thread_id: UUID,
        title: str | None,
    ) -> UUID: ...


class PostgresAuctionRepository:
    def __init__(self, executor: QueryExecutor) -> None:
        self._executor = executor

    async def search(
        self,
        principal: Principal,
        filters: AuctionSearchFilters,
    ) -> Sequence[AuctionRecord]:
        principal.require(Permission.AUCTION_READ)
        value = filters.normalized()
        rows = await self._executor.fetch(
            _SEARCH_SQL,
            value.trustee,
            value.mortgagor_first,
            value.mortgagor_last,
            value.address,
            value.city,
            value.zip_code,
            value.report_year,
            value.report_month,
            value.loan_type,
            value.min_equity,
            value.max_equity,
            value.min_margin,
            value.max_margin,
            value.limit,
            value.offset,
        )
        return [AuctionRecord.model_validate(dict(row)) for row in rows]


class PostgresRetrievalRepository:
    def __init__(self, executor: QueryExecutor) -> None:
        self._executor = executor

    async def hybrid(
        self,
        principal: Principal,
        *,
        query: str,
        embedding: Sequence[float],
        limit: int = 12,
        document_ids: Sequence[UUID] | None = None,
    ) -> Sequence[RetrievalMatch]:
        principal.require(Permission.DOCUMENT_READ)
        if len(embedding) != 1536:
            raise ValueError("Embedding must have exactly 1536 dimensions.")
        if not 1 <= limit <= 50:
            raise ValueError("Retrieval limit must be between 1 and 50.")
        vector = "[" + ",".join(format(component, ".9g") for component in embedding) + "]"
        rows = await self._executor.fetch(
            _HYBRID_SQL,
            query,
            vector,
            limit,
            list(document_ids) if document_ids is not None else None,
        )
        return [RetrievalMatch.model_validate(dict(row)) for row in rows]


class PostgresConversationRepository:
    def __init__(self, executor: QueryExecutor) -> None:
        self._executor = executor

    async def create_thread(
        self,
        principal: Principal,
        *,
        thread_id: UUID,
        title: str | None,
    ) -> UUID:
        row = await self._executor.fetchrow(
            _CREATE_THREAD_SQL,
            thread_id,
            principal.user_id,
            title,
        )
        if row is None:
            raise PermissionError("Thread identity belongs to another principal.")
        return UUID(str(row["id"]))
