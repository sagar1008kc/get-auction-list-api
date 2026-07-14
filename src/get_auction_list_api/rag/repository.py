"""Parameterized PostgreSQL FTS and vector retrieval calls."""

from collections.abc import Sequence

from get_auction_list_api.database import QueryExecutor
from get_auction_list_api.rag.models import RetrievalChannel, RetrievalFilters, RetrievedEvidence

_FTS_SQL = """
select dc.id as chunk_id, dv.document_id, dc.document_version_id, d.source_uri, d.title,
       dc.content, dc.source_coordinates, coalesce(dc.token_count, 0) as token_count,
       true as approved, true as untrusted,
       ts_rank_cd(dc.search_vector, websearch_to_tsquery('english', $1))::float8 as score
from public.document_chunks dc
join public.document_versions dv on dv.id = dc.document_version_id
join public.documents d on d.id = dv.document_id
where d.status = 'active'
  and dc.search_vector @@ websearch_to_tsquery('english', $1)
  and ($2::uuid[] is null or d.id = any($2))
  and ($3::text[] is null or d.source_type = any($3))
  and ($4::text[] is null or d.canonical_source_key = any($4))
order by score desc, dc.id
limit $5
"""

_VECTOR_SQL = """
select dc.id as chunk_id, dv.document_id, dc.document_version_id, d.source_uri, d.title,
       dc.content, dc.source_coordinates, coalesce(dc.token_count, 0) as token_count,
       true as approved, true as untrusted,
       (1 - (dc.embedding <=> $1::extensions.vector))::float8 as score
from public.document_chunks dc
join public.document_versions dv on dv.id = dc.document_version_id
join public.documents d on d.id = dv.document_id
where d.status = 'active' and dc.embedding is not null
  and ($2::uuid[] is null or d.id = any($2))
  and ($3::text[] is null or d.source_type = any($3))
  and ($4::text[] is null or d.canonical_source_key = any($4))
order by dc.embedding <=> $1::extensions.vector, dc.id
limit $5
"""


class PostgresRAGRepository:
    def __init__(self, executor: QueryExecutor, *, dimensions: int = 1536) -> None:
        self._executor = executor
        self._dimensions = dimensions

    async def fts(
        self,
        *,
        query: str,
        filters: RetrievalFilters,
        limit: int,
    ) -> Sequence[RetrievedEvidence]:
        self._validate(query=query, limit=limit)
        rows = await self._executor.fetch(
            _FTS_SQL,
            query,
            list(filters.document_ids) if filters.document_ids else None,
            list(filters.document_types) if filters.document_types else None,
            list(filters.source_keys) if filters.source_keys else None,
            limit,
        )
        return [
            RetrievedEvidence.model_validate({**dict(row), "channel": RetrievalChannel.FTS})
            for row in rows
        ]

    async def vector(
        self,
        *,
        embedding: Sequence[float],
        filters: RetrievalFilters,
        limit: int,
    ) -> Sequence[RetrievedEvidence]:
        if len(embedding) != self._dimensions:
            raise ValueError("Query embedding dimensions do not match the index.")
        self._validate(query="vector", limit=limit)
        vector = "[" + ",".join(format(component, ".9g") for component in embedding) + "]"
        rows = await self._executor.fetch(
            _VECTOR_SQL,
            vector,
            list(filters.document_ids) if filters.document_ids else None,
            list(filters.document_types) if filters.document_types else None,
            list(filters.source_keys) if filters.source_keys else None,
            limit,
        )
        return [
            RetrievedEvidence.model_validate({**dict(row), "channel": RetrievalChannel.VECTOR})
            for row in rows
        ]

    @staticmethod
    def _validate(*, query: str, limit: int) -> None:
        if not query.strip() or len(query) > 2000 or not 1 <= limit <= 100:
            raise ValueError("Invalid retrieval query or limit.")
