"""Strict evidence, filter, context, and citation contracts."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RetrievalChannel(StrEnum):
    FTS = "fts"
    VECTOR = "vector"
    FUSED = "fused"


class RetrievedEvidence(BaseModel):
    """The only evidence shape accepted by the RAG synthesis boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    source_uri: str
    title: str
    content: str
    source_coordinates: dict[str, object]
    score: float
    channel: RetrievalChannel
    token_count: int = Field(ge=0)
    approved: bool
    untrusted: bool = True


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_ids: tuple[UUID, ...] | None = None
    document_types: tuple[str, ...] | None = None
    source_keys: tuple[str, ...] | None = None


class EvidenceCitation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    title: str
    url: str | None
    page_number: int | None = None
    sheet_name: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    quote: str
    retrieved_at: datetime


class RAGContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence: tuple[RetrievedEvidence, ...]
    citations: tuple[EvidenceCitation, ...]
    rendered_context: str
    confidence: float = Field(ge=0, le=1)
    no_answer: bool
    no_answer_message: str | None = None
