"""Validated contracts for read-only public-record operations."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ToolErrorCategory(StrEnum):
    INVALID_INPUT = "invalid_input"
    FORBIDDEN_DESTINATION = "forbidden_destination"
    TIMEOUT = "timeout"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    RESPONSE_TOO_LARGE = "response_too_large"
    CONTRACT_CHANGED = "contract_changed"
    CANCELLED = "cancelled"


class PublicRecordToolError(Exception):
    def __init__(
        self, category: ToolErrorCategory, message: str, *, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.retryable = retryable


class ToolMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parser_version: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str
    audit_id: str
    source_url: HttpUrl
    cache_hit: bool = False


class TrusteeSaleSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    url: HttpUrl
    document_type: str


class ForeclosureRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    notice_url: HttpUrl
    property_address: str | None = None
    sale_date: str | None = None
    trustee: str | None = None


class ForeclosureNotice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    title: str
    text: str
    notice_url: HttpUrl


class WcadProperty(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    property_id: str
    address: str
    owner_name: str | None = None
    detail_url: HttpUrl


class WcadPropertyDetails(WcadProperty):
    legal_description: str | None = None
    market_value: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)


class CorrelationCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    foreclosure_record_id: str
    property_id: str
    confidence: float = Field(ge=0, le=1)
    matched_fields: tuple[str, ...]
    differing_fields: tuple[str, ...]


class ToolResult(BaseModel):
    """Stable envelope shared by all six MCP operations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[dict[str, Any], ...]
    metadata: ToolMetadata
    warnings: tuple[str, ...] = ()
