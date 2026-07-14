"""Framework-independent domain contracts and deterministic normalization."""

import hashlib
import re
import unicodedata
from datetime import date
from decimal import Decimal
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[_\W]+")


def normalize_search_text(value: str | None) -> str | None:
    """Produce a conservative matching form without inventing source values."""

    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = _SPACE.sub(" ", _PUNCTUATION.sub(" ", normalized)).strip()
    return normalized or None


def stable_auction_key(
    *,
    source_key: str,
    source_record_key: str,
    normalization_version: str,
) -> str:
    """Hash immutable source identity; RID is intentionally not part of identity."""

    parts = (source_key.strip(), source_record_key.strip(), normalization_version.strip())
    if not all(parts):
        raise ValueError("Stable identity components must be non-empty.")
    canonical = "\x1f".join(parts).encode()
    return hashlib.sha256(canonical).hexdigest()


def stable_idempotency_key(
    *,
    operation: str,
    principal_scope: str,
    canonical_payload: str,
) -> str:
    """Derive a retry key from an explicitly canonicalized request payload."""

    parts = (operation.strip(), principal_scope.strip(), canonical_payload)
    if not all(parts):
        raise ValueError("Idempotency components must be non-empty.")
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class AuctionSearchFilters(BaseModel):
    """Validated structured-search filters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trustee: str | None = Field(default=None, max_length=200)
    mortgagor_first: str | None = Field(default=None, max_length=100)
    mortgagor_last: str | None = Field(default=None, max_length=100)
    address: str | None = Field(default=None, max_length=300)
    city: str | None = Field(default=None, max_length=100)
    zip_code: str | None = Field(default=None, pattern=r"^\d{5}(?:-\d{4})?$")
    report_year: int | None = Field(default=None, ge=2000, le=2100)
    report_month: int | None = Field(default=None, ge=1, le=12)
    loan_type: str | None = Field(default=None, max_length=80)
    min_equity: Decimal | None = None
    max_equity: Decimal | None = None
    min_margin: Decimal | None = None
    max_margin: Decimal | None = None
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if (
            self.min_equity is not None
            and self.max_equity is not None
            and self.min_equity > self.max_equity
        ):
            raise ValueError("min_equity cannot exceed max_equity")
        if (
            self.min_margin is not None
            and self.max_margin is not None
            and self.min_margin > self.max_margin
        ):
            raise ValueError("min_margin cannot exceed max_margin")
        return self

    def normalized(self) -> Self:
        values = self.model_dump()
        for field in (
            "trustee",
            "mortgagor_first",
            "mortgagor_last",
            "address",
            "city",
            "loan_type",
        ):
            values[field] = normalize_search_text(values[field])
        return type(self).model_validate(values)


class AuctionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    rid: str | None = None
    stable_key: str | None = None
    property_address: str | None = None
    city: str | None = None
    zip_code: str | None = None
    trustee_name: str | None = None
    sale_date: date | None = None
    source_url: str
    source_name: str
    document_version_id: UUID | None = None
    source_coordinates: dict[str, object] = Field(default_factory=dict)
    match_score: float


class RetrievalMatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    content: str
    source_coordinates: dict[str, object]
    score: float


class AuctionListCTA(BaseModel):
    """Dashboard payload limited to its five supported filter fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trustee: str | None = None
    mortgagor: str | None = None
    address: str | None = None
    city: str | None = None
    zip_code: str | None = None
