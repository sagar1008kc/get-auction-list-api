"""Framework-independent structured auction search and response construction."""

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from get_auction_list_api.api.metrics import AUCTION_SEARCHES
from get_auction_list_api.auth import Principal
from get_auction_list_api.domain import AuctionRecord, AuctionSearchFilters


class MatchLevel(StrEnum):
    CANONICAL = "canonical"
    EXACT = "exact"
    PREFIX = "prefix"
    FUZZY = "fuzzy"
    CANDIDATE = "candidate"


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    source_kind: str = "auction_record"
    title: str
    official_source: bool
    url: str | None
    document_version_id: str | None
    page_number: int | None = None
    sheet_name: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    retrieved_at: datetime


class CTAFilters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    trustee: str | None = None
    mortgagor_first_name: str | None = Field(default=None, alias="mortgagorFirstName")
    mortgagor_last_name: str | None = Field(default=None, alias="mortgagorLastName")
    year: int | None = None
    month: int | None = None


class AuctionCTA(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = "View matching auctions"
    href: str = "/dashboard/auctions-list"
    filters: CTAFilters


class AuctionSearchItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record: AuctionRecord
    match_level: MatchLevel
    confidence: float = Field(ge=0, le=1)
    citation_id: str
    limitations: tuple[str, ...] = ()


class AuctionSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    results: tuple[AuctionSearchItem, ...]
    citations: tuple[Citation, ...]
    cta: AuctionCTA | None


class SearchRepository(Protocol):
    async def search(
        self,
        principal: Principal,
        filters: AuctionSearchFilters,
    ) -> Sequence[AuctionRecord]: ...


def _level(score: float, *, stable_key: str | None) -> tuple[MatchLevel, float]:
    if stable_key is not None and score >= 4:
        return MatchLevel.CANONICAL, 1.0
    if score >= 3:
        return MatchLevel.EXACT, min(1.0, score / 3)
    if score >= 2:
        return MatchLevel.PREFIX, min(0.94, score / 3)
    if score >= 0.55:
        return MatchLevel.FUZZY, min(0.79, score)
    return MatchLevel.CANDIDATE, max(0.0, min(0.54, score))


class AuctionSearchService:
    def __init__(self, repository: SearchRepository) -> None:
        self._repository = repository

    async def search(
        self,
        principal: Principal,
        filters: AuctionSearchFilters,
    ) -> AuctionSearchResponse:
        records = await self._repository.search(principal, filters)
        AUCTION_SEARCHES.labels("match" if records else "no_match").inc()
        items: list[AuctionSearchItem] = []
        citations: list[Citation] = []
        now = datetime.now(UTC)
        for index, record in enumerate(records, start=1):
            citation_id = f"auction-{index}"
            level, confidence = _level(record.match_score, stable_key=record.stable_key)
            limitations = (
                ("Candidate match; verify names and address against the cited source.",)
                if level is MatchLevel.CANDIDATE
                else ()
            )
            coordinates = record.source_coordinates
            citations.append(
                Citation(
                    id=citation_id,
                    title=record.source_name,
                    official_source="wilco" in record.source_url,
                    url=record.source_url,
                    document_version_id=(
                        str(record.document_version_id) if record.document_version_id else None
                    ),
                    page_number=_integer(coordinates.get("page_number")),
                    sheet_name=_string(coordinates.get("sheet_name")),
                    row_start=_integer(coordinates.get("row_start")),
                    row_end=_integer(coordinates.get("row_end")),
                    retrieved_at=now,
                )
            )
            items.append(
                AuctionSearchItem(
                    record=record,
                    match_level=level,
                    confidence=confidence,
                    citation_id=citation_id,
                    limitations=limitations,
                )
            )
        cta = None
        if records:
            cta = AuctionCTA(
                filters=CTAFilters(
                    trustee=filters.trustee,
                    mortgagor_first_name=filters.mortgagor_first,
                    mortgagor_last_name=filters.mortgagor_last,
                    year=filters.report_year,
                    month=filters.report_month,
                )
            )
        return AuctionSearchResponse(
            results=tuple(items),
            citations=tuple(citations),
            cta=cta,
        )


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None
