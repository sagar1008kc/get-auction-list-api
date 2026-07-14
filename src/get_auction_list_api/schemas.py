"""Public API schemas using snake_case output and documented camelCase input aliases."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Intent(StrEnum):
    KNOWLEDGE_POLICY = "knowledge_policy"
    AUCTION_SEARCH = "auction_search"
    PUBLIC_PROPERTY_LOOKUP = "public_property_lookup"
    COMBINED_RESEARCH = "combined_research"
    UNSUPPORTED_OR_UNSAFE = "unsupported_or_unsafe"


class ChatContext(ApiModel):
    current_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("current_path", "currentPath"),
        max_length=500,
    )
    selected_auction_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("selected_auction_id", "selectedAuctionId"),
    )

    @field_validator("current_path")
    @classmethod
    def relative_path_only(cls, value: str | None) -> str | None:
        if value is not None and (not value.startswith("/") or value.startswith("//")):
            raise ValueError("current_path must be a same-origin relative path")
        return value


class ChatRequest(ApiModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("thread_id", "threadId"),
    )
    client_message_id: UUID = Field(
        validation_alias=AliasChoices("client_message_id", "clientMessageId")
    )
    locale: str = Field(default="en-US", pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
    timezone: str = Field(default="UTC", max_length=100)
    context: ChatContext = Field(default_factory=ChatContext)

    @field_validator("message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class Citation(ApiModel):
    id: str
    source_kind: str
    title: str
    official_source: bool
    url: str | None = None
    document_id: UUID | None = None
    document_version_id: UUID | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    chunk_id: UUID | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    quote: str | None = Field(default=None, max_length=500)


class Mortgagor(ApiModel):
    display_name: str
    first_name: str | None = None
    last_name: str | None = None


class AuctionResultCard(ApiModel):
    record_key: str
    fc_id: str | None = None
    match_level: Literal["canonical", "exact", "prefix", "fuzzy", "candidate"]
    match_confidence: float = Field(ge=0, le=1)
    county: str = "Williamson"
    report_year: int | None = None
    report_month: int | None = None
    auction_date: str | None = None
    property_address: str | None = None
    city: str | None = None
    state: str | None = "TX"
    zip_code: str | None = None
    trustees: list[str] = Field(default_factory=list)
    mortgagors: list[Mortgagor] = Field(default_factory=list)
    loan_type: str | None = None
    estimated_equity: Decimal | None = None
    estimated_margin: Decimal | None = None
    source_citation_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class WcadCandidate(ApiModel):
    property_id: str
    address: str
    owner_name: str | None = None
    legal_description: str | None = None
    market_value: str | None = None
    zip_code: str | None = None
    subdivision: str | None = None
    parcel_or_account_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    source_citation_ids: list[str] = Field(default_factory=list)
    retrieved_at: datetime | None = None


class PropertySummary(ApiModel):
    property_id: str
    address: str
    owner_name: str | None = None
    legal_description: str | None = None
    market_value: str | None = None
    source_citation_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    candidates: list[WcadCandidate] = Field(default_factory=list)
    selected_property_id: str | None = None
    match_confidence: float | None = Field(default=None, ge=0, le=1)
    requires_user_selection: bool = False


class CTAFilters(ApiModel):
    trustee: str | None = None
    mortgagor_first_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("mortgagor_first_name", "mortgagorFirstName"),
    )
    mortgagor_last_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("mortgagor_last_name", "mortgagorLastName"),
    )
    address: str | None = None
    city: str | None = None
    zip_code: str | None = Field(
        default=None,
        validation_alias=AliasChoices("zip_code", "zipCode"),
    )
    year: int | None = None
    month: int | None = None


class CTA(ApiModel):
    label: str = "View matching auctions"
    href: str = "/dashboard/auctions-list"
    filters: CTAFilters


class UnavailableSource(ApiModel):
    source: str
    reason: str
    retryable: bool


class FinalResponse(ApiModel):
    request_id: str
    correlation_id: str
    trace_id: str
    run_id: UUID
    thread_id: UUID
    assistant_message_id: UUID
    # Additive status value: "failed" covers complete dependency failure.
    status: Literal["completed", "partial", "failed"]
    intent: Intent
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    auction_results: list[AuctionResultCard] = Field(default_factory=list)
    property_summary: PropertySummary | None = None
    cta: CTA | None = None
    disclaimer: str | None = None
    unavailable_sources: list[UnavailableSource] = Field(default_factory=list)
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Final answer confidence (not router/intent confidence).",
    )


FeedbackCategory = Annotated[
    Literal[
        "incorrect_match",
        "missing_citation",
        "outdated_source",
        "unsafe",
        "unhelpful",
        "other",
    ],
    Field(),
]


class FeedbackRequest(ApiModel):
    run_id: UUID
    message_id: UUID
    rating: Literal["up", "down"]
    categories: list[FeedbackCategory] = Field(default_factory=list, max_length=6)
    comment: str | None = Field(default=None, max_length=1000)
    cta_clicked: bool = False


class FeedbackResponse(ApiModel):
    feedback_id: UUID
    created: bool
