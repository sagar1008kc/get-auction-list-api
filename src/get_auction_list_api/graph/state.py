"""Checkpoint-safe state for the single controlled workflow."""

from typing import Annotated, NotRequired, TypedDict

from get_auction_list_api.schemas import (
    AuctionResultCard,
    Citation,
    FinalResponse,
    Intent,
    PropertySummary,
    UnavailableSource,
    WcadCandidate,
)


def merge_lists[T](left: list[T] | None, right: list[T] | None) -> list[T]:
    """Reducer for parallel branch outputs; never invents values on its own."""

    return [*(left or []), *(right or [])]


class RetryBudget(TypedDict):
    classifier: int
    retrieval: int
    tools: int


class CapabilityFlags(TypedDict):
    policy: bool
    auction: bool
    public_record: bool


class AgentState(TypedDict):
    request_id: str
    correlation_id: str
    trace_id: str
    run_id: str
    thread_id: str
    assistant_message_id: str
    user_id: str
    message: str
    locale: str
    timezone: str
    intent: NotRequired[Intent]
    intent_confidence: NotRequired[float]
    retrieval_confidence: NotRequired[float]
    auction_match_confidence: NotRequired[float]
    property_correlation_confidence: NotRequired[float]
    grounding_confidence: NotRequired[float]
    final_answer_confidence: NotRequired[float]
    capabilities: NotRequired[CapabilityFlags]
    entities: NotRequired[dict[str, str | int]]
    evidence: NotRequired[Annotated[list[dict[str, object]], merge_lists]]
    citations: NotRequired[Annotated[list[Citation], merge_lists]]
    auction_results: NotRequired[Annotated[list[AuctionResultCard], merge_lists]]
    wcad_candidates: NotRequired[Annotated[list[WcadCandidate], merge_lists]]
    property_summary: NotRequired[PropertySummary | None]
    unavailable_sources: NotRequired[Annotated[list[UnavailableSource], merge_lists]]
    errors: NotRequired[Annotated[list[str], merge_lists]]
    retry_budget: RetryBudget
    grounded: NotRequired[bool]
    disclaimer: NotRequired[str | None]
    answer: NotRequired[str]
    answer_citation_ids: NotRequired[list[str]]
    final_response: NotRequired[FinalResponse]
