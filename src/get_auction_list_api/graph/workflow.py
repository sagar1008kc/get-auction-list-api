"""One bounded LangGraph workflow; routing never grants arbitrary tool access.

Capability surface stays simple and enterprise-bounded:
- RAG agent: ``knowledge_rag`` (approved policy/knowledge)
- Tool/MCP agent: ``mcp_public_tools`` (public records)
- Deterministic SQL tool: ``sql_auction_search`` (indexed auctions)

Combined research fan-out/fan-in uses native LangGraph ``Send``, not internal gather.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, Send

from get_auction_list_api.api.metrics import (
    AUCTION_SEARCHES,
    GRAPH_NODE_DURATION,
    RETRIEVAL_REQUESTS,
    RETRIEVAL_RESULTS,
)
from get_auction_list_api.graph.state import AgentState, CapabilityFlags
from get_auction_list_api.observability.langfuse import LangfuseTelemetry, ObservationType
from get_auction_list_api.observability.telemetry import tracer
from get_auction_list_api.schemas import (
    CTA,
    AuctionResultCard,
    Citation,
    CTAFilters,
    FinalResponse,
    Intent,
    PropertySummary,
    UnavailableSource,
    WcadCandidate,
)

DISCLAIMER = (
    "This AI-generated summary is for informational purposes only and is not legal, "
    "financial, or investment advice. Official county auction notices, Williamson Central "
    "Appraisal District (WCAD) records, and other official sources control over this "
    "summary. Records may be incomplete, delayed, updated, postponed, or cancelled. "
    "Always verify critical details with the appropriate official source before making "
    "decisions."
)
NO_MATCH = (
    "I did not find a matching record in the currently indexed auction data. Records may "
    "be incomplete, delayed, updated, postponed or cancelled. Verify with the appropriate "
    "official county source."
)
_INJECTION = re.compile(
    r"(?is)\b(ignore (?:all |any )?(?:previous|prior|system) instructions|"
    r"reveal (?:the )?(?:system )?prompt|developer message|chain[- ]of[- ]thought)\b"
)
_ADDRESS = re.compile(
    r"\b\d{1,7}\s+(?:[A-Za-z][A-Za-z0-9']*\s+){0,5}[A-Za-z][A-Za-z0-9']*\s+"
    r"(?:st|street|rd|road|dr|drive|ln|lane|ave|avenue|blvd|boulevard|ct|court|trl|trail)\b",
    re.IGNORECASE,
)
_TRUSTEE = re.compile(
    r"\btrustee\s+(?:named|is|for)?\s*"
    r"((?:[A-Za-z][A-Za-z'-]{0,39})(?:\s+[A-Za-z][A-Za-z'-]{0,39}){0,3})\b",
    re.IGNORECASE,
)
_MORTGAGOR = re.compile(
    r"\bmortgagor\s+(?:named|is|for)?\s*"
    r"((?:[A-Za-z][A-Za-z'-]{0,39})(?:\s+[A-Za-z][A-Za-z'-]{0,39}){0,2})\b",
    re.IGNORECASE,
)
_LAST_FIRST_NAME = re.compile(
    r"\b([A-Za-z][A-Za-z'-]{1,39}),\s*([A-Za-z][A-Za-z'-]{1,39}"
    r"(?:\s+[A-Za-z][A-Za-z'-]{1,39})?)\b"
)
_DETAIL_ABOUT = re.compile(
    r"\b(?:details?|info|information|tell me|who is|find|search|look\s*up|show)\b"
    r".{0,40}\b(?:about|for|on)\b",
    re.IGNORECASE,
)
_ZIP = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
_YEAR = re.compile(
    r"\b(?:report\s+year|year|in)\s+(20\d{2})\b|(?<!\d)(20\d{2})(?!\d)",
    re.I,
)
_CITY = re.compile(
    r"\b(?:in|at)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?=\s+(?:tx|texas|\d{5})|,|$)",
    re.IGNORECASE,
)
_STREET_SUFFIX = {
    "st": "street",
    "street": "street",
    "rd": "road",
    "road": "road",
    "dr": "drive",
    "drive": "drive",
    "ln": "lane",
    "lane": "lane",
    "ave": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "ct": "court",
    "court": "court",
    "trl": "trail",
    "trail": "trail",
}
_MONTH_NAMES = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}
_AUCTION_WORDS = frozenset(
    {
        "auction",
        "auctions",
        "listing",
        "listings",
        "trustee",
        "mortgagor",
        "foreclosure",
        "trustee sale",
        "how many",
    }
)
_POLICY_WORDS = frozenset({"privacy", "disclaimer", "policy", "terms", "terms of use"})
_SCHEDULE_WORDS = frozenset(
    {
        "schedule",
        "calendar",
        "sale date",
        "sale dates",
        "when is",
        "when will",
        "trustee sale schedule",
        "trustee sales schedule",
        "foreclosure schedule",
        "sale calendar",
    }
)
_PUBLIC_WORDS = frozenset(
    {
        "wcad",
        "appraisal",
        "appraisal district",
        "property tax",
        "public record",
        "public records",
        "owner of record",
        "parcel",
        "account number",
        "property id",
        "cad",
    }
)
_TRUSTEE_STOP = frozenset({"sale", "sales", "auction", "auctions", "sale date", "list"})
_NAME_BOUNDARY = frozenset(
    {
        "in",
        "at",
        "on",
        "for",
        "with",
        "and",
        "or",
        "near",
        "from",
        "year",
        "month",
        "trustee",
        "mortgagor",
        "auction",
        "foreclosure",
        "wcad",
    }
)
_EVIDENCE_EXCERPT_CHARS = 280
_SAFE_STREAM_EVENTS = frozenset(
    {
        "run.started",
        "route.selected",
        "retrieval.started",
        "retrieval.completed",
        "tool.started",
        "tool.completed",
        "answer.delta",
        "answer.completed",
        "run.failed",
    }
)


class Classifier(Protocol):
    async def classify(self, message: str) -> Intent: ...


class Synthesizer(Protocol):
    async def synthesize(self, state: AgentState) -> Any: ...


class EntityExtractor(Protocol):
    async def extract(self, message: str) -> dict[str, str | int]: ...


class ThreadOwnerStore(Protocol):
    async def ensure_owner(self, *, user_id: str, thread_id: str) -> None: ...


KnowledgeSearch = Callable[
    [str],
    Awaitable[tuple[list[dict[str, object]], list[Citation]]],
]
AuctionSearch = Callable[
    [Mapping[str, str | int]],
    Awaitable[tuple[list[AuctionResultCard], list[Citation]]],
]
PublicSearch = Callable[
    [Mapping[str, str | int]],
    Awaitable[
        tuple[
            PropertySummary | None,
            list[WcadCandidate],
            list[Citation],
            list[UnavailableSource],
        ]
    ],
]


async def _empty_knowledge(_query: str) -> tuple[list[dict[str, object]], list[Citation]]:
    return [], []


async def _empty_auction(
    _entities: Mapping[str, str | int],
) -> tuple[list[AuctionResultCard], list[Citation]]:
    return [], []


async def _empty_public(
    _entities: Mapping[str, str | int],
) -> tuple[
    PropertySummary | None,
    list[WcadCandidate],
    list[Citation],
    list[UnavailableSource],
]:
    return None, [], [], []


@dataclass(frozen=True, slots=True)
class GraphServices:
    classifier: Classifier | None = None
    synthesizer: Synthesizer | None = None
    entity_extractor: EntityExtractor | None = None
    knowledge_search: KnowledgeSearch = _empty_knowledge
    auction_search: AuctionSearch = _empty_auction
    public_search: PublicSearch = _empty_public
    classifier_timeout_seconds: float = 1.5
    node_timeout_seconds: float = 8
    extractor_timeout_seconds: float = 1.5


def _emit(event: str, data: dict[str, object] | None = None) -> None:
    if event not in _SAFE_STREAM_EVENTS:
        return
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"event": event, "data": data or {}})


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", value.casefold())
    tokens = []
    for token in cleaned.split():
        tokens.append(_STREET_SUFFIX.get(token, token))
    return " ".join(tokens)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match whole words/phrases so 'GetAuctionList' does not imply 'auction'."""

    if " " in phrase:
        return phrase in text
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _looks_like_storage_lookup(message: str) -> bool:
    """Person names, LAST FIRST forms, or addresses → indexed spreadsheet search."""

    text = message.casefold().strip()
    if not text:
        return False
    if _LAST_FIRST_NAME.search(message):
        return True
    if _ADDRESS.search(message):
        return True
    if _DETAIL_ABOUT.search(message) and re.search(r"[a-z]{2,}", text):
        # "provide details about zavala, angela" / "who is angela zavala"
        remainder = re.sub(
            r"\b(?:provide|details?|info|information|tell me|who is|find|search|"
            r"look\s*up|show|about|for|on|me|the|a|an|please)\b",
            " ",
            text,
            flags=re.I,
        )
        tokens = [token for token in remainder.split() if len(token) > 1]
        return len(tokens) >= 1
    # Bare "First Last" or "LAST FIRST" two-token name queries
    tokens = [token for token in re.sub(r"[^a-z0-9,\s'-]", " ", text).split() if token]
    if len(tokens) in {2, 3} and all(token.replace("'", "").isalpha() for token in tokens):
        stop = {
            "what",
            "when",
            "where",
            "how",
            "why",
            "is",
            "the",
            "a",
            "an",
            "for",
            "july",
            "august",
            "williamson",
            "county",
            "schedule",
            "policy",
            "privacy",
            "disclaimer",
        }
        if not any(token in stop for token in tokens):
            return True
    return False


def _wants_indexed_auction_rows(text: str) -> bool:
    """True when the user asks for spreadsheet-index filters, not only a county calendar."""

    if _looks_like_storage_lookup(text):
        return True
    if _contains_phrase(text, "mortgagor"):
        return True
    if "auction list" in text or "indexed auction" in text:
        return True
    if re.search(r"\btrustee\s+(?:named|is)\s+[a-z]", text):
        return True
    if re.search(r"\btrustee\s+[a-z][a-z'-]+\s+[a-z]", text) and "schedule" not in text:
        return True
    if _contains_phrase(text, "auction") and not any(
        _contains_phrase(text, word) for word in _SCHEDULE_WORDS
    ):
        return True
    return False


def _detect_capabilities(message: str) -> CapabilityFlags:
    text = message.casefold()
    policy = any(_contains_phrase(text, word) for word in _POLICY_WORDS)
    schedule = any(_contains_phrase(text, word) for word in _SCHEDULE_WORDS)
    public_record = any(_contains_phrase(text, word) for word in _PUBLIC_WORDS) or schedule
    auction = any(_contains_phrase(text, word) for word in _AUCTION_WORDS) or (
        _contains_phrase(text, "williamson")
        and any(_contains_phrase(text, token) for token in ("july", "2026", "listing", "listings"))
    )
    # Person/address spreadsheet lookups — but not when the ask is clearly WCAD/public.
    if _looks_like_storage_lookup(message) and not public_record:
        auction = True
    # County schedule/calendar → MCP; keep auction SQL when the ask also needs listings.
    if schedule and not _wants_indexed_auction_rows(text):
        auction = False
        public_record = True
    return CapabilityFlags(policy=policy, auction=auction, public_record=public_record)


def _intent_from_capabilities(caps: CapabilityFlags) -> Intent | None:
    active = sum((caps["policy"], caps["auction"], caps["public_record"]))
    if active >= 2:
        return Intent.COMBINED_RESEARCH
    if caps["policy"]:
        return Intent.KNOWLEDGE_POLICY
    if caps["auction"]:
        return Intent.AUCTION_SEARCH
    if caps["public_record"]:
        return Intent.PUBLIC_PROPERTY_LOOKUP
    return None


def _bounded_name(value: str, *, max_tokens: int) -> str | None:
    tokens = [token for token in value.strip(" .,").split() if token]
    filtered: list[str] = []
    for token in tokens:
        if token.casefold() in _NAME_BOUNDARY:
            break
        filtered.append(token)
        if len(filtered) >= max_tokens:
            break
    if not filtered:
        return None
    joined = " ".join(filtered)
    if joined.casefold() in _TRUSTEE_STOP:
        return None
    if filtered[0].casefold() in _TRUSTEE_STOP:
        return None
    return joined


def _deterministic_entities(message: str) -> dict[str, str | int]:
    entities: dict[str, str | int] = {}
    if address := _ADDRESS.search(message):
        entities["address"] = " ".join(address.group(0).split())
    if trustee := _TRUSTEE.search(message):
        name = _bounded_name(trustee.group(1), max_tokens=4)
        if name is not None:
            entities["trustee"] = name
    if last_first := _LAST_FIRST_NAME.search(message):
        # Spreadsheet trustees often appear as "LAST, FIRST".
        entities.setdefault(
            "trustee",
            f"{last_first.group(1).strip()}, {last_first.group(2).strip()}",
        )
    if mortgagor := _MORTGAGOR.search(message):
        name = _bounded_name(mortgagor.group(1), max_tokens=3)
        if name is not None:
            parts = name.split()
            entities["mortgagor_first_name"] = parts[0]
            if len(parts) > 1:
                entities["mortgagor_last_name"] = parts[-1]
    if zip_code := _ZIP.search(message):
        entities["zip_code"] = zip_code.group(1)
    if year := _YEAR.search(message):
        entities["report_year"] = int(year.group(1) or year.group(2))
    text = message.casefold()
    for month, number in _MONTH_NAMES.items():
        if re.search(rf"\b{month}\b", text):
            entities["report_month"] = number
            break
    if city := _CITY.search(message):
        candidate = " ".join(city.group(1).split())
        if candidate.casefold() not in {"tx", "texas"} and not candidate[:1].isdigit():
            entities["city"] = candidate
    if any(word in text for word in _SCHEDULE_WORDS):
        entities["public_lookup"] = "county_schedule"
    if (
        "trustee" not in entities
        and "address" not in entities
        and "mortgagor_first_name" not in entities
        and _looks_like_storage_lookup(message)
    ):
        remainder = re.sub(
            r"\b(?:provide|details?|info|information|tell me|who is|find|search|"
            r"look\s*up|show|about|for|on|me|the|a|an|please|named|is)\b",
            " ",
            message,
            flags=re.I,
        )
        remainder = re.sub(r"[^A-Za-z,'\s-]", " ", remainder)
        name = _bounded_name(" ".join(remainder.split()), max_tokens=4)
        if name is not None and len(name.split()) >= 2:
            entities["trustee"] = name
    return entities


def _score_address_overlap(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    a = _normalize_text(left)
    b = _normalize_text(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    left_tokens = set(a.split())
    right_tokens = set(b.split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return round(overlap, 4)


def _correlation_score(
    *,
    auction: AuctionResultCard,
    candidate: WcadCandidate,
) -> float:
    scores = [
        _score_address_overlap(auction.property_address, candidate.address),
        1.0
        if auction.zip_code
        and candidate.zip_code
        and auction.zip_code[:5] == candidate.zip_code[:5]
        else 0.0,
    ]
    if auction.city and candidate.address:
        city = auction.city.casefold()
        scores.append(0.4 if city in candidate.address.casefold() else 0.0)
    if candidate.legal_description and auction.property_address:
        scores.append(
            0.35 * _score_address_overlap(auction.property_address, candidate.legal_description)
        )
    if candidate.parcel_or_account_id and auction.fc_id:
        scores.append(
            1.0 if candidate.parcel_or_account_id.casefold() == auction.fc_id.casefold() else 0.0
        )
    if candidate.retrieved_at is not None:
        scores.append(0.05)
    return round(min(1.0, sum(scores) / max(1.0, len(scores))), 4)


def _citation_ids(state: AgentState) -> set[str]:
    return {citation.id for citation in state.get("citations", [])}


def _safe_progress(event: dict[str, object]) -> dict[str, object] | None:
    name = event.get("event")
    if not isinstance(name, str) or name not in _SAFE_STREAM_EVENTS:
        return None
    data = event.get("data")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return None
    return {"event": name, "data": cast(dict[str, object], data)}


class ControlledAgentGraph:
    def __init__(
        self,
        services: GraphServices | None = None,
        *,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        thread_owner_store: ThreadOwnerStore | None = None,
        telemetry: LangfuseTelemetry | None = None,
    ) -> None:
        self._services = services or GraphServices()
        self._thread_owner_store = thread_owner_store
        self._telemetry = telemetry
        graph = StateGraph(AgentState)
        graph.add_node(
            "initialize_run",
            self._observed_node("initialize_run", self._initialize_run),
        )
        graph.add_node("validation", self._observed_node("validation", self._validation))
        graph.add_node("routing", self._observed_node("routing", self._routing))
        graph.add_node("extraction", self._observed_node("extraction", self._extraction))
        graph.add_node(
            "knowledge_rag",
            self._observed_node("knowledge_rag", self._knowledge, kind="retriever"),
        )
        graph.add_node(
            "sql_auction_search",
            self._observed_node("sql_auction_search", self._auction, kind="retriever"),
        )
        graph.add_node(
            "mcp_public_tools",
            self._observed_node("mcp_public_tools", self._public, kind="tool"),
        )
        graph.add_node(
            "unsupported",
            cast(Any, self._observed_node("unsupported", self._unsupported)),
        )
        graph.add_node(
            "evidence_correlation",
            self._observed_node("evidence_correlation", self._correlate),
        )
        graph.add_node(
            "grounding_verification",
            self._observed_node("grounding_verification", self._verify, kind="guardrail"),
        )
        graph.add_node(
            "compliance_disclaimer",
            self._observed_node("compliance_disclaimer", self._compliance, kind="guardrail"),
        )
        graph.add_node(
            "synthesis",
            self._observed_node("synthesis", self._synthesis, kind="generation"),
        )
        graph.add_node(
            "trace_finalization",
            self._observed_node("trace_finalization", self._finalize),
        )
        graph.add_edge(START, "initialize_run")
        graph.add_edge("initialize_run", "validation")
        graph.add_conditional_edges(
            "validation",
            self._after_validation,
            {
                "routing": "routing",
                "unsupported": "unsupported",
            },
        )
        graph.add_edge("routing", "extraction")
        graph.add_conditional_edges(
            "extraction",
            self._dispatch_capabilities,
            [
                "knowledge_rag",
                "sql_auction_search",
                "mcp_public_tools",
                "unsupported",
            ],
        )
        for node in (
            "knowledge_rag",
            "sql_auction_search",
            "mcp_public_tools",
            "unsupported",
        ):
            graph.add_edge(node, "evidence_correlation")
        graph.add_edge("evidence_correlation", "grounding_verification")
        graph.add_edge("grounding_verification", "compliance_disclaimer")
        graph.add_edge("compliance_disclaimer", "synthesis")
        graph.add_edge("synthesis", "trace_finalization")
        graph.add_edge("trace_finalization", END)
        self.compiled = graph.compile(checkpointer=checkpointer)

    def _observed_node(
        self,
        name: str,
        operation: Callable[[AgentState], Awaitable[dict[str, object]]],
        *,
        kind: ObservationType = "span",
    ) -> Any:
        async def run(state: AgentState) -> dict[str, object]:
            started = time.perf_counter()
            outcome = "success"
            observation = (
                self._telemetry.observe(
                    name,
                    kind=kind,
                    metadata={"intent": str(state.get("intent", "unknown"))},
                )
                if self._telemetry
                else None
            )
            try:
                with tracer().start_as_current_span(f"graph.{name}") as span:
                    span.set_attribute("graph.node", name)
                    if observation is None:
                        return await operation(state)
                    with observation:
                        return await operation(state)
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except Exception:
                outcome = "error"
                raise
            finally:
                GRAPH_NODE_DURATION.labels(name, outcome).observe(time.perf_counter() - started)

        return run

    async def _initialize_run(self, state: AgentState) -> dict[str, object]:
        """Reset per-request transients while preserving thread/request identity."""

        return {
            "intent": Intent.UNSUPPORTED_OR_UNSAFE,
            "intent_confidence": 0.0,
            "retrieval_confidence": 0.0,
            "auction_match_confidence": 0.0,
            "property_correlation_confidence": 0.0,
            "grounding_confidence": 0.0,
            "final_answer_confidence": 0.0,
            "capabilities": CapabilityFlags(policy=False, auction=False, public_record=False),
            "entities": {},
            # Reducer channels must use Overwrite so prior-turn merges do not accumulate.
            "evidence": Overwrite([]),
            "citations": Overwrite([]),
            "auction_results": Overwrite([]),
            "wcad_candidates": Overwrite([]),
            "property_summary": None,
            "unavailable_sources": Overwrite([]),
            "errors": Overwrite([]),
            "grounded": False,
            "disclaimer": None,
            "answer": "",
            "answer_citation_ids": [],
            "final_response": None,
            "message": state["message"].strip(),
            "request_id": state["request_id"],
            "correlation_id": state["correlation_id"],
            "trace_id": state["trace_id"],
            "run_id": state["run_id"],
            "thread_id": state["thread_id"],
            "assistant_message_id": state["assistant_message_id"],
            "user_id": state["user_id"],
            "locale": state["locale"],
            "timezone": state["timezone"],
            "retry_budget": state["retry_budget"],
        }

    def _after_validation(self, state: AgentState) -> Literal["routing", "unsupported"]:
        # Route on validation errors only so initialize_run's safe default intent
        # cannot skip classification for valid requests.
        if state.get("errors"):
            return "unsupported"
        return "routing"

    def _dispatch_capabilities(self, state: AgentState) -> list[Send]:
        if state.get("intent") == Intent.UNSUPPORTED_OR_UNSAFE:
            return [Send("unsupported", state)]
        caps = state.get("capabilities") or CapabilityFlags(
            policy=False, auction=False, public_record=False
        )
        sends: list[Send] = []
        if caps["policy"]:
            sends.append(Send("knowledge_rag", state))
        if caps["auction"]:
            sends.append(Send("sql_auction_search", state))
        if caps["public_record"]:
            sends.append(Send("mcp_public_tools", state))
        return sends or [Send("unsupported", state)]

    async def _validation(self, state: AgentState) -> dict[str, object]:
        message = state["message"].strip()
        if not message or len(message) > 4000:
            return {
                "intent": Intent.UNSUPPORTED_OR_UNSAFE,
                "intent_confidence": 1.0,
                "errors": ["invalid_request"],
            }
        if _INJECTION.search(message):
            return {
                "intent": Intent.UNSUPPORTED_OR_UNSAFE,
                "intent_confidence": 1.0,
                "errors": ["prompt_injection"],
            }
        return {"message": message, "errors": []}

    async def _routing(self, state: AgentState) -> dict[str, object]:
        caps = _detect_capabilities(state["message"])
        intent = _intent_from_capabilities(caps)
        confidence = 0.97
        if intent is None:
            classifier = self._services.classifier
            if classifier is None or state["retry_budget"]["classifier"] <= 0:
                if _looks_like_storage_lookup(state["message"]):
                    intent = Intent.AUCTION_SEARCH
                    caps = CapabilityFlags(policy=False, auction=True, public_record=False)
                    confidence = 0.7
                else:
                    return {
                        "intent": Intent.UNSUPPORTED_OR_UNSAFE,
                        "intent_confidence": 0.4,
                        "capabilities": caps,
                    }
            else:
                try:
                    intent = await asyncio.wait_for(
                        classifier.classify(state["message"]),
                        timeout=self._services.classifier_timeout_seconds,
                    )
                    confidence = 0.65
                    if intent == Intent.KNOWLEDGE_POLICY:
                        caps = CapabilityFlags(policy=True, auction=False, public_record=False)
                    elif intent == Intent.AUCTION_SEARCH:
                        caps = CapabilityFlags(policy=False, auction=True, public_record=False)
                    elif intent == Intent.PUBLIC_PROPERTY_LOOKUP:
                        caps = CapabilityFlags(policy=False, auction=False, public_record=True)
                    elif intent == Intent.COMBINED_RESEARCH:
                        caps = CapabilityFlags(policy=True, auction=True, public_record=True)
                    else:
                        caps = CapabilityFlags(policy=False, auction=False, public_record=False)
                except (TimeoutError, OSError, ValueError):
                    intent = Intent.UNSUPPORTED_OR_UNSAFE
                    confidence = 0.4
        # Prefer spreadsheet lookup over unsupported for names/addresses.
        if intent == Intent.UNSUPPORTED_OR_UNSAFE and _looks_like_storage_lookup(state["message"]):
            intent = Intent.AUCTION_SEARCH
            caps = CapabilityFlags(policy=False, auction=True, public_record=False)
            confidence = max(confidence, 0.75)
        _emit(
            "route.selected",
            {
                "intent": intent.value,
                "confidence": confidence,
                "capabilities": dict(caps),
            },
        )
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "capabilities": caps,
        }

    async def _extraction(self, state: AgentState) -> dict[str, object]:
        entities = _deterministic_entities(state["message"])
        ambiguous = (
            "mortgagor" in state["message"].casefold() and "mortgagor_first_name" not in entities
        ) or ("trustee" in state["message"].casefold() and "trustee" not in entities)
        needs_llm = ambiguous or (
            state.get("intent") == Intent.AUCTION_SEARCH
            and "trustee" not in entities
            and "address" not in entities
            and "mortgagor_first_name" not in entities
            and _looks_like_storage_lookup(state["message"])
        )
        extractor = self._services.entity_extractor
        if needs_llm and extractor is not None:
            try:
                extracted = await asyncio.wait_for(
                    extractor.extract(state["message"]),
                    timeout=self._services.extractor_timeout_seconds,
                )
            except (TimeoutError, OSError, ValueError):
                extracted = {}
            for key, value in extracted.items():
                if key in {
                    "mortgagor_first_name",
                    "mortgagor_last_name",
                    "trustee",
                    "address",
                    "city",
                    "zip_code",
                    "report_year",
                    "report_month",
                }:
                    if isinstance(value, str) and value.strip():
                        if key in {"report_year", "report_month"}:
                            try:
                                entities[key] = int(value)
                            except ValueError:
                                continue
                        elif key == "trustee":
                            name = _bounded_name(value, max_tokens=4)
                            if name is not None:
                                entities[key] = name
                            elif _LAST_FIRST_NAME.search(value):
                                entities[key] = " ".join(value.split())[:120]
                            else:
                                entities[key] = " ".join(value.split())[:120]
                        elif key.startswith("mortgagor_"):
                            name = _bounded_name(value, max_tokens=1)
                            if name is not None:
                                entities[key] = name
                        else:
                            entities[key] = " ".join(value.split())[:300]
                    elif isinstance(value, int) and key in {"report_year", "report_month"}:
                        entities[key] = value
        return {"entities": entities}

    async def _bounded(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        attempts: int,
    ) -> Any:
        for attempt in range(max(1, attempts)):
            try:
                return await asyncio.wait_for(
                    operation(),
                    timeout=self._services.node_timeout_seconds,
                )
            except (TimeoutError, OSError):
                if attempt + 1 >= max(1, attempts):
                    raise
                await asyncio.sleep(0.05 * (2**attempt))
        raise RuntimeError("unreachable")

    @staticmethod
    def _compact_evidence(items: list[dict[str, object]]) -> list[dict[str, object]]:
        compacted: list[dict[str, object]] = []
        for index, item in enumerate(items, start=1):
            content = str(item.get("excerpt") or item.get("content") or item.get("summary") or "")
            compacted.append(
                {
                    "evidence_id": str(item.get("evidence_id") or item.get("chunk_id") or index),
                    "title": str(item.get("title") or "")[:200],
                    "excerpt": content[:_EVIDENCE_EXCERPT_CHARS],
                    "score": _as_float(item.get("score")),
                    "citation_id": item.get("citation_id"),
                    "source_metadata": {
                        key: item.get(key)
                        for key in ("document_id", "chunk_id", "page_number")
                        if key in item
                    },
                }
            )
        return compacted

    async def _knowledge(self, state: AgentState) -> dict[str, object]:
        _emit("retrieval.started", {"source": "approved_policy"})
        try:
            evidence, citations = await self._bounded(
                lambda: self._services.knowledge_search(state["message"]),
                attempts=state["retry_budget"]["retrieval"],
            )
            RETRIEVAL_REQUESTS.labels("success").inc()
            RETRIEVAL_RESULTS.observe(len(evidence))
            citation_id = citations[0].id if citations else None
            stamped = [
                {**item, "citation_id": item.get("citation_id") or citation_id} for item in evidence
            ]
            compact = self._compact_evidence(stamped)
            score = max((_as_float(item.get("score")) for item in compact), default=0.0)
            _emit(
                "retrieval.completed",
                {"source": "approved_policy", "result_count": len(compact)},
            )
            return {
                "evidence": compact,
                "citations": citations,
                "retrieval_confidence": score,
            }
        except (TimeoutError, OSError):
            RETRIEVAL_REQUESTS.labels("error").inc()
            _emit("retrieval.completed", {"source": "approved_policy", "result_count": 0})
            return {
                "evidence": [],
                "citations": [],
                "unavailable_sources": [
                    UnavailableSource(
                        source="approved_policy", reason="unavailable", retryable=True
                    )
                ],
                "retrieval_confidence": 0.0,
            }

    async def _auction(self, state: AgentState) -> dict[str, object]:
        _emit("retrieval.started", {"source": "auction_index"})
        try:
            cards, citations = await self._bounded(
                lambda: self._services.auction_search(state.get("entities", {})),
                attempts=state["retry_budget"]["retrieval"],
            )
            AUCTION_SEARCHES.labels("success").inc()
            confidence = max((card.match_confidence for card in cards), default=0.0)
            _emit(
                "retrieval.completed",
                {"source": "auction_index", "result_count": len(cards)},
            )
            return {
                "auction_results": cards,
                "citations": citations,
                "auction_match_confidence": confidence,
            }
        except (TimeoutError, OSError, ValueError, RuntimeError):
            AUCTION_SEARCHES.labels("error").inc()
            _emit("retrieval.completed", {"source": "auction_index", "result_count": 0})
            return {
                "auction_results": [],
                "citations": [],
                "unavailable_sources": [
                    UnavailableSource(source="auction_index", reason="unavailable", retryable=True)
                ],
                "auction_match_confidence": 0.0,
            }

    async def _public(self, state: AgentState) -> dict[str, object]:
        _emit("tool.started", {"tool": "mcp_public_tools"})
        try:
            summary, candidates, citations, unavailable = await self._bounded(
                lambda: self._services.public_search(state.get("entities", {})),
                attempts=state["retry_budget"]["tools"],
            )
            confidence = 0.0
            if summary is not None and summary.match_confidence is not None:
                confidence = summary.match_confidence
            elif candidates:
                confidence = max(item.confidence for item in candidates)
            _emit(
                "tool.completed",
                {"tool": "mcp_public_tools", "candidate_count": len(candidates)},
            )
            return {
                "property_summary": summary,
                "wcad_candidates": candidates,
                "citations": citations,
                "unavailable_sources": unavailable,
                "property_correlation_confidence": confidence,
            }
        except (TimeoutError, OSError):
            _emit("tool.completed", {"tool": "mcp_public_tools", "candidate_count": 0})
            return {
                "property_summary": None,
                "wcad_candidates": [],
                "citations": [],
                "unavailable_sources": [
                    UnavailableSource(source="public_records", reason="unavailable", retryable=True)
                ],
                "property_correlation_confidence": 0.0,
            }

    async def _unsupported(self, _state: AgentState) -> dict[str, object]:
        return {"evidence": [], "citations": []}

    async def _correlate(self, state: AgentState) -> dict[str, object]:
        citations = list(
            {citation.id: citation for citation in state.get("citations", [])}.values()
        )
        candidates = list(state.get("wcad_candidates", []))
        auctions = list(state.get("auction_results", []))
        summary = state.get("property_summary")
        correlation = float(state.get("property_correlation_confidence") or 0.0)

        if auctions and candidates:
            ranked: list[tuple[float, WcadCandidate]] = []
            for candidate in candidates:
                best = max(
                    _correlation_score(auction=card, candidate=candidate) for card in auctions
                )
                ranked.append((best, candidate.model_copy(update={"confidence": best})))
            ranked.sort(key=lambda item: item[0], reverse=True)
            candidates = [item[1] for item in ranked]
            correlation = ranked[0][0] if ranked else 0.0
            requires_selection = len(candidates) > 1 and (
                correlation < 0.8 or (len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < 0.1)
            )
            selected = candidates[0] if candidates else None
            if selected is not None:
                summary = PropertySummary(
                    property_id=selected.property_id,
                    address=selected.address,
                    owner_name=selected.owner_name,
                    legal_description=selected.legal_description,
                    market_value=selected.market_value,
                    source_citation_ids=list(selected.source_citation_ids),
                    limitations=list(summary.limitations) if summary is not None else [],
                    candidates=candidates,
                    selected_property_id=None if requires_selection else selected.property_id,
                    match_confidence=correlation,
                    requires_user_selection=requires_selection,
                )
        elif candidates and summary is None:
            requires_selection = len(candidates) > 1 and (
                candidates[0].confidence < 0.8
                or (
                    len(candidates) > 1
                    and abs(candidates[0].confidence - candidates[1].confidence) < 0.1
                )
            )
            top = candidates[0]
            summary = PropertySummary(
                property_id=top.property_id,
                address=top.address,
                owner_name=top.owner_name,
                legal_description=top.legal_description,
                market_value=top.market_value,
                source_citation_ids=list(top.source_citation_ids),
                limitations=[],
                candidates=candidates,
                selected_property_id=None if requires_selection else top.property_id,
                match_confidence=top.confidence,
                requires_user_selection=requires_selection,
            )
            correlation = top.confidence
        elif summary is not None and candidates and not summary.candidates:
            summary = summary.model_copy(update={"candidates": candidates})

        return {
            "citations": Overwrite(citations),
            "wcad_candidates": Overwrite(candidates),
            "property_summary": summary,
            "property_correlation_confidence": correlation,
        }

    async def _verify(self, state: AgentState) -> dict[str, object]:
        allowed = _citation_ids(state)
        errors = list(state.get("errors", []))
        evidence = list(state.get("evidence", []))
        auctions = list(state.get("auction_results", []))
        summary = state.get("property_summary")
        candidates = list(state.get("wcad_candidates", []))

        def grounded_ids(ids: list[str]) -> bool:
            return bool(ids) and set(ids).issubset(allowed)

        valid_auctions: list[AuctionResultCard] = []
        for card in auctions:
            if grounded_ids(card.source_citation_ids):
                valid_auctions.append(card)
            else:
                errors.append("ungrounded_auction_removed")

        valid_evidence: list[dict[str, object]] = []
        for item in evidence:
            citation_id = item.get("citation_id")
            if isinstance(citation_id, str):
                if citation_id in allowed:
                    valid_evidence.append(item)
                else:
                    errors.append("ungrounded_policy_removed")
            elif allowed:
                # Compact RAG evidence may omit citation_id; require at least one policy citation.
                valid_evidence.append(item)
            else:
                errors.append("ungrounded_policy_removed")

        if evidence and not allowed:
            valid_evidence = []
            errors.append("ungrounded_policy_removed")

        valid_summary = summary
        if summary is not None:
            if not grounded_ids(summary.source_citation_ids):
                valid_summary = None
                errors.append("ungrounded_public_record_removed")
            else:
                kept = [
                    candidate
                    for candidate in candidates
                    if grounded_ids(candidate.source_citation_ids)
                ]
                candidates = kept
                if valid_summary is not None:
                    valid_summary = valid_summary.model_copy(update={"candidates": kept})

        factual = bool(valid_evidence or valid_auctions or valid_summary)
        grounded = (not factual) or bool(allowed)
        if factual and not allowed:
            grounded = False
            valid_evidence, valid_auctions, valid_summary, candidates = [], [], None, []
            errors.append("ungrounded_results_removed")

        confidence = 1.0 if grounded else 0.0
        if factual and grounded:
            confidence = 0.9
        return {
            "grounded": grounded,
            "grounding_confidence": confidence,
            "auction_results": Overwrite(valid_auctions),
            "property_summary": valid_summary,
            "wcad_candidates": Overwrite(candidates),
            "evidence": Overwrite(valid_evidence),
            "errors": Overwrite(errors),
        }

    async def _compliance(self, state: AgentState) -> dict[str, object]:
        public_intents = {
            Intent.AUCTION_SEARCH,
            Intent.PUBLIC_PROPERTY_LOOKUP,
            Intent.COMBINED_RESEARCH,
        }
        return {"disclaimer": DISCLAIMER if state.get("intent") in public_intents else None}

    async def _synthesis(self, state: AgentState) -> dict[str, object]:
        intent = state.get("intent")
        if intent == Intent.UNSUPPORTED_OR_UNSAFE:
            answer = "I can only help with approved auction, property-record, and policy questions."
            citation_ids: list[str] = []
            confidence = 1.0
        elif (
            self._services.synthesizer is not None
            and state.get("grounded")
            and (
                state.get("evidence")
                or state.get("auction_results")
                or state.get("property_summary")
            )
        ):
            try:
                result = await asyncio.wait_for(
                    self._services.synthesizer.synthesize(state),
                    timeout=self._services.node_timeout_seconds,
                )
                allowed = _citation_ids(state)
                if isinstance(result, str):
                    answer = result
                    citation_ids = sorted(allowed) if allowed else []
                    confidence = 0.7
                else:
                    answer = str(getattr(result, "answer", ""))
                    raw_ids = getattr(result, "citation_ids", ())
                    citation_ids = [
                        item for item in list(raw_ids) if isinstance(item, str) and item in allowed
                    ]
                    if (
                        allowed
                        and not citation_ids
                        and (
                            state.get("evidence")
                            or state.get("auction_results")
                            or state.get("property_summary")
                        )
                    ):
                        raise ValueError("Factual synthesis must reference known citations.")
                    confidence = _as_float(getattr(result, "confidence", 0.7), default=0.7)
            except (TimeoutError, OSError, ValueError):
                answer, citation_ids, confidence = self._deterministic_synthesis(state)
        else:
            answer, citation_ids, confidence = self._deterministic_synthesis(state)

        _emit("answer.delta", {"sequence": 1, "text": answer})
        return {
            "answer": answer,
            "answer_citation_ids": citation_ids,
            "final_answer_confidence": confidence,
        }

    def _deterministic_synthesis(self, state: AgentState) -> tuple[str, list[str], float]:
        citation_ids = sorted(_citation_ids(state))
        intent = state.get("intent")
        unavailable = list(state.get("unavailable_sources") or [])
        if any("schedule_not_found" in item.reason for item in unavailable):
            return (
                "I checked the official Williamson County trustee-sale pages and did not find "
                "a schedule matching that month. The county site may not have posted it yet—"
                "please verify on the official pages.",
                citation_ids,
                0.55,
            )
        if state.get("auction_results"):
            count = len(state["auction_results"])
            return (
                f"I found {count} matching auction record(s).",
                citation_ids,
                float(state.get("auction_match_confidence") or 0.8),
            )
        summary = state.get("property_summary")
        if summary is not None:
            if summary.property_id.startswith("county-schedule"):
                lines = [item for item in summary.limitations if item.strip()]
                answer = (
                    lines[0]
                    if lines
                    else "I found official county trustee-sale source materials."
                )
                return (
                    answer,
                    citation_ids,
                    float(summary.match_confidence or 0.75),
                )
            if summary.requires_user_selection:
                return (
                    "I found multiple WCAD property candidates. Please select the correct record.",
                    citation_ids,
                    float(summary.match_confidence or 0.5),
                )
            return (
                "I found a candidate property record in the approved public sources.",
                citation_ids,
                float(
                    summary.match_confidence or state.get("property_correlation_confidence") or 0.75
                ),
            )
        if state.get("evidence"):
            excerpt = str(
                state["evidence"][0].get("excerpt")
                or state["evidence"][0].get("summary")
                or state["evidence"][0].get("content")
                or ""
            )
            return excerpt or "I found approved policy guidance.", citation_ids, 0.8
        if intent in {Intent.AUCTION_SEARCH, Intent.COMBINED_RESEARCH}:
            return NO_MATCH, citation_ids, 0.55
        return (
            "I could not find sufficient supporting information in the approved sources.",
            citation_ids,
            0.4,
        )

    def _build_cta(self, state: AgentState) -> CTA | None:
        auctions = state.get("auction_results") or []
        if not auctions:
            return None
        entities = state.get("entities", {})
        count = len(auctions)
        label = f"View {count} matching auction{'s' if count != 1 else ''}"
        year = entities.get("report_year")
        month = entities.get("report_month")
        return CTA(
            label=label,
            filters=CTAFilters(
                trustee=str(entities["trustee"]) if "trustee" in entities else None,
                mortgagor_first_name=(
                    str(entities["mortgagor_first_name"])
                    if "mortgagor_first_name" in entities
                    else None
                ),
                mortgagor_last_name=(
                    str(entities["mortgagor_last_name"])
                    if "mortgagor_last_name" in entities
                    else None
                ),
                address=str(entities["address"]) if "address" in entities else None,
                city=str(entities["city"]) if "city" in entities else None,
                zip_code=str(entities["zip_code"]) if "zip_code" in entities else None,
                year=year if isinstance(year, int) else None,
                month=month if isinstance(month, int) else None,
            ),
        )

    async def _finalize(self, state: AgentState) -> dict[str, object]:
        unavailable = list(state.get("unavailable_sources", []))
        has_payload = bool(
            state.get("citations")
            or state.get("auction_results")
            or state.get("property_summary")
            or state.get("evidence")
        )
        status: Literal["completed", "partial", "failed"]
        if unavailable and not has_payload:
            status = "failed"
        elif unavailable:
            status = "partial"
        else:
            status = "completed"

        response = FinalResponse(
            request_id=state["request_id"],
            correlation_id=state["correlation_id"],
            trace_id=state["trace_id"],
            run_id=UUID(state["run_id"]),
            thread_id=UUID(state["thread_id"]),
            assistant_message_id=UUID(state["assistant_message_id"]),
            status=status,
            intent=state["intent"],
            answer=state.get("answer") or "",
            citations=state.get("citations", []),
            auction_results=state.get("auction_results", []),
            property_summary=state.get("property_summary"),
            cta=self._build_cta(state),
            disclaimer=state.get("disclaimer"),
            unavailable_sources=unavailable,
            confidence=state.get("final_answer_confidence"),
        )
        _emit(
            "answer.completed",
            {"response": response.model_dump(mode="json")},
        )
        return {"final_response": response}

    def _config(self, state: AgentState) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": f"{state['user_id']}:{state['thread_id']}",
                "checkpoint_ns": "get-auction-list",
            }
        }

    async def run(self, state: AgentState) -> FinalResponse:
        if self._thread_owner_store is not None:
            await self._thread_owner_store.ensure_owner(
                user_id=state["user_id"],
                thread_id=state["thread_id"],
            )
        config = self._config(state)
        if self._telemetry is None:
            result = await self.compiled.ainvoke(state, config=config)
        else:
            with self._telemetry.observe(
                "graph.run",
                kind="agent",
                trace_id=UUID(state["run_id"]).hex,
                metadata={
                    "run_id": state["run_id"],
                    "thread_id": state["thread_id"],
                    "locale": state["locale"],
                },
            ):
                result = await self.compiled.ainvoke(state, config=config)
            intent = cast(Intent, result["intent"])
            self._telemetry.score(
                trace_id=UUID(state["run_id"]).hex,
                name="router_confidence",
                value=float(result.get("intent_confidence", 0)),
                comment=intent.value,
            )
            self._telemetry.score(
                trace_id=UUID(state["run_id"]).hex,
                name="final_answer_confidence",
                value=float(result.get("final_answer_confidence", 0)),
            )
            self._telemetry.score(
                trace_id=UUID(state["run_id"]).hex,
                name="grounded",
                value=1.0 if result.get("grounded") else 0.0,
            )
        return cast(FinalResponse, result["final_response"])

    async def astream(self, state: AgentState) -> AsyncIterator[dict[str, object]]:
        """Yield safe public progress events from LangGraph ``astream``."""

        if self._thread_owner_store is not None:
            await self._thread_owner_store.ensure_owner(
                user_id=state["user_id"],
                thread_id=state["thread_id"],
            )
        yield {
            "event": "run.started",
            "data": {
                "status": "accepted",
                "assistant_message_id": state["assistant_message_id"],
            },
        }
        config = self._config(state)
        saw_answer_completed = False
        final: FinalResponse | None = None
        try:
            async for item in self.compiled.astream(
                state,
                config=config,
                stream_mode=["custom", "values"],
            ):
                mode: str
                payload: object
                if isinstance(item, tuple) and len(item) == 2:
                    mode, payload = item
                else:
                    mode, payload = "values", item
                if mode == "custom" and isinstance(payload, dict):
                    safe = _safe_progress(payload)
                    if safe is not None:
                        if safe["event"] == "answer.completed":
                            saw_answer_completed = True
                        yield safe
                elif mode == "values" and isinstance(payload, dict):
                    response = payload.get("final_response")
                    if isinstance(response, FinalResponse):
                        final = response
            if final is not None and not saw_answer_completed:
                yield {
                    "event": "answer.completed",
                    "data": {"response": final.model_dump(mode="json")},
                }
        except asyncio.CancelledError:
            raise
        except Exception:
            yield {
                "event": "run.failed",
                "data": {
                    "error": {
                        "code": "CHAT_RUN_FAILED",
                        "message": "The chat run could not be completed.",
                        "retryable": True,
                    }
                },
            }
