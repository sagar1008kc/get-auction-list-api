"""Typed adapters between graph contracts and production data services."""

from collections.abc import Callable, Mapping
from uuid import UUID

from get_auction_list_api.auth import Principal
from get_auction_list_api.domain import AuctionSearchFilters
from get_auction_list_api.graph.workflow import (
    Classifier,
    EntityExtractor,
    GraphServices,
    Synthesizer,
)
from get_auction_list_api.llm.embeddings import EmbeddingProvider
from get_auction_list_api.public_records.models import (
    PublicRecordToolError,
    ToolErrorCategory,
    TrusteeSaleSource,
    WcadProperty,
    WcadPropertyDetails,
)
from get_auction_list_api.public_records.service import PublicRecordsService
from get_auction_list_api.rag.models import EvidenceCitation
from get_auction_list_api.rag.retrieval import HybridRetriever
from get_auction_list_api.schemas import (
    AuctionResultCard,
    Citation,
    PropertySummary,
    UnavailableSource,
    WcadCandidate,
)
from get_auction_list_api.tools.auction_search import (
    AuctionSearchResponse,
    AuctionSearchService,
)
from get_auction_list_api.tools.auction_search import (
    Citation as AuctionCitation,
)

PrincipalResolver = Callable[[], Principal]
TraceIdResolver = Callable[[], str]

_ENTITY_TO_FILTER = {
    "mortgagor_first_name": "mortgagor_first",
    "mortgagor_last_name": "mortgagor_last",
}


def _rag_citation(value: EvidenceCitation) -> Citation:
    return Citation(
        id=value.id,
        source_kind="policy_document",
        title=value.title,
        official_source=True,
        url=value.url,
        document_id=value.document_id,
        document_version_id=value.document_version_id,
        page_number=value.page_number,
        sheet_name=value.sheet_name,
        row_start=value.row_start,
        row_end=value.row_end,
        chunk_id=value.chunk_id,
        retrieved_at=value.retrieved_at,
        quote=value.quote,
    )


def _optional_uuid(value: str | None) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _auction_citation(value: AuctionCitation) -> Citation:
    return Citation(
        id=value.id,
        source_kind=value.source_kind,
        title=value.title,
        official_source=value.official_source,
        url=value.url,
        document_version_id=_optional_uuid(value.document_version_id),
        page_number=value.page_number,
        sheet_name=value.sheet_name,
        row_start=value.row_start,
        row_end=value.row_end,
        retrieved_at=value.retrieved_at,
    )


def _auction_cards(response: AuctionSearchResponse) -> list[AuctionResultCard]:
    cards: list[AuctionResultCard] = []
    for item in response.results:
        record = item.record
        cards.append(
            AuctionResultCard(
                record_key=record.stable_key or str(record.id),
                fc_id=record.rid,
                match_level=item.match_level.value,
                match_confidence=item.confidence,
                auction_date=record.sale_date.isoformat() if record.sale_date else None,
                property_address=record.property_address,
                city=record.city,
                zip_code=record.zip_code,
                trustees=[record.trustee_name] if record.trustee_name else [],
                source_citation_ids=[item.citation_id],
                limitations=list(item.limitations),
                report_year=_coord_int(record.source_coordinates.get("report_year")),
                report_month=_coord_int(record.source_coordinates.get("report_month")),
            )
        )
    return cards


def _coord_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


async def _county_schedule_search(
    public_service: PublicRecordsService,
    *,
    entities: Mapping[str, str | int],
    trace_id: str,
) -> tuple[
    PropertySummary | None,
    list[WcadCandidate],
    list[Citation],
    list[UnavailableSource],
]:
    year = entities.get("report_year")
    month = entities.get("report_month")
    year_value = year if isinstance(year, int) else None
    month_value = month if isinstance(month, int) else None
    try:
        discovered = await public_service.discover_trustee_sale_sources(
            trace_id=trace_id,
            year=year_value,
            month=month_value,
        )
    except PublicRecordToolError as error:
        return (
            None,
            [],
            [],
            [
                UnavailableSource(
                    source="public_records",
                    reason=error.category.value,
                    retryable=error.retryable
                    or error.category
                    in {
                        ToolErrorCategory.TIMEOUT,
                        ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                    },
                )
            ],
        )

    if not discovered.items:
        period = (
            f"{year_value}-{month_value:02d}"
            if year_value is not None and month_value is not None
            else "the requested period"
        )
        # Source was reachable; month simply has no matching posted materials.
        calendar_url = str(discovered.metadata.source_url)
        citation = Citation(
            id="county-schedule-index",
            source_kind="public_record",
            title="Williamson County trustee sale calendar",
            official_source=True,
            url=calendar_url,
            retrieved_at=discovered.metadata.retrieved_at,
            quote=f"No matching trustee-sale schedule materials for {period}.",
        )
        summary = PropertySummary(
            property_id="county-schedule",
            address=f"Williamson County trustee sale schedule ({period})",
            source_citation_ids=[citation.id],
            limitations=[
                f"No trustee-sale schedule materials matched {period} on the official "
                "Williamson County calendar. The county may not have posted details yet—"
                f"verify at {calendar_url}.",
                *discovered.warnings,
            ],
            candidates=[],
            selected_property_id="county-schedule",
            match_confidence=0.55,
            requires_user_selection=False,
        )
        return summary, [], [citation], []

    citations: list[Citation] = []
    lines: list[str] = []
    for index, item in enumerate(discovered.items, start=1):
        source = TrusteeSaleSource.model_validate(item)
        citation_id = f"county-schedule-{index}"
        citations.append(
            Citation(
                id=citation_id,
                source_kind="public_record",
                title=source.title,
                official_source=True,
                url=str(source.url),
                retrieved_at=discovered.metadata.retrieved_at,
                quote=source.title[:500],
            )
        )
        lines.append(f"{source.title} ({source.url})")

    period_label = (
        f" for {year_value}-{month_value:02d}"
        if year_value is not None and month_value is not None
        else ""
    )
    summary_text = (
        f"Official Williamson County trustee-sale sources{period_label}: "
        + "; ".join(lines)
        + ". Always verify dates on the linked official pages."
    )
    summary = PropertySummary(
        property_id="county-schedule",
        address=f"Williamson County trustee sale schedule{period_label}",
        source_citation_ids=[item.id for item in citations],
        limitations=[summary_text, *discovered.warnings],
        candidates=[],
        selected_property_id="county-schedule",
        match_confidence=0.85,
        requires_user_selection=False,
    )
    return summary, [], citations, []


def _auction_filter_payload(entities: Mapping[str, str | int]) -> dict[str, object]:
    payload: dict[str, object] = {}
    supported = AuctionSearchFilters.model_fields
    for key, value in entities.items():
        mapped = _ENTITY_TO_FILTER.get(key, key)
        if mapped in supported:
            payload[mapped] = value
    return payload


def build_graph_services(
    *,
    embeddings: EmbeddingProvider | None,
    retriever: HybridRetriever | None,
    auction_service: AuctionSearchService | None,
    public_service: PublicRecordsService,
    principal_resolver: PrincipalResolver,
    trace_id_resolver: TraceIdResolver,
    classifier: Classifier | None = None,
    synthesizer: Synthesizer | None = None,
    entity_extractor: EntityExtractor | None = None,
    classifier_timeout_seconds: float = 1.5,
    node_timeout_seconds: float = 8,
) -> GraphServices:
    """Build graph callables while keeping schema translation out of workflow nodes."""

    async def knowledge_search(
        query: str,
    ) -> tuple[list[dict[str, object]], list[Citation]]:
        # Authenticated chat callers may retrieve approved policy docs; JWT is enforced at
        # the /v1/chat* boundary rather than a fine-grained document:read gate.
        if embeddings is None or retriever is None:
            raise OSError("Knowledge retrieval is not configured.")
        batch = await embeddings.embed((query,))
        context = await retriever.retrieve(query=query, embedding=batch.vectors[0])
        if context.no_answer:
            return [], []
        citations = [_rag_citation(value) for value in context.citations]
        citation_by_chunk = {
            str(item.chunk_id): item.id for item in context.citations if item.chunk_id is not None
        }
        evidence: list[dict[str, object]] = [
            {
                "chunk_id": str(item.chunk_id),
                "title": item.title,
                "content": item.content,
                "score": item.score,
                "citation_id": citation_by_chunk.get(str(item.chunk_id))
                or (citations[0].id if citations else None),
            }
            for item in context.evidence
        ]
        return evidence, citations

    async def auction_search(
        entities: Mapping[str, str | int],
    ) -> tuple[list[AuctionResultCard], list[Citation]]:
        if auction_service is None:
            raise OSError("Auction search is not configured.")
        filters = AuctionSearchFilters.model_validate(_auction_filter_payload(entities))
        response = await auction_service.search(principal_resolver(), filters)
        return _auction_cards(response), [_auction_citation(value) for value in response.citations]

    async def public_search(
        entities: Mapping[str, str | int],
    ) -> tuple[
        PropertySummary | None,
        list[WcadCandidate],
        list[Citation],
        list[UnavailableSource],
    ]:
        if entities.get("public_lookup") == "county_schedule":
            return await _county_schedule_search(
                public_service,
                entities=entities,
                trace_id=trace_id_resolver(),
            )
        address = entities.get("address")
        if not isinstance(address, str) or not address:
            return None, [], [], []
        try:
            search = await public_service.search_property(
                address=address,
                trace_id=trace_id_resolver(),
                limit=5,
            )
            if not search.items:
                return None, [], [], []
            properties = [WcadProperty.model_validate(item) for item in search.items]
        except PublicRecordToolError as error:
            return (
                None,
                [],
                [],
                [
                    UnavailableSource(
                        source="public_records",
                        reason=error.category.value,
                        retryable=error.retryable,
                    )
                ],
            )

        citations: list[Citation] = []
        candidates: list[WcadCandidate] = []
        for index, property_row in enumerate(properties, start=1):
            citation_id = f"wcad-{index}"
            citations.append(
                Citation(
                    id=citation_id,
                    source_kind="property_record",
                    title="Williamson Central Appraisal District property record",
                    official_source=True,
                    url=str(property_row.detail_url),
                    retrieved_at=search.metadata.retrieved_at,
                )
            )
            confidence = max(0.35, 1.0 - (0.1 * (index - 1)))
            candidates.append(
                WcadCandidate(
                    property_id=property_row.property_id,
                    address=property_row.address,
                    owner_name=property_row.owner_name,
                    confidence=confidence,
                    source_citation_ids=[citation_id],
                    retrieved_at=search.metadata.retrieved_at,
                )
            )

        requires_selection = len(candidates) > 1 and (
            candidates[0].confidence < 0.8
            or abs(candidates[0].confidence - candidates[1].confidence) < 0.15
        )
        selected = candidates[0]
        details = None
        if not requires_selection:
            try:
                details_result = await public_service.get_property_details(
                    property_id=selected.property_id,
                    trace_id=trace_id_resolver(),
                )
                details = (
                    WcadPropertyDetails.model_validate(details_result.items[0])
                    if details_result.items
                    else WcadPropertyDetails(**properties[0].model_dump())
                )
                citations[0] = citations[0].model_copy(
                    update={
                        "url": str(details.detail_url),
                        "retrieved_at": details_result.metadata.retrieved_at,
                    }
                )
                selected = selected.model_copy(
                    update={
                        "address": details.address,
                        "owner_name": details.owner_name,
                        "legal_description": details.legal_description,
                        "market_value": details.market_value,
                        "retrieved_at": details_result.metadata.retrieved_at,
                    }
                )
                candidates[0] = selected
                limitations = list(search.warnings + details_result.warnings)
            except PublicRecordToolError as error:
                return (
                    None,
                    candidates,
                    citations,
                    [
                        UnavailableSource(
                            source="wcad_details",
                            reason=error.category.value,
                            retryable=error.retryable,
                        )
                    ],
                )
        else:
            limitations = [
                *search.warnings,
                "Multiple WCAD candidates require user selection before details are fetched.",
            ]

        summary = PropertySummary(
            property_id=selected.property_id,
            address=selected.address,
            owner_name=selected.owner_name,
            legal_description=selected.legal_description,
            market_value=selected.market_value,
            source_citation_ids=list(selected.source_citation_ids),
            limitations=limitations,
            candidates=candidates,
            selected_property_id=None if requires_selection else selected.property_id,
            match_confidence=selected.confidence,
            requires_user_selection=requires_selection,
        )
        return summary, candidates, citations, []

    return GraphServices(
        classifier=classifier,
        synthesizer=synthesizer,
        entity_extractor=entity_extractor,
        knowledge_search=knowledge_search,
        auction_search=auction_search,
        public_search=public_search,
        classifier_timeout_seconds=classifier_timeout_seconds,
        node_timeout_seconds=node_timeout_seconds,
    )
