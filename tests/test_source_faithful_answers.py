"""Source-faithful answering: schedule→MCP, empty index, policy citations, Storage ingest."""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import openpyxl
import pytest

from get_auction_list_api.graph.adapters import build_graph_services
from get_auction_list_api.graph.state import AgentState
from get_auction_list_api.graph.workflow import ControlledAgentGraph, GraphServices
from get_auction_list_api.ingestion.auction_rows import build_publishable_rows
from get_auction_list_api.ingestion.path_meta import parse_auction_storage_path
from get_auction_list_api.ingestion.storage import parse_supabase_uri
from get_auction_list_api.parsers.xlsx import XlsxParser
from get_auction_list_api.public_records.models import (
    PublicRecordToolError,
    ToolErrorCategory,
    ToolMetadata,
    ToolResult,
    TrusteeSaleSource,
)
from get_auction_list_api.public_records.service import PublicRecordsService
from get_auction_list_api.schemas import (
    AuctionResultCard,
    Citation,
    Intent,
    PropertySummary,
    UnavailableSource,
    WcadCandidate,
)


def _state(message: str) -> AgentState:
    return AgentState(
        request_id=str(uuid4()),
        correlation_id=str(uuid4()),
        trace_id=uuid4().hex,
        run_id=str(uuid4()),
        thread_id=str(uuid4()),
        assistant_message_id=str(uuid4()),
        user_id=str(uuid4()),
        message=message,
        locale="en-US",
        timezone="UTC",
        retry_budget={"classifier": 1, "retrieval": 1, "tools": 2},
    )


@pytest.mark.asyncio
async def test_schedule_question_routes_to_mcp_not_auction_sql() -> None:
    calls = {"auction": 0, "public": 0}

    async def auction(
        _entities: Mapping[str, str | int],
    ) -> tuple[list[AuctionResultCard], list[Citation]]:
        calls["auction"] += 1
        return [], []

    async def public(
        entities: Mapping[str, str | int],
    ) -> tuple[
        PropertySummary | None,
        list[WcadCandidate],
        list[Citation],
        list[UnavailableSource],
    ]:
        calls["public"] += 1
        assert entities.get("public_lookup") == "county_schedule"
        assert entities.get("report_year") == 2026
        assert entities.get("report_month") == 8
        citation = Citation(
            id="county-schedule-1",
            source_kind="public_record",
            title="August 2026 foreclosure notice",
            official_source=True,
            url="https://www.wilcotx.gov/308/Foreclosure-Trustee-Sales",
            retrieved_at=datetime.now(UTC),
        )
        summary = PropertySummary(
            property_id="county-schedule",
            address="Williamson County trustee sale schedule for 2026-08",
            source_citation_ids=["county-schedule-1"],
            limitations=["Official sources list August materials."],
            selected_property_id="county-schedule",
            match_confidence=0.9,
        )
        return summary, [], [citation], []

    graph = ControlledAgentGraph(GraphServices(auction_search=auction, public_search=public))
    response = await graph.run(
        _state("When is the Williamson County trustee sale schedule for August 2026?")
    )
    assert response.intent == Intent.PUBLIC_PROPERTY_LOOKUP
    assert calls["public"] == 1
    assert calls["auction"] == 0
    assert response.citations[0].id == "county-schedule-1"
    assert response.cta is None


@pytest.mark.asyncio
async def test_empty_auction_index_returns_no_match_and_null_cta() -> None:
    async def auction(
        _entities: Mapping[str, str | int],
    ) -> tuple[list[AuctionResultCard], list[Citation]]:
        return [], []

    graph = ControlledAgentGraph(GraphServices(auction_search=auction))
    response = await graph.run(_state("Show auctions for trustee Angela Zavala in July 2026"))
    assert response.intent == Intent.AUCTION_SEARCH
    assert response.auction_results == []
    assert response.cta is None
    assert "did not find a matching record" in response.answer.casefold()


@pytest.mark.asyncio
async def test_policy_rag_requires_citations() -> None:
    async def policy(
        _query: str,
    ) -> tuple[list[dict[str, object]], list[Citation]]:
        return (
            [
                {
                    "chunk_id": "c1",
                    "title": "Privacy",
                    "content": "We retain account data only as described.",
                    "score": 0.91,
                    "citation_id": "policy-1",
                }
            ],
            [
                Citation(
                    id="policy-1",
                    source_kind="policy_document",
                    title="Privacy Policy",
                    official_source=True,
                    url="https://getauctionlist.com/privacy",
                    retrieved_at=datetime.now(UTC),
                    quote="We retain account data only as described.",
                )
            ],
        )

    graph = ControlledAgentGraph(GraphServices(knowledge_search=policy))
    response = await graph.run(_state("What does the privacy policy say about retention?"))
    assert response.intent == Intent.KNOWLEDGE_POLICY
    assert response.citations
    assert response.citations[0].id == "policy-1"
    assert response.citations[0].url == "https://getauctionlist.com/privacy"


@pytest.mark.asyncio
async def test_mcp_timeout_surfaces_unavailable_sources() -> None:
    class TimeoutService:
        async def discover_trustee_sale_sources(self, **_kwargs: object) -> ToolResult:
            raise PublicRecordToolError(
                ToolErrorCategory.TIMEOUT,
                "County page timed out.",
                retryable=True,
            )

        async def search_property(self, **_kwargs: object) -> ToolResult:
            raise AssertionError("WCAD must not run for schedule lookups")

        async def get_property_details(self, **_kwargs: object) -> ToolResult:
            raise AssertionError("unused")

    services = build_graph_services(
        embeddings=None,
        retriever=None,
        auction_service=None,
        public_service=TimeoutService(),  # type: ignore[arg-type]
        principal_resolver=lambda: (_ for _ in ()).throw(AssertionError("unused")),
        trace_id_resolver=lambda: "trace",
    )
    graph = ControlledAgentGraph(services)
    response = await graph.run(
        _state("When is the Williamson County trustee sale schedule for August 2026?")
    )
    assert response.status == "failed"
    assert response.unavailable_sources
    assert response.unavailable_sources[0].source == "public_records"
    assert response.unavailable_sources[0].reason == "timeout"
    assert response.unavailable_sources[0].retryable is True
    assert response.cta is None


@pytest.mark.asyncio
async def test_county_schedule_adapter_returns_citations_from_mcp() -> None:
    class FakeService:
        async def discover_trustee_sale_sources(
            self, *, trace_id: str, year: int | None = None, month: int | None = None
        ) -> ToolResult:
            assert year == 2026
            assert month == 8
            item = TrusteeSaleSource(
                title="August 2026 Foreclosure Sales",
                url="https://www.wilcotx.gov/308/Foreclosure-Trustee-Sales#aug",
                document_type="webpage",
            )
            return ToolResult(
                items=(item.model_dump(mode="json"),),
                metadata=ToolMetadata(
                    parser_version="test",
                    trace_id=trace_id,
                    audit_id=str(uuid4()),
                    source_url="https://www.wilcotx.gov/308/Foreclosure-Trustee-Sales",
                ),
            )

        async def search_property(self, **_kwargs: object) -> ToolResult:
            raise AssertionError("schedule path must not call WCAD")

    services = build_graph_services(
        embeddings=None,
        retriever=None,
        auction_service=None,
        public_service=FakeService(),  # type: ignore[arg-type]
        principal_resolver=lambda: (_ for _ in ()).throw(AssertionError("unused")),
        trace_id_resolver=lambda: "trace-1",
    )
    summary, candidates, citations, unavailable = await services.public_search(
        {"public_lookup": "county_schedule", "report_year": 2026, "report_month": 8}
    )
    assert unavailable == []
    assert candidates == []
    assert summary is not None
    assert summary.property_id == "county-schedule"
    assert citations[0].url is not None
    assert "wilcotx.gov" in citations[0].url


def test_storage_path_ingest_builds_july_2026_rows() -> None:
    path = "williamson_county/getAuctionList_July_2026.xlsx"
    assert parse_supabase_uri(f"supabase://auction_files/{path}") == (
        "auction_files",
        path,
    )
    meta = parse_auction_storage_path(path)
    assert meta.report_year == 2026
    assert meta.report_month == 7
    assert meta.county == "Williamson"

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Auctions"
    sheet.append(["Report ID", "Trustee", "Property Address", "City", "Zip Code", "Mortgagor"])
    sheet.append(
        ["FC-1", "Angela Zavala", "1021 Cowberry Dr", "Round Rock", "78681", "John Smith"]
    )
    buffer = io.BytesIO()
    workbook.save(buffer)
    parsed = XlsxParser().parse(buffer.getvalue())
    rows = build_publishable_rows(parsed.units, meta=meta)
    assert len(rows) == 1
    assert rows[0].report_year == 2026
    assert rows[0].report_month == 7
    assert rows[0].normalized.stable_key
    assert rows[0].normalized.trustee is not None
    assert rows[0].normalized.address is not None
    assert rows[0].normalized.coordinates.sheet_name == "Auctions"
    assert rows[0].normalized.coordinates.row_start == 2


@pytest.mark.asyncio
async def test_discover_empty_month_is_not_contract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <html><body>
      <div id="julyDate" class="date"><a href="July/files.aspx">July 7, 2026</a></div>
    </body></html>
    """

    class Client:
        async def get(self, url: str) -> tuple[Any, bool]:
            request = httpx.Request("GET", url)
            response = httpx.Response(200, text=html, request=request)
            return type(
                "R",
                (),
                {
                    "body": response.content,
                    "final_url": url,
                    "content_type": "text/html",
                },
            )(), False

    service = PublicRecordsService(Client())  # type: ignore[arg-type]
    result = await service.discover_trustee_sale_sources(trace_id="t", year=2026, month=8)
    assert result.items == ()
    assert result.warnings


@pytest.mark.asyncio
async def test_discover_calendar_month_from_apps_wilco() -> None:
    html = """
    <html><body>
      <div id="julyDate" class="date"><a href="July/files.aspx">July 7, 2026</a></div>
      <div id="augustDate" class="date"><a href="August/files.aspx">August 4, 2026</a></div>
    </body></html>
    """

    class Client:
        async def get(self, url: str) -> tuple[Any, bool]:
            assert "apps.wilco.org" in url or "trustee_sales" in url
            request = httpx.Request("GET", url)
            response = httpx.Response(200, text=html, request=request)
            return type(
                "R",
                (),
                {
                    "body": response.content,
                    "final_url": url,
                    "content_type": "text/html",
                },
            )(), False

    service = PublicRecordsService(Client())  # type: ignore[arg-type]
    result = await service.discover_trustee_sale_sources(trace_id="t", year=2026, month=7)
    assert len(result.items) == 1
    source = TrusteeSaleSource.model_validate(result.items[0])
    assert "July 7, 2026" in source.title
    assert "July/files.aspx" in str(source.url)