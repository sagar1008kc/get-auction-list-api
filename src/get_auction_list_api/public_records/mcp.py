"""Official FastMCP server exposing six fixed read-only operations."""

from collections.abc import Awaitable
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from get_auction_list_api.api.metrics import TOOL_CALLS, tool_label
from get_auction_list_api.observability.telemetry import tracer
from get_auction_list_api.public_records.models import ForeclosureRecord, ToolResult, WcadProperty
from get_auction_list_api.public_records.service import PublicRecordsService


async def _instrument_tool[T](name: str, operation: Awaitable[T]) -> T:
    outcome = "success"
    try:
        with tracer().start_as_current_span(f"mcp.{name}") as span:
            span.set_attribute("mcp.tool.name", name)
            return await operation
    except Exception:
        outcome = "error"
        raise
    finally:
        TOOL_CALLS.labels(tool_label(name), outcome).inc()


def create_mcp_server(service: PublicRecordsService) -> FastMCP[None]:
    server: FastMCP[None] = FastMCP(
        "GetAuctionList Public Records",
        instructions="Read-only, allowlisted public-record lookups.",
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
    )
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)

    @server.tool(name="county.discover_trustee_sale_sources", annotations=readonly)
    async def discover_trustee_sale_sources(
        year: int | None = None,
        month: int | None = None,
        trace_id: str = "",
    ) -> ToolResult:
        """Discover official Williamson County trustee-sale sources."""

        return await _instrument_tool(
            "county.discover_trustee_sale_sources",
            service.discover_trustee_sale_sources(
                year=year,
                month=month,
                trace_id=trace_id or uuid4().hex,
            ),
        )

    @server.tool(name="county.search_foreclosure_records", annotations=readonly)
    async def search_foreclosure_records(
        query: str,
        limit: int = 10,
        trace_id: str = "",
    ) -> ToolResult:
        """Search approved Williamson County foreclosure records."""

        return await _instrument_tool(
            "county.search_foreclosure_records",
            service.search_foreclosure_records(
                query=query,
                limit=limit,
                trace_id=trace_id or uuid4().hex,
            ),
        )

    @server.tool(name="county.get_foreclosure_notice", annotations=readonly)
    async def get_foreclosure_notice(record_id: str, trace_id: str = "") -> ToolResult:
        """Read one approved county foreclosure notice by identifier."""

        return await _instrument_tool(
            "county.get_foreclosure_notice",
            service.get_foreclosure_notice(
                record_id=record_id,
                trace_id=trace_id or uuid4().hex,
            ),
        )

    @server.tool(name="wcad.search_property", annotations=readonly)
    async def search_property(
        address: str,
        limit: int = 10,
        trace_id: str = "",
    ) -> ToolResult:
        """Search WCAD by a normalized street address."""

        return await _instrument_tool(
            "wcad.search_property",
            service.search_property(
                address=address,
                limit=limit,
                trace_id=trace_id or uuid4().hex,
            ),
        )

    @server.tool(name="wcad.get_property_details", annotations=readonly)
    async def get_property_details(property_id: str, trace_id: str = "") -> ToolResult:
        """Read details for one WCAD property identifier."""

        return await _instrument_tool(
            "wcad.get_property_details",
            service.get_property_details(
                property_id=property_id,
                trace_id=trace_id or uuid4().hex,
            ),
        )

    @server.tool(name="property.correlate_records", annotations=readonly)
    async def correlate_records(
        foreclosure_records: list[ForeclosureRecord],
        properties: list[WcadProperty],
        trace_id: str = "",
    ) -> ToolResult:
        """Rank deterministic county/WCAD correlation candidates."""

        return await _instrument_tool(
            "property.correlate_records",
            service.correlate_records(
                foreclosure_records=foreclosure_records,
                properties=properties,
                trace_id=trace_id or uuid4().hex,
            ),
        )

    return server
