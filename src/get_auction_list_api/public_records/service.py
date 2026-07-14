"""Deterministic adapters and correlation for approved public records."""

import re
from collections.abc import Awaitable, Callable, Sequence
from difflib import SequenceMatcher
from urllib.parse import quote, urlencode, urljoin
from uuid import uuid4

from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, TypeAdapter

from get_auction_list_api.public_records.http import ApprovedHttpClient
from get_auction_list_api.public_records.models import (
    CorrelationCandidate,
    ForeclosureNotice,
    ForeclosureRecord,
    PublicRecordToolError,
    ToolErrorCategory,
    ToolMetadata,
    ToolResult,
    TrusteeSaleSource,
    WcadProperty,
    WcadPropertyDetails,
)

AuditSink = Callable[[str, str, int, str | None], Awaitable[None]]
PARSER_VERSION = "public-records/1.0"
COUNTY_DOCUMENTS = "https://apps.wilco.org/countyclerk/trustee_sales/"
WCAD_SEARCH = "https://search.wcad.org/"
_HTTP_URL = TypeAdapter(HttpUrl)


async def _noop_audit(_audit_id: str, _tool: str, _count: int, _error: str | None) -> None:
    return None


def _text(value: str) -> str:
    return " ".join(value.split())


def _url(value: str) -> HttpUrl:
    return _HTTP_URL.validate_python(value)


class PublicRecordsService:
    """Read-only source facade used both in-process and through FastMCP."""

    def __init__(
        self,
        client: ApprovedHttpClient,
        *,
        audit_sink: AuditSink = _noop_audit,
    ) -> None:
        self._client = client
        self._audit = audit_sink

    async def _result(
        self,
        *,
        tool: str,
        trace_id: str,
        source_url: str,
        items: Sequence[BaseModel],
        cache_hit: bool,
        warnings: tuple[str, ...] = (),
    ) -> ToolResult:
        audit_id = str(uuid4())
        values = tuple(item.model_dump(mode="json") for item in items)
        await self._audit(audit_id, tool, len(values), None)
        return ToolResult(
            items=values,
            metadata=ToolMetadata(
                parser_version=PARSER_VERSION,
                trace_id=trace_id,
                audit_id=audit_id,
                source_url=_url(source_url),
                cache_hit=cache_hit,
            ),
            warnings=warnings,
        )

    async def discover_trustee_sale_sources(
        self,
        *,
        trace_id: str,
        year: int | None = None,
        month: int | None = None,
    ) -> ToolResult:
        # The wilcotx.gov landing page is mostly an iframe; the live calendar is on
        # apps.wilco.org with per-month sale dates (e.g. "July 7, 2026").
        calendar_response, calendar_hit = await self._client.get(COUNTY_DOCUMENTS)
        calendar_soup = BeautifulSoup(calendar_response.body, "html.parser")
        sources: list[TrusteeSaleSource] = []
        month_names = (
            "",
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
        )

        for link in calendar_soup.select("a[href]"):
            href = str(link.get("href", "")).strip()
            title = _text(link.get_text(" ", strip=True))
            if not title or not href:
                continue
            lowered = f"{title} {href}".casefold()
            # Calendar entries look like "July 7, 2026" → July/files.aspx
            date_match = re.search(
                r"\b(january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\s+(\d{1,2}),\s*(20\d{2})\b",
                title,
                re.IGNORECASE,
            )
            if date_match is None and "files.aspx" not in lowered:
                continue
            entry_month = None
            entry_year = None
            if date_match is not None:
                entry_month = month_names.index(date_match.group(1).casefold())
                entry_year = int(date_match.group(3))
            elif month is not None:
                month_name = month_names[month]
                if month_name not in lowered:
                    continue
            if year is not None and entry_year is not None and entry_year != year:
                continue
            if year is not None and entry_year is None and str(year) not in lowered:
                continue
            if month is not None and entry_month is not None and entry_month != month:
                continue
            if month is not None and entry_month is None:
                month_name = month_names[month]
                month_token_ok = (
                    re.search(rf"(?:^|\D)0?{month}(?:\D|$)", lowered) is not None
                    or month_name in lowered
                )
                if not month_token_ok:
                    continue
            candidate = urljoin(calendar_response.final_url, href)
            sources.append(
                TrusteeSaleSource(
                    title=title if date_match is None else f"Trustee sale date: {title}",
                    url=_url(candidate),
                    document_type="webpage",
                )
            )

        # Also retain any PDF/notice links on the same calendar host.
        for link in calendar_soup.select("a[href]"):
            href = str(link.get("href", ""))
            title = _text(link.get_text(" ", strip=True))
            candidate = urljoin(calendar_response.final_url, href)
            lowered = f"{title} {href}".casefold()
            if ".pdf" not in lowered:
                continue
            if year is not None and str(year) not in lowered:
                continue
            if month is not None:
                month_name = month_names[month]
                month_token_ok = (
                    re.search(rf"(?:^|\D)0?{month}(?:\D|$)", lowered) is not None
                    or month_name in lowered
                )
                if not month_token_ok:
                    continue
            sources.append(
                TrusteeSaleSource(
                    title=title or "County trustee-sale PDF",
                    url=_url(candidate),
                    document_type="pdf",
                )
            )

        # De-dupe while preserving order.
        deduped: list[TrusteeSaleSource] = []
        seen: set[str] = set()
        for source in sources:
            key = str(source.url)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        sources = deduped

        if not sources:
            # Page fetched successfully but no month match — not an upstream outage.
            if year is not None or month is not None:
                return await self._result(
                    tool="county.discover_trustee_sale_sources",
                    trace_id=trace_id,
                    source_url=calendar_response.final_url,
                    items=(),
                    cache_hit=calendar_hit,
                    warnings=(
                        "No trustee-sale schedule materials matched the requested year/month "
                        "on the approved county calendar.",
                    ),
                )
            raise PublicRecordToolError(
                ToolErrorCategory.CONTRACT_CHANGED,
                "The county source layout no longer matches the supported contract.",
            )
        return await self._result(
            tool="county.discover_trustee_sale_sources",
            trace_id=trace_id,
            source_url=calendar_response.final_url,
            items=sources,
            cache_hit=calendar_hit,
        )

    async def search_foreclosure_records(
        self,
        *,
        query: str,
        trace_id: str,
        limit: int = 10,
    ) -> ToolResult:
        if not query.strip() or not 1 <= limit <= 25:
            raise PublicRecordToolError(ToolErrorCategory.INVALID_INPUT, "Invalid search query.")
        url = f"{COUNTY_DOCUMENTS}?{urlencode({'q': query.strip()})}"
        response, hit = await self._client.get(url)
        soup = BeautifulSoup(response.body, "html.parser")
        records: list[ForeclosureRecord] = []
        for row in soup.select("[data-record-id], tr"):
            content = _text(row.get_text(" ", strip=True))
            if query.casefold() not in content.casefold():
                continue
            link = row.select_one("a[href]")
            if link is None:
                continue
            record_id = str(row.get("data-record-id") or link.get("data-id") or len(records) + 1)
            records.append(
                ForeclosureRecord(
                    record_id=record_id,
                    notice_url=_url(urljoin(response.final_url, str(link.get("href")))),
                    property_address=content[:300] or None,
                )
            )
            if len(records) == limit:
                break
        if not soup.select("body"):
            raise PublicRecordToolError(
                ToolErrorCategory.CONTRACT_CHANGED,
                "The county search response contract changed.",
            )
        return await self._result(
            tool="county.search_foreclosure_records",
            trace_id=trace_id,
            source_url=response.final_url,
            items=records,
            cache_hit=hit,
        )

    async def get_foreclosure_notice(self, *, record_id: str, trace_id: str) -> ToolResult:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", record_id):
            raise PublicRecordToolError(
                ToolErrorCategory.INVALID_INPUT, "Invalid record identifier."
            )
        url = urljoin(COUNTY_DOCUMENTS, quote(record_id, safe="") + ".html")
        response, hit = await self._client.get(url)
        soup = BeautifulSoup(response.body, "html.parser")
        title_node = soup.select_one("h1") or soup.title
        title = _text(title_node.get_text(" ", strip=True)) if title_node is not None else ""
        body = soup.select_one("main, article, #content, body")
        text = _text(body.get_text(" ", strip=True)) if body else ""
        if not title or not text:
            raise PublicRecordToolError(
                ToolErrorCategory.CONTRACT_CHANGED,
                "The county notice response contract changed.",
            )
        notice = ForeclosureNotice(
            record_id=record_id,
            title=title,
            text=text[:20_000],
            notice_url=_url(response.final_url),
        )
        return await self._result(
            tool="county.get_foreclosure_notice",
            trace_id=trace_id,
            source_url=response.final_url,
            items=(notice,),
            cache_hit=hit,
        )

    async def search_property(
        self,
        *,
        address: str,
        trace_id: str,
        limit: int = 10,
    ) -> ToolResult:
        if not address.strip() or not 1 <= limit <= 25:
            raise PublicRecordToolError(ToolErrorCategory.INVALID_INPUT, "Invalid property search.")
        url = f"{WCAD_SEARCH}?{urlencode({'address': address.strip()})}"
        response, hit = await self._client.get(url)
        soup = BeautifulSoup(response.body, "html.parser")
        properties: list[WcadProperty] = []
        for row in soup.select("[data-property-id], tr"):
            link = row.select_one("a[href]")
            content = _text(row.get_text(" ", strip=True))
            if link is None or address.casefold() not in content.casefold():
                continue
            property_id = str(row.get("data-property-id") or link.get("data-id") or "")
            if not property_id:
                continue
            properties.append(
                WcadProperty(
                    property_id=property_id,
                    address=content[:300],
                    detail_url=_url(urljoin(response.final_url, str(link.get("href")))),
                )
            )
            if len(properties) == limit:
                break
        if not soup.select("body"):
            raise PublicRecordToolError(
                ToolErrorCategory.CONTRACT_CHANGED,
                "The WCAD search response contract changed.",
            )
        return await self._result(
            tool="wcad.search_property",
            trace_id=trace_id,
            source_url=response.final_url,
            items=properties,
            cache_hit=hit,
        )

    async def get_property_details(self, *, property_id: str, trace_id: str) -> ToolResult:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", property_id):
            raise PublicRecordToolError(
                ToolErrorCategory.INVALID_INPUT, "Invalid property identifier."
            )
        url = urljoin(WCAD_SEARCH, f"Property-Detail?prop_id={quote(property_id, safe='')}")
        response, hit = await self._client.get(url)
        soup = BeautifulSoup(response.body, "html.parser")
        address_node = soup.select_one("[data-field='address'], .property-address, #address")
        if address_node is None:
            raise PublicRecordToolError(
                ToolErrorCategory.CONTRACT_CHANGED,
                "The WCAD detail response contract changed.",
            )
        attributes: dict[str, str] = {}
        for row in soup.select("tr, dl"):
            key_node = row.select_one("th, dt")
            value_node = row.select_one("td, dd")
            if key_node is not None and value_node is not None:
                attributes[_text(key_node.get_text(" ", strip=True))] = _text(
                    value_node.get_text(" ", strip=True)
                )
        item = WcadPropertyDetails(
            property_id=property_id,
            address=_text(address_node.get_text(" ", strip=True)),
            detail_url=_url(response.final_url),
            attributes=attributes,
        )
        return await self._result(
            tool="wcad.get_property_details",
            trace_id=trace_id,
            source_url=response.final_url,
            items=(item,),
            cache_hit=hit,
        )

    async def correlate_records(
        self,
        *,
        foreclosure_records: Sequence[ForeclosureRecord],
        properties: Sequence[WcadProperty],
        trace_id: str,
    ) -> ToolResult:
        candidates: list[CorrelationCandidate] = []
        for foreclosure in foreclosure_records[:25]:
            for prop in properties[:25]:
                left = _text(foreclosure.property_address or "").casefold()
                right = _text(prop.address).casefold()
                confidence = SequenceMatcher(None, left, right).ratio() if left and right else 0.0
                if confidence < 0.35:
                    continue
                candidates.append(
                    CorrelationCandidate(
                        foreclosure_record_id=foreclosure.record_id,
                        property_id=prop.property_id,
                        confidence=round(confidence, 4),
                        matched_fields=("address",) if confidence >= 0.8 else (),
                        differing_fields=() if confidence >= 0.8 else ("address",),
                    )
                )
        candidates.sort(key=lambda item: (-item.confidence, item.property_id))
        return await self._result(
            tool="property.correlate_records",
            trace_id=trace_id,
            source_url=WCAD_SEARCH,
            items=candidates,
            cache_hit=False,
            warnings=(
                ("No sufficiently similar candidates were found.",) if not candidates else ()
            ),
        )
