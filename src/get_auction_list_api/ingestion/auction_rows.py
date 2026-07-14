"""Map parsed spreadsheet units onto normalized auction rows ready for publication."""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from get_auction_list_api.domain import normalize_search_text
from get_auction_list_api.ingestion.path_meta import AuctionFileMeta
from get_auction_list_api.normalization.auction import (
    NormalizedAuctionRow,
    build_stable_key,
    calculate_amounts,
    normalize_address,
    normalize_name,
    split_mortgagors,
)
from get_auction_list_api.parsers.models import ParsedUnit

NORMALIZATION_VERSION = "auction-v1"

_COLUMN_ALIASES: dict[str, str] = {
    "report_id": "rid",
    "id": "rid",
    "property_address": "property_address",
    "address": "property_address",
    "city": "city",
    "city_name": "city",
    "zip_code": "zip_code",
    "zip": "zip_code",
    "state": "state",
    "trustee": "trustee",
    "mortgagor": "mortgagor",
    "mortgagor_name": "mortgagor",
    "borrower": "mortgagor",
    "loan_type": "loan_type",
    "type": "loan_type",
    "auction_date": "sale_date",
    "sale_date": "sale_date",
    "estimated_equity": "equity",
    "equity": "equity",
    "est_equity": "equity",
    "estimated_margin": "margin",
    "margin": "margin",
    "est_margin": "margin",
    "opening_bid": "opening_bid",
    "tax_assessed_value": "market_value",
    "assessed_value": "market_value",
    "unpaid_balance": "debt",
    "estimated_unpaid_balance": "debt",
    "est_bal": "debt",
}


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def map_fields(raw: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        alias = _COLUMN_ALIASES.get(_norm_key(str(key)))
        if alias is None or value is None or value == "":
            continue
        mapped[alias] = value
    return mapped


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _as_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _optional_decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value).replace("$", "").replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def unit_to_normalized_row(
    unit: ParsedUnit,
    *,
    meta: AuctionFileMeta,
) -> NormalizedAuctionRow | None:
    fields = map_fields(unit.fields)
    address_raw = _as_str(fields.get("property_address"))
    trustee_raw = _as_str(fields.get("trustee"))
    mortgagor_raw = _as_str(fields.get("mortgagor"))
    if not any((address_raw, trustee_raw, mortgagor_raw, fields.get("rid"))):
        return None

    trustee = normalize_name(trustee_raw) if trustee_raw else None
    address = normalize_address(address_raw) if address_raw else None
    mortgagors = split_mortgagors(mortgagor_raw)
    amounts = calculate_amounts(
        market_value=fields.get("market_value"),
        debt=fields.get("debt"),
        opening_bid=fields.get("opening_bid"),
    )
    equity = _optional_decimal(fields.get("equity"))
    margin = _optional_decimal(fields.get("margin"))
    if equity is not None or margin is not None:
        amounts = amounts.model_copy(
            update={
                "estimated_equity": equity if equity is not None else amounts.estimated_equity,
                "estimated_margin": margin if margin is not None else amounts.estimated_margin,
            }
        )

    identity = (
        (address.normalized if address else None)
        or (trustee.full_name if trustee else None)
        or (mortgagors[0].full_name if mortgagors else None)
        or _as_str(fields.get("rid"))
        or f"row-{unit.coordinates.row_start}"
    )
    locator = f"{unit.coordinates.sheet_name or 'sheet'}:{unit.coordinates.row_start or 0}"
    stable_key = build_stable_key(
        source_authority=meta.county,
        source_locator=locator,
        report_period=meta.report_period,
        normalized_identity=identity,
        version=NORMALIZATION_VERSION,
    )
    return NormalizedAuctionRow(
        stable_key=stable_key,
        normalization_version=NORMALIZATION_VERSION,
        trustee=trustee,
        mortgagors=mortgagors,
        address=address,
        amounts=amounts,
        coordinates=unit.coordinates,
    )


class PublishableAuctionRow:
    """Concrete row payload for the auction_records publisher."""

    __slots__ = (
        "city",
        "county",
        "loan_type",
        "normalized",
        "property_address",
        "report_month",
        "report_year",
        "rid",
        "sale_date",
        "state",
        "zip_code",
    )

    def __init__(
        self,
        *,
        normalized: NormalizedAuctionRow,
        rid: str | None,
        property_address: str | None,
        city: str | None,
        state: str | None,
        zip_code: str | None,
        sale_date: date | None,
        loan_type: str | None,
        county: str,
        report_year: int,
        report_month: int,
    ) -> None:
        self.normalized = normalized
        self.rid = rid
        self.property_address = property_address
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.sale_date = sale_date
        self.loan_type = loan_type
        self.county = county
        self.report_year = report_year
        self.report_month = report_month


def build_publishable_rows(
    units: tuple[ParsedUnit, ...],
    *,
    meta: AuctionFileMeta,
) -> tuple[PublishableAuctionRow, ...]:
    rows: list[PublishableAuctionRow] = []
    for unit in units:
        fields = map_fields(unit.fields)
        normalized = unit_to_normalized_row(unit, meta=meta)
        if normalized is None:
            continue
        zip_code = _as_str(fields.get("zip_code"))
        if zip_code and not re.fullmatch(r"\d{5}(?:-\d{4})?", zip_code):
            zip_code = None
        rows.append(
            PublishableAuctionRow(
                normalized=normalized,
                rid=_as_str(fields.get("rid")),
                property_address=_as_str(fields.get("property_address")),
                city=_as_str(fields.get("city")),
                state=_as_str(fields.get("state")) or "TX",
                zip_code=zip_code,
                sale_date=_as_date(fields.get("sale_date")),
                loan_type=normalize_search_text(_as_str(fields.get("loan_type"))),
                county=meta.county,
                report_year=meta.report_year,
                report_month=meta.report_month,
            )
        )
    return tuple(rows)
