"""Deterministic auction normalization; uncertain source data stays explicit."""

import hashlib
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from get_auction_list_api.domain import normalize_search_text
from get_auction_list_api.parsers.models import SourceCoordinates

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_ORGANIZATION_WORDS = {
    "llc",
    "inc",
    "corp",
    "corporation",
    "company",
    "bank",
    "trust",
    "ministries",
    "association",
    "lp",
    "ltd",
}
_ADDRESS_WORDS = {
    "street": "st",
    "st": "st",
    "avenue": "ave",
    "ave": "ave",
    "road": "rd",
    "rd": "rd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "boulevard": "blvd",
    "blvd": "blvd",
    "court": "ct",
    "ct": "ct",
    "highway": "hwy",
    "hwy": "hwy",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
}
_MULTI_NAME = re.compile(r"\s*(?:;|\||\band\b|&)\s*", re.IGNORECASE)


class EntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    UNKNOWN = "unknown"


class NormalizedName(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: str
    entity_type: EntityType
    display_name: str
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
    suffix: str | None = None
    full_name: str
    variants: tuple[str, ...]
    confidence: Decimal = Field(ge=0, le=1)


class NormalizedAddress(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: str
    normalized: str
    fingerprint: str


class AuctionAmounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    market_value: Decimal | None = None
    debt: Decimal | None = None
    opening_bid: Decimal | None = None
    estimated_equity: Decimal | None = None
    estimated_margin: Decimal | None = None


class NormalizedAuctionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stable_key: str
    normalization_version: str
    trustee: NormalizedName | None
    mortgagors: tuple[NormalizedName, ...]
    address: NormalizedAddress | None
    amounts: AuctionAmounts
    coordinates: SourceCoordinates


def _tokens(value: str) -> list[str]:
    return (normalize_search_text(value) or "").split()


def normalize_name(value: str) -> NormalizedName:
    raw = " ".join(value.split())
    tokens = _tokens(raw)
    if not tokens:
        raise ValueError("Name is empty after normalization.")
    if any(token in _ORGANIZATION_WORDS for token in tokens):
        full = " ".join(tokens)
        return NormalizedName(
            raw=raw,
            entity_type=EntityType.ORGANIZATION,
            display_name=raw,
            full_name=full,
            variants=(full,),
            confidence=Decimal("1"),
        )

    comma_parts = [part.strip() for part in raw.split(",", 1)]
    suffix = tokens[-1] if tokens[-1] in _SUFFIXES else None
    if suffix:
        tokens = tokens[:-1]
    if not tokens:
        raise ValueError("Person name has no identity tokens.")
    if len(comma_parts) == 2 and comma_parts[1]:
        last_tokens = _tokens(comma_parts[0])
        given_tokens = _tokens(comma_parts[1])
        if given_tokens and given_tokens[-1] in _SUFFIXES:
            suffix = given_tokens.pop()
        first = given_tokens[0] if given_tokens else None
        middle = " ".join(given_tokens[1:]) or None
        last = " ".join(last_tokens) or None
    elif len(tokens) >= 2:
        first, last = tokens[0], tokens[-1]
        middle = " ".join(tokens[1:-1]) or None
    else:
        first = middle = None
        last = tokens[0]
    first_last = " ".join(part for part in (first, middle, last, suffix) if part)
    last_first = " ".join(part for part in (last, first, middle, suffix) if part)
    variants = tuple(dict.fromkeys((first_last, last_first)))
    return NormalizedName(
        raw=raw,
        entity_type=EntityType.PERSON if first else EntityType.UNKNOWN,
        display_name=raw,
        first_name=first,
        middle_name=middle,
        last_name=last,
        suffix=suffix,
        full_name=first_last,
        variants=variants,
        confidence=Decimal("0.95") if first else Decimal("0.55"),
    )


def split_mortgagors(value: str | None) -> tuple[NormalizedName, ...]:
    if not value:
        return ()
    return tuple(normalize_name(part) for part in _MULTI_NAME.split(value) if part.strip())


def normalize_address(value: str) -> NormalizedAddress:
    tokens = _tokens(value)
    if not tokens:
        raise ValueError("Address is empty after normalization.")
    canonical = " ".join(_ADDRESS_WORDS.get(token, token) for token in tokens)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    return NormalizedAddress(
        raw=" ".join(value.split()), normalized=canonical, fingerprint=fingerprint
    )


def parse_decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    cleaned = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation as error:
        raise ValueError("Invalid decimal source value.") from error


def calculate_amounts(
    *,
    market_value: object = None,
    debt: object = None,
    opening_bid: object = None,
) -> AuctionAmounts:
    market = parse_decimal(market_value)
    debt_value = parse_decimal(debt)
    bid = parse_decimal(opening_bid)
    equity = market - debt_value if market is not None and debt_value is not None else None
    margin = market - bid if market is not None and bid is not None else None
    return AuctionAmounts(
        market_value=market,
        debt=debt_value,
        opening_bid=bid,
        estimated_equity=equity,
        estimated_margin=margin,
    )


def build_stable_key(
    *,
    source_authority: str,
    source_locator: str,
    report_period: str,
    normalized_identity: str,
    version: str,
) -> str:
    parts = tuple(
        (normalize_search_text(part) or "")
        for part in (
            source_authority,
            source_locator,
            report_period,
            normalized_identity,
            version,
        )
    )
    if not all(parts):
        raise ValueError("Stable key inputs must be non-empty.")
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
