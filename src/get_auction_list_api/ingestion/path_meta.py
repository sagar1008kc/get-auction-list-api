"""Derive county / report period metadata from auction Storage paths."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_FILE_RE = re.compile(
    r"^getAuctionList_([A-Za-z]+)_((?:19|20)\d{2})\.xlsx$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AuctionFileMeta:
    county: str
    report_year: int
    report_month: int
    storage_path: str

    @property
    def report_period(self) -> str:
        return f"{self.report_year:04d}-{self.report_month:02d}"


def _title_case_county(folder: str) -> str:
    base = folder.replace("_county", "").replace("_", " ").strip()
    if not base:
        return folder
    return " ".join(part.capitalize() for part in base.split())


def parse_auction_storage_path(path: str) -> AuctionFileMeta:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("Auction Storage path must include county folder and file name.")
    county_folder, file_name = parts[0], parts[-1]
    match = _FILE_RE.match(file_name)
    if match is None:
        raise ValueError("Auction file name does not match getAuctionList_{month}_{year}.xlsx.")
    month = _MONTH_NAMES.get(match.group(1).casefold())
    if month is None:
        raise ValueError("Auction file month name is not recognized.")
    return AuctionFileMeta(
        county=_title_case_county(county_folder),
        report_year=int(match.group(2)),
        report_month=month,
        storage_path=path,
    )
