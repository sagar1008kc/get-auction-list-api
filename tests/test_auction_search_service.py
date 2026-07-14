from collections.abc import Sequence
from uuid import uuid4

import pytest

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.domain import AuctionRecord, AuctionSearchFilters
from get_auction_list_api.tools.auction_search import AuctionSearchService, MatchLevel


class Repository:
    def __init__(self, records: Sequence[AuctionRecord]) -> None:
        self.records = records

    async def search(
        self,
        principal: Principal,
        filters: AuctionSearchFilters,
    ) -> Sequence[AuctionRecord]:
        return self.records


def principal() -> Principal:
    return Principal(
        user_id=uuid4(),
        permissions=frozenset({Permission.AUCTION_READ}),
    )


def record(score: float) -> AuctionRecord:
    return AuctionRecord(
        id=uuid4(),
        stable_key="stable",
        property_address="1021 Cowberry Dr",
        source_url="https://apps.wilco.org/countyclerk/trustee_sales/July/a.pdf",
        source_name="Williamson County",
        source_coordinates={"page_number": 3},
        match_score=score,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "level"),
    [(4.0, MatchLevel.CANONICAL), (3.0, MatchLevel.EXACT), (2.0, MatchLevel.PREFIX)],
)
async def test_deterministic_match_tiers(score: float, level: MatchLevel) -> None:
    response = await AuctionSearchService(Repository((record(score),))).search(
        principal(),
        AuctionSearchFilters(trustee="Angela Zavala"),
    )
    assert response.results[0].match_level is level
    assert response.results[0].citation_id == response.citations[0].id


@pytest.mark.asyncio
async def test_low_score_remains_candidate_and_cta_uses_required_fields() -> None:
    response = await AuctionSearchService(Repository((record(0.2),))).search(
        principal(),
        AuctionSearchFilters(
            mortgagor_first="John",
            mortgagor_last="Smith",
            report_year=2026,
            report_month=7,
        ),
    )
    assert response.results[0].match_level is MatchLevel.CANDIDATE
    assert response.results[0].limitations
    assert response.cta is not None
    assert response.cta.model_dump(by_alias=True)["filters"] == {
        "trustee": None,
        "mortgagorFirstName": "John",
        "mortgagorLastName": "Smith",
        "year": 2026,
        "month": 7,
    }
