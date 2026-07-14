from collections.abc import Sequence
from decimal import Decimal
from uuid import uuid4

import pytest

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.database import DatabaseRow
from get_auction_list_api.domain import (
    AuctionSearchFilters,
    normalize_search_text,
    stable_auction_key,
    stable_idempotency_key,
)
from get_auction_list_api.repositories import PostgresAuctionRepository, PostgresRetrievalRepository


class RecordingExecutor:
    def __init__(self, rows: Sequence[DatabaseRow] = ()) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *args: object) -> Sequence[DatabaseRow]:
        self.calls.append((query, args))
        return self.rows

    async def fetchrow(self, query: str, *args: object) -> DatabaseRow | None:
        self.calls.append((query, args))
        return None

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        return "OK"


def principal(*permissions: Permission) -> Principal:
    return Principal(user_id=uuid4(), roles=frozenset({"ops"}), permissions=frozenset(permissions))


def test_stable_key_is_deterministic_and_has_no_rid_semantics() -> None:
    first = stable_auction_key(
        source_key="wilco-june",
        source_record_key="sheet1:row9",
        normalization_version="v1",
    )
    second = stable_auction_key(
        source_key="wilco-june",
        source_record_key="sheet1:row9",
        normalization_version="v1",
    )

    assert first == second
    assert len(first) == 64


def test_normalization_is_conservative_and_deterministic() -> None:
    assert normalize_search_text("  García,  LLC. ") == "garcía llc"
    assert normalize_search_text("...") is None


def test_idempotency_key_is_scoped_and_deterministic() -> None:
    values = {
        stable_idempotency_key(
            operation="ingest",
            principal_scope="user-1",
            canonical_payload='{"source":"approved"}',
        )
        for _ in range(2)
    }

    assert len(values) == 1
    assert next(iter(values)) != stable_idempotency_key(
        operation="ingest",
        principal_scope="user-2",
        canonical_payload='{"source":"approved"}',
    )


@pytest.mark.asyncio
async def test_repository_passes_user_input_only_as_query_parameters() -> None:
    record_id = uuid4()
    executor = RecordingExecutor(
        rows=(
            {
                "id": record_id,
                "rid": "RID-7",
                "stable_key": None,
                "property_address": None,
                "city": None,
                "zip_code": None,
                "trustee_name": None,
                "sale_date": None,
                "source_url": "https://www.wilcotx.gov/308/Foreclosure-Trustee-Sales",
                "source_name": "Williamson County",
                "document_version_id": None,
                "source_coordinates": {},
                "match_score": 3.0,
            },
        )
    )
    repository = PostgresAuctionRepository(executor)
    attack = "Smith'); drop table auction_records;--"

    records = await repository.search(
        principal(Permission.AUCTION_READ),
        AuctionSearchFilters(trustee=attack),
    )

    query, arguments = executor.calls[0]
    assert attack not in query
    assert arguments[0] == normalize_search_text(attack)
    assert records[0].id == record_id


@pytest.mark.asyncio
async def test_repository_allows_authenticated_search_without_fine_grained_gate() -> None:
    executor = RecordingExecutor()
    repository = PostgresAuctionRepository(executor)

    rows = await repository.search(principal(), AuctionSearchFilters())

    assert rows == []
    assert executor.calls


@pytest.mark.asyncio
async def test_hybrid_retrieval_allows_authenticated_caller_without_document_read() -> None:
    executor = RecordingExecutor()
    repository = PostgresRetrievalRepository(executor)
    embedding = [0.0] * 1536

    rows = await repository.hybrid(
        principal(Permission.AUCTION_READ, Permission.TOOL_EXECUTE),
        query="privacy policy",
        embedding=embedding,
    )
    assert rows == []
    assert executor.calls


def test_search_rejects_inverted_ranges() -> None:
    with pytest.raises(ValueError):
        AuctionSearchFilters(min_equity=Decimal(10), max_equity=Decimal(1))
