from decimal import Decimal

import pytest

from get_auction_list_api.normalization.auction import (
    EntityType,
    build_stable_key,
    calculate_amounts,
    normalize_address,
    normalize_name,
    split_mortgagors,
)


@pytest.mark.parametrize(
    ("source", "variants"),
    [
        ("Zavala, Angela M.", ("angela m zavala", "zavala angela m")),
        ("Angela M Zavala", ("angela m zavala", "zavala angela m")),
        ("Smith, John Jr.", ("john smith jr", "smith john jr")),
    ],
)
def test_person_name_orders_are_searchable(source: str, variants: tuple[str, ...]) -> None:
    normalized = normalize_name(source)
    assert normalized.variants == variants
    assert normalized.entity_type is EntityType.PERSON


def test_organizations_are_not_forced_into_person_fields() -> None:
    normalized = normalize_name("First National Bank, LLC")
    assert normalized.entity_type is EntityType.ORGANIZATION
    assert normalized.first_name is None
    assert normalized.variants == ("first national bank llc",)


def test_multiple_mortgagors_are_preserved_in_order() -> None:
    values = split_mortgagors("Smith, John; Jane Doe & Example Holdings LLC")
    assert [value.last_name for value in values] == ["smith", "doe", None]
    assert values[-1].entity_type is EntityType.ORGANIZATION


def test_address_abbreviations_have_same_fingerprint() -> None:
    long = normalize_address("1021 Cowberry Drive")
    short = normalize_address("1021 COWBERRY DR.")
    assert long.normalized == "1021 cowberry dr"
    assert long.fingerprint == short.fingerprint


def test_decimal_calculations_never_use_binary_float_math() -> None:
    amounts = calculate_amounts(
        market_value="250,000.10",
        debt="$100000.05",
        opening_bid=150000.03,
    )
    assert amounts.estimated_equity == Decimal("150000.05")
    assert amounts.estimated_margin == Decimal("100000.07")


def test_stable_key_is_versioned_and_deterministic() -> None:
    arguments = {
        "source_authority": "Williamson",
        "source_locator": "July:row-9",
        "report_period": "2026-07",
        "normalized_identity": "1021 cowberry dr",
        "version": "auction-v1",
    }
    assert build_stable_key(**arguments) == build_stable_key(**arguments)
    assert build_stable_key(**arguments) != build_stable_key(
        **{**arguments, "version": "auction-v2"}
    )
