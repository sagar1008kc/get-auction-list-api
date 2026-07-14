from pathlib import Path

import pytest
from prometheus_client import generate_latest

from get_auction_list_api.api.metrics import (
    GRAPH_NODE_DURATION,
    HTTP_REQUESTS,
    bounded,
    route_label,
    tool_label,
)
from get_auction_list_api.evaluation.runner import evaluate_dataset
from get_auction_list_api.observability.logging import REDACTED, redact
from get_auction_list_api.parsers.html import HtmlParser
from get_auction_list_api.parsers.pdf import PdfParser
from get_auction_list_api.parsers.xlsx import XlsxParser

FIXTURES = Path(__file__).with_name("fixtures")
DATASET = Path(__file__).parents[1] / "src/get_auction_list_api/evaluation/datasets/v1.jsonl"


def test_metric_labels_collapse_unbounded_values() -> None:
    assert route_label("/users/private-id") == "other"
    assert tool_label("attacker.dynamic.tool") == "other"
    assert bounded("dynamic", frozenset({"known"})) == "other"

    HTTP_REQUESTS.labels("GET", route_label("/users/123"), "2xx").inc()
    GRAPH_NODE_DURATION.labels("routing", "success").observe(0.01)
    rendered = generate_latest().decode()

    assert 'route="other"' in rendered
    assert "/users/123" not in rendered
    assert 'node="routing"' in rendered


def test_telemetry_redaction_masks_pii_tokens_and_secrets() -> None:
    payload = redact(
        {
            "message": "Email owner@example.com at 512-555-0100, 100 Example Road",
            "jwt": "eyJabcdefghijk.abcdefghijkl.abcdefghijkl",
            "authorization": "Bearer secret-value",
        }
    )
    rendered = repr(payload)

    assert rendered.count(REDACTED) >= 4
    assert "owner@example.com" not in rendered
    assert "512-555-0100" not in rendered
    assert "secret-value" not in rendered


def test_committed_sanitized_fixtures_exercise_real_parsers() -> None:
    pdf = PdfParser().parse((FIXTURES / "county_trustee_notice.pdf").read_bytes())
    xlsx = XlsxParser().parse((FIXTURES / "auction_list.xlsx").read_bytes())
    policy = HtmlParser().parse(
        (FIXTURES / "privacy_disclaimer.html").read_bytes(),
        source_url="https://getauctionlist.com/privacy",
    )
    wcad_search = HtmlParser().parse(
        (FIXTURES / "wcad_search_results.html").read_bytes(),
        source_url="https://search.wcad.org/",
    )
    wcad_detail = HtmlParser().parse(
        (FIXTURES / "wcad_property_details.html").read_bytes(),
        source_url="https://search.wcad.org/Property/P-100",
    )

    assert "TEST-100" in pdf.units[0].text
    assert xlsx.units[0].fields["Record ID"] == "TEST-100"
    assert "Official records control" in policy.units[1].text
    assert "P-100" in wcad_search.units[0].text
    assert any("$100,000" in unit.text for unit in wcad_detail.units)


@pytest.mark.asyncio
async def test_deterministic_evaluation_dataset_meets_all_gates() -> None:
    scores = await evaluate_dataset(DATASET)

    assert set(scores) == {
        "router",
        "filter",
        "search",
        "retrieval",
        "grounding",
        "relevance",
        "citations",
        "no_answer",
        "tool_selection",
        "arguments",
        "success",
        "disclaimer",
    }
    assert scores == {name: 1.0 for name in scores}
