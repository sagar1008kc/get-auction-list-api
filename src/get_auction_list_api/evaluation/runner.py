"""Versioned, deterministic evaluation runner with no network or model dependency."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from get_auction_list_api.graph import AgentState, ControlledAgentGraph, GraphServices
from get_auction_list_api.schemas import (
    AuctionResultCard,
    Citation,
    Intent,
    PropertySummary,
    UnavailableSource,
    WcadCandidate,
)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    message: str
    expected_intent: Intent
    expected_entities: dict[str, str | int] = Field(default_factory=dict)
    expected_tool: str
    expected_records: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    expected_no_answer: bool = False
    expected_grounded: bool = True
    expected_disclaimer: bool = False
    source_outcome: str = "success"


def _citation(identifier: str) -> Citation:
    return Citation(
        id=identifier,
        source_kind="evaluation_fixture",
        title="Sanitized approved source fixture",
        official_source=True,
        retrieved_at=datetime.now(UTC),
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
        retry_budget={"classifier": 1, "retrieval": 1, "tools": 1},
    )


async def _evaluate_case(case: EvaluationCase) -> dict[str, bool]:
    captured: dict[str, object] = {"tool": "none", "entities": {}}

    async def knowledge(_query: str) -> tuple[list[dict[str, object]], list[Citation]]:
        captured["tool"] = "knowledge_retrieval"
        if case.source_outcome == "error":
            raise OSError("deterministic source failure")
        evidence: list[dict[str, object]] = (
            [{"summary": "The approved policy fixture describes privacy and records controls."}]
            if case.expected_citations
            else []
        )
        return evidence, [_citation(value) for value in case.expected_citations]

    async def auction(
        entities: Mapping[str, str | int],
    ) -> tuple[list[AuctionResultCard], list[Citation]]:
        captured.update(tool="auction_search", entities=dict(entities))
        if case.source_outcome == "error":
            raise OSError("deterministic source failure")
        cards = [
            AuctionResultCard(
                record_key=value,
                match_level="exact",
                match_confidence=1,
                source_citation_ids=case.expected_citations,
            )
            for value in case.expected_records
        ]
        return cards, [_citation(value) for value in case.expected_citations]

    async def public(
        entities: Mapping[str, str | int],
    ) -> tuple[
        PropertySummary | None,
        list[WcadCandidate],
        list[Citation],
        list[UnavailableSource],
    ]:
        captured.update(tool="public_records", entities=dict(entities))
        if case.source_outcome == "error":
            raise OSError("deterministic source failure")
        summary = (
            PropertySummary(
                property_id="P-100",
                address=str(entities.get("address", "Sanitized address")),
                source_citation_ids=case.expected_citations,
            )
            if case.expected_records
            else None
        )
        return summary, [], [_citation(value) for value in case.expected_citations], []

    graph = ControlledAgentGraph(
        GraphServices(
            knowledge_search=knowledge,
            auction_search=auction,
            public_search=public,
        )
    )
    response = await graph.run(_state(case.message))
    record_keys = [item.record_key for item in response.auction_results]
    if response.property_summary is not None:
        record_keys.append(response.property_summary.property_id)
    citation_ids = [item.id for item in response.citations]
    factual = bool(response.auction_results or response.property_summary or response.citations)
    no_answer = (
        "did not find" in response.answer.casefold()
        or "could not find sufficient" in response.answer.casefold()
    )
    return {
        "router": response.intent is case.expected_intent,
        "filter": captured["entities"] == case.expected_entities,
        "search": record_keys == case.expected_records,
        "retrieval": citation_ids == case.expected_citations,
        "grounding": (not factual or bool(response.citations)) == case.expected_grounded,
        "relevance": set(citation_ids) == set(case.expected_citations),
        "citations": (not factual or bool(response.citations)),
        "no_answer": no_answer == case.expected_no_answer,
        "tool_selection": captured["tool"] == case.expected_tool,
        "arguments": captured["entities"] == case.expected_entities,
        "success": response.status in {"completed", "partial", "failed"},
        "disclaimer": bool(response.disclaimer) == case.expected_disclaimer,
    }


async def evaluate_dataset(path: Path) -> dict[str, float]:
    content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    cases = [
        EvaluationCase.model_validate_json(line) for line in content.splitlines() if line.strip()
    ]
    if not cases:
        raise ValueError("Evaluation dataset is empty.")
    outcomes = await asyncio.gather(*(_evaluate_case(case) for case in cases))
    names = outcomes[0].keys()
    return {name: sum(value[name] for value in outcomes) / len(outcomes) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("datasets") / "v1.jsonl",
    )
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()
    scores = asyncio.run(evaluate_dataset(args.dataset))
    print(json.dumps(scores, sort_keys=True))
    failed = {name: score for name, score in scores.items() if score < args.threshold}
    if failed:
        raise SystemExit(f"evaluation thresholds failed: {failed}")


if __name__ == "__main__":
    main()
