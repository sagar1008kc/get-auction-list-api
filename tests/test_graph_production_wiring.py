from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.config import Settings
from get_auction_list_api.graph.context import bind_graph_context, current_graph_context
from get_auction_list_api.graph.state import AgentState
from get_auction_list_api.graph.workflow import ControlledAgentGraph, GraphServices
from get_auction_list_api.schemas import (
    AuctionResultCard,
    Citation,
    Intent,
    PropertySummary,
    UnavailableSource,
    WcadCandidate,
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
        retry_budget={"classifier": 1, "retrieval": 1, "tools": 2},
    )


def _citation() -> Citation:
    return Citation(
        id="approved-1",
        source_kind="policy_document",
        title="Approved policy",
        official_source=True,
        retrieved_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_owner_classifier_and_grounded_synthesis_run_in_order() -> None:
    calls: list[str] = []

    class Owner:
        async def ensure_owner(self, *, user_id: str, thread_id: str) -> None:
            assert user_id and thread_id
            calls.append("owner")

    class Classifier:
        async def classify(self, _message: str) -> Intent:
            calls.append("classifier")
            return Intent.KNOWLEDGE_POLICY

    class Synthesizer:
        async def synthesize(self, state: AgentState) -> object:
            assert state["citations"][0].id == "approved-1"
            calls.append("synthesizer")

            class Result:
                answer = "Grounded answer."
                citation_ids = ("approved-1",)
                confidence = 0.88

            return Result()

    async def knowledge(
        _query: str,
    ) -> tuple[list[dict[str, object]], list[Citation]]:
        calls.append("knowledge")
        return [{"content": "Approved evidence.", "score": 0.9}], [_citation()]

    graph = ControlledAgentGraph(
        GraphServices(
            classifier=Classifier(),
            synthesizer=Synthesizer(),
            knowledge_search=knowledge,
        ),
        thread_owner_store=Owner(),
    )
    response = await graph.run(_state("What can you tell me about this?"))

    assert response.answer == "Grounded answer."
    assert response.confidence == 0.88
    assert calls == ["owner", "classifier", "knowledge", "synthesizer"]


@pytest.mark.asyncio
async def test_combined_fanout_preserves_verified_partial_result() -> None:
    async def auction(
        _entities: Mapping[str, str | int],
    ) -> tuple[list[AuctionResultCard], list[Citation]]:
        return [
            AuctionResultCard(
                record_key="record-1",
                match_level="exact",
                match_confidence=1,
                source_citation_ids=["approved-1"],
            )
        ], [_citation()]

    async def public(
        _entities: Mapping[str, str | int],
    ) -> tuple[
        PropertySummary | None,
        list[WcadCandidate],
        list[Citation],
        list[UnavailableSource],
    ]:
        return (
            None,
            [],
            [],
            [UnavailableSource(source="wcad", reason="unavailable", retryable=True)],
        )

    graph = ControlledAgentGraph(GraphServices(auction_search=auction, public_search=public))
    response = await graph.run(_state("Find foreclosure auction and WCAD at 1021 Cowberry Dr"))

    assert response.status == "partial"
    assert response.auction_results[0].record_key == "record-1"
    assert response.unavailable_sources[0].source == "wcad"


def test_graph_context_is_request_scoped() -> None:
    principal = Principal(
        user_id=uuid4(),
        roles=frozenset({"user"}),
        permissions=frozenset(
            {Permission.TOOL_EXECUTE, Permission.AUCTION_READ, Permission.DOCUMENT_READ}
        ),
    )

    with bind_graph_context(principal, "trace-1"):
        assert current_graph_context().principal == principal
        assert current_graph_context().trace_id == "trace-1"

    with pytest.raises(RuntimeError, match="not bound"):
        current_graph_context()


def test_checkpointing_requires_database_url() -> None:
    with pytest.raises(ValidationError, match="Checkpointing requires database_url"):
        Settings(environment="test", checkpoint_enabled=True)
