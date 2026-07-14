"""Verify the three Front Door sample prompts against live indexes."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.config import get_settings
from get_auction_list_api.database import AsyncDatabase
from get_auction_list_api.graph import ControlledAgentGraph
from get_auction_list_api.graph.adapters import build_graph_services
from get_auction_list_api.graph.context import bind_graph_context
from get_auction_list_api.graph.llm import OpenAIGroundedSynthesizer, OpenAIIntentClassifier
from get_auction_list_api.graph.state import AgentState
from get_auction_list_api.llm import (
    OpenAIEmbeddingProvider,
    OpenAIStructuredModel,
    create_embedding_transport,
    create_openai_client,
)
from get_auction_list_api.public_records import ApprovedHttpClient, PublicRecordsService
from get_auction_list_api.rag.repository import PostgresRAGRepository
from get_auction_list_api.rag.retrieval import HybridRetriever
from get_auction_list_api.repositories import PostgresAuctionRepository
from get_auction_list_api.tools.auction_search import AuctionSearchService

PROMPTS = (
    "How many Williamson County auction listings are available for July 2026?",
    "When is the Williamson County trustee sale schedule for August 2026?",
    "What does the GetAuctionList disclaimer say?",
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


async def main() -> None:
    settings = get_settings()
    assert settings.database_url is not None
    assert settings.openai_api_key is not None
    database = AsyncDatabase(settings.database_url.get_secret_value(), min_size=1, max_size=2)
    await database.connect()
    openai = create_openai_client(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
    try:
        model = OpenAIStructuredModel(model=settings.openai_chat_model, client=openai)
        embeddings = OpenAIEmbeddingProvider(
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            transport=create_embedding_transport(openai),
        )
        public = PublicRecordsService(
            ApprovedHttpClient(
                settings.approved_source_hosts,
                timeout_seconds=settings.public_http_timeout_seconds,
                max_attempts=settings.public_http_max_attempts,
                max_response_bytes=settings.public_http_max_response_bytes,
                cache_ttl_seconds=settings.public_http_cache_ttl_seconds,
            )
        )
        principal = Principal(
            user_id=uuid4(),
            roles=frozenset({"user"}),
            permissions=frozenset(
                {Permission.AUCTION_READ, Permission.DOCUMENT_READ, Permission.TOOL_EXECUTE}
            ),
        )
        graph = ControlledAgentGraph(
            build_graph_services(
                embeddings=embeddings,
                retriever=HybridRetriever(
                    PostgresRAGRepository(
                        database, dimensions=settings.openai_embedding_dimensions
                    )
                ),
                auction_service=AuctionSearchService(PostgresAuctionRepository(database)),
                public_service=public,
                principal_resolver=lambda: principal,
                trace_id_resolver=lambda: "verify",
                classifier=OpenAIIntentClassifier(model),
                synthesizer=OpenAIGroundedSynthesizer(model),
            )
        )
        with bind_graph_context(principal, "verify"):
            for prompt in PROMPTS:
                response = await graph.run(_state(prompt))
                print("---")
                print("prompt:", prompt)
                print("intent:", response.intent.value)
                print("status:", response.status)
                print("answer:", response.answer[:400])
                print("auction_results:", len(response.auction_results))
                print("citations:", [c.id for c in response.citations[:5]])
                print("cta:", None if response.cta is None else response.cta.label)
                print(
                    "unavailable:",
                    [(u.source, u.reason) for u in response.unavailable_sources],
                )
    finally:
        await openai.close()
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
