"""Knowledge retrieval no longer requires a fine-grained document:read permission."""

from uuid import uuid4

import pytest

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.graph.adapters import build_graph_services
from get_auction_list_api.public_records import ApprovedHttpClient, PublicRecordsService


def _public_service() -> PublicRecordsService:
    return PublicRecordsService(
        ApprovedHttpClient(
            ("example.com",),
            resolver=lambda _host: ["93.184.216.34"],
        )
    )


@pytest.mark.asyncio
async def test_knowledge_search_allows_authenticated_user_without_document_read() -> None:
    principal = Principal(
        user_id=uuid4(),
        roles=frozenset({"user"}),
        permissions=frozenset({Permission.AUCTION_READ, Permission.TOOL_EXECUTE}),
    )
    services = build_graph_services(
        embeddings=None,
        retriever=None,
        auction_service=None,
        public_service=_public_service(),
        principal_resolver=lambda: principal,
        trace_id_resolver=lambda: "trace",
    )

    with pytest.raises(OSError, match="Knowledge retrieval is not configured"):
        await services.knowledge_search("What does the privacy policy say?")
