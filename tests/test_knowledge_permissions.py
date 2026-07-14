from uuid import uuid4

import pytest

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.errors import AppError
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
async def test_knowledge_search_requires_document_read_for_user_role_path() -> None:
    denied = Principal(
        user_id=uuid4(),
        roles=frozenset({"user"}),
        permissions=frozenset({Permission.AUCTION_READ, Permission.TOOL_EXECUTE}),
    )
    allowed = Principal(
        user_id=denied.user_id,
        roles=frozenset({"user"}),
        permissions=frozenset(
            {Permission.AUCTION_READ, Permission.DOCUMENT_READ, Permission.TOOL_EXECUTE}
        ),
    )
    holder: dict[str, Principal] = {"principal": denied}

    services = build_graph_services(
        embeddings=None,
        retriever=None,
        auction_service=None,
        public_service=_public_service(),
        principal_resolver=lambda: holder["principal"],
        trace_id_resolver=lambda: "trace",
    )

    with pytest.raises(AppError) as error:
        await services.knowledge_search("What does the privacy policy say?")
    assert error.value.status_code == 403

    holder["principal"] = allowed
    with pytest.raises(OSError, match="Knowledge retrieval is not configured"):
        await services.knowledge_search("What does the privacy policy say?")
