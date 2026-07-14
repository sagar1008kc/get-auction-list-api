from uuid import uuid4

from fastapi.testclient import TestClient

from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.config import Settings
from get_auction_list_api.dependencies import AppDependencies
from get_auction_list_api.main import create_app
from get_auction_list_api.runtime import InMemoryFeedbackStore


class Authenticator:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    async def validate(self, _token: str) -> Principal:
        return self.principal


def test_feedback_create_and_idempotent_replay() -> None:
    principal = Principal(
        user_id=uuid4(),
        roles=frozenset({"user"}),
        permissions=frozenset(
            {Permission.TOOL_EXECUTE, Permission.AUCTION_READ, Permission.DOCUMENT_READ}
        ),
    )
    dependencies = AppDependencies(
        settings=Settings(environment="test"),
        authenticator=Authenticator(principal),
        feedback_store=InMemoryFeedbackStore(),
    )
    app = create_app(dependencies=dependencies)
    payload = {
        "run_id": str(uuid4()),
        "message_id": str(uuid4()),
        "rating": "up",
    }

    with TestClient(app) as client:
        created = client.post(
            "/v1/feedback",
            json=payload,
            headers={"Authorization": "Bearer test"},
        )
        replay = client.post(
            "/v1/feedback",
            json=payload,
            headers={"Authorization": "Bearer test"},
        )

    assert created.status_code == 201
    assert created.json()["created"] is True
    assert replay.status_code == 200
    assert replay.json()["created"] is False
    assert replay.json()["feedback_id"] == created.json()["feedback_id"]
