import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from get_auction_list_api.config import Settings
from get_auction_list_api.main import create_app


def test_rejects_untrusted_host_and_oversized_body() -> None:
    settings = Settings(environment="test", max_request_body_bytes=1024)
    with TestClient(create_app(settings=settings)) as client:
        untrusted = client.get("/health/live", headers={"host": "evil.example"})
        oversized = client.post(
            "/health/live",
            content=b"x" * 1025,
            headers={"content-type": "application/octet-stream"},
        )

    assert untrusted.status_code == 400
    assert oversized.status_code == 413


def test_cors_is_restricted_to_configured_origin() -> None:
    settings = Settings(environment="test", cors_origins=("https://app.example",))
    with TestClient(create_app(settings=settings)) as client:
        approved = client.options(
            "/health/live",
            headers={
                "origin": "https://app.example",
                "access-control-request-method": "GET",
            },
        )
        denied = client.options(
            "/health/live",
            headers={
                "origin": "https://evil.example",
                "access-control-request-method": "GET",
            },
        )

    assert approved.headers["access-control-allow-origin"] == "https://app.example"
    assert "access-control-allow-origin" not in denied.headers


def test_wildcard_security_settings_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(cors_origins=("*",))
    with pytest.raises(ValidationError):
        Settings(trusted_hosts=("*",))
