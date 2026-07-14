from fastapi import FastAPI
from fastapi.testclient import TestClient

from get_auction_list_api.config import Settings
from get_auction_list_api.errors import AppError, ErrorCode
from get_auction_list_api.main import create_app


def _app_with_failure() -> FastAPI:
    app = create_app(settings=Settings(environment="test"))

    @app.get("/expected-failure")
    async def expected_failure() -> None:
        raise AppError(
            code=ErrorCode.NOT_READY,
            message="A required dependency is unavailable.",
            status_code=503,
            retryable=True,
        )

    return app


def test_typed_error_uses_safe_contract_and_context_headers() -> None:
    with TestClient(_app_with_failure()) as client:
        response = client.get(
            "/expected-failure",
            headers={"X-Request-ID": "6376bda4-b9b5-4262-812e-83668dfabbb1"},
        )

    body = response.json()["error"]
    assert response.status_code == 503
    assert body["code"] == "NOT_READY"
    assert body["retryable"] is True
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["X-Trace-ID"]
