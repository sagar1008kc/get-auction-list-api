from fastapi.testclient import TestClient

from get_auction_list_api.config import Settings
from get_auction_list_api.dependencies import AppDependencies, NamedReadinessProbe
from get_auction_list_api.main import create_app


def test_liveness_has_no_dependency_checks() -> None:
    called = False

    async def probe() -> bool:
        nonlocal called
        called = True
        return True

    settings = Settings(environment="test")
    dependencies = AppDependencies(
        settings=settings,
        readiness_probes=(NamedReadinessProbe(name="database", check=probe),),
    )

    with TestClient(create_app(dependencies=dependencies)) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert called is False


def test_readiness_reports_degraded_dependency_without_details() -> None:
    async def unavailable() -> bool:
        raise OSError("credential-bearing upstream detail")

    settings = Settings(environment="test")
    dependencies = AppDependencies(
        settings=settings,
        readiness_probes=(NamedReadinessProbe(name="database", check=unavailable),),
    )

    with TestClient(create_app(dependencies=dependencies)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "service": "get-auction-list-api",
        "version": "development",
        "components": {"database": "unavailable"},
    }
    assert "credential-bearing" not in response.text


def test_readiness_is_ok_without_enabled_integrations() -> None:
    with TestClient(create_app(settings=Settings(environment="test"))) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["components"] is None
    assert response.headers["cache-control"] == "no-store"
