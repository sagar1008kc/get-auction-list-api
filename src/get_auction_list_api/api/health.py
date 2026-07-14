"""Platform health endpoints."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from get_auction_list_api.dependencies import AppDependencies, get_dependencies

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    service: str
    version: str
    components: dict[str, Literal["ok", "unavailable"]] | None = None


@router.get("/live", response_model=HealthResponse)
async def liveness(
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
) -> HealthResponse:
    """Report process liveness without calling external dependencies."""

    return HealthResponse(
        status="ok",
        service=dependencies.settings.service_name,
        version=dependencies.settings.service_version,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
async def readiness(
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
) -> HealthResponse | JSONResponse:
    """Report whether all enabled dependencies are ready."""

    checks = await dependencies.readiness()
    ready = all(checks.values())
    body = HealthResponse(
        status="ok" if ready else "unavailable",
        service=dependencies.settings.service_name,
        version=dependencies.settings.service_version,
        components={
            name: "ok" if available else "unavailable" for name, available in sorted(checks.items())
        }
        or None,
    )
    if ready:
        return body
    return JSONResponse(status_code=503, content=body.model_dump(mode="json"))
