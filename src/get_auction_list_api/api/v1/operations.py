"""Authorized ingestion operations; web requests only enqueue durable work."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from get_auction_list_api.api.security import current_principal
from get_auction_list_api.auth import Permission, Principal, forbidden
from get_auction_list_api.dependencies import (
    AppDependencies,
    IngestionOperations,
    get_dependencies,
)
from get_auction_list_api.errors import AppError, ErrorCode

router = APIRouter(prefix="/v1/ingestion", tags=["operations"])


class IngestionJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["supabase_storage", "registered_url"]
    storage_bucket: str | None = Field(default=None, max_length=100)
    storage_path: str | None = Field(default=None, max_length=1000)
    source_id: UUID | None = None
    document_type: str = Field(max_length=80)
    force_new_version: bool = False

    @model_validator(mode="after")
    def source_coordinates(self) -> "IngestionJobRequest":
        if self.source_type == "supabase_storage" and not (
            self.storage_bucket and self.storage_path
        ):
            raise ValueError("Storage jobs require bucket and path.")
        if self.source_type == "registered_url" and self.source_id is None:
            raise ValueError("URL jobs require a registered source_id.")
        return self


class SourceSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ids: list[UUID] = Field(min_length=1, max_length=100)
    mode: Literal["incremental", "full_reconcile"] = "incremental"


def _operations(dependencies: AppDependencies) -> IngestionOperations:
    if dependencies.ingestion_operations is None:
        raise AppError(
            code=ErrorCode.NOT_READY,
            message="Ingestion operations are not ready.",
            status_code=503,
            retryable=True,
        )
    return dependencies.ingestion_operations


@router.post("/jobs", status_code=202)
async def create_job(
    body: IngestionJobRequest,
    request: Request,
    principal: Annotated[Principal, Depends(current_principal)],
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
) -> dict[str, object]:
    principal.require(Permission.INGESTION_WRITE)
    result = await _operations(dependencies).enqueue(
        principal,
        body.model_dump(mode="json", exclude_none=True),
    )
    return {**result, "request_id": request.state.request_id}


@router.get("/jobs/{job_id}")
async def job_status(
    job_id: UUID,
    principal: Annotated[Principal, Depends(current_principal)],
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
) -> dict[str, object]:
    principal.require(Permission.INGESTION_WRITE)
    result = await _operations(dependencies).status(principal, job_id)
    if result is None:
        raise AppError(
            code=ErrorCode.INVALID_REQUEST,
            message="The ingestion job was not found.",
            status_code=404,
        )
    return result


@router.post("/sources/sync", status_code=202)
async def sync_sources(
    body: SourceSyncRequest,
    principal: Annotated[Principal, Depends(current_principal)],
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
) -> dict[str, object]:
    principal.require(Permission.INGESTION_WRITE)
    if body.mode == "full_reconcile" and "admin" not in principal.roles:
        raise forbidden("Full reconciliation requires the admin role.")
    return await _operations(dependencies).sync(principal, body.model_dump(mode="json"))
