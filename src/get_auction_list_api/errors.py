"""Typed, client-safe application errors."""

from enum import StrEnum
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from get_auction_list_api.observability.logging import get_logger


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    SERVER_BUSY = "SERVER_BUSY"
    NOT_READY = "NOT_READY"
    PUBLIC_SOURCE_TIMEOUT = "PUBLIC_SOURCE_TIMEOUT"
    PUBLIC_SOURCE_UNAVAILABLE = "PUBLIC_SOURCE_UNAVAILABLE"
    PUBLIC_SOURCE_CONTRACT_CHANGED = "PUBLIC_SOURCE_CONTRACT_CHANGED"
    GROUNDING_FAILED = "GROUNDING_FAILED"
    CANCELLED = "CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool
    request_id: str
    correlation_id: str
    trace_id: str
    details: dict[str, str | int | bool | None] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class AppError(Exception):
    """An expected error safe to map to the public API contract."""

    def __init__(
        self,
        *,
        code: ErrorCode,
        message: str,
        status_code: int,
        retryable: bool = False,
        details: dict[str, str | int | bool | None] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details


def _context(request: Request) -> tuple[str, str, str]:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    correlation_id = getattr(request.state, "correlation_id", request_id)
    trace_id = getattr(request.state, "trace_id", "")
    return request_id, correlation_id, trace_id


def _response(request: Request, error: AppError) -> JSONResponse:
    request_id, correlation_id, trace_id = _context(request)
    body = ErrorResponse(
        error=ErrorBody(
            code=error.code,
            message=error.message,
            retryable=error.retryable,
            request_id=request_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            details=error.details,
        )
    )
    return JSONResponse(status_code=error.status_code, content=body.model_dump(mode="json"))


def install_error_handlers(app: FastAPI) -> None:
    """Install safe mappings without exposing stack traces or request values."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
        return _response(request, error)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        fields = sorted({".".join(str(part) for part in item["loc"]) for item in error.errors()})
        return _response(
            request,
            AppError(
                code=ErrorCode.INVALID_REQUEST,
                message="Request validation failed.",
                status_code=422,
                details={"fields": ",".join(fields)},
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        get_logger().error(
            "unhandled_exception",
            exception_type=type(error).__name__,
            request_id=getattr(request.state, "request_id", None),
        )
        return _response(
            request,
            AppError(
                code=ErrorCode.INTERNAL_ERROR,
                message="An internal error occurred.",
                status_code=500,
                retryable=True,
            ),
        )
