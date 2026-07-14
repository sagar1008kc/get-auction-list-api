"""Authenticated chat, SSE, and feedback endpoints."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from get_auction_list_api.api.metrics import FEEDBACK, STREAMS, TTFT
from get_auction_list_api.api.security import current_principal
from get_auction_list_api.auth import Permission, Principal
from get_auction_list_api.dependencies import AppDependencies, get_dependencies
from get_auction_list_api.errors import AppError, ErrorCode
from get_auction_list_api.graph.context import bind_graph_context
from get_auction_list_api.graph.state import AgentState
from get_auction_list_api.graph.workflow import ControlledAgentGraph
from get_auction_list_api.observability.logging import get_logger
from get_auction_list_api.schemas import (
    ChatRequest,
    FeedbackRequest,
    FeedbackResponse,
    FinalResponse,
)

router = APIRouter(prefix="/v1", tags=["chat"])


def _state(request: Request, body: ChatRequest, principal: Principal) -> AgentState:
    thread_id = body.thread_id or uuid4()
    return AgentState(
        request_id=request.state.request_id,
        correlation_id=request.state.correlation_id,
        trace_id=request.state.trace_id,
        run_id=str(uuid4()),
        thread_id=str(thread_id),
        assistant_message_id=str(uuid4()),
        user_id=str(principal.user_id),
        message=body.message,
        locale=body.locale,
        timezone=body.timezone,
        retry_budget={"classifier": 1, "retrieval": 1, "tools": 2},
    )


def _graph(dependencies: AppDependencies) -> ControlledAgentGraph:
    if dependencies.graph is None:
        raise AppError(
            code=ErrorCode.NOT_READY,
            message="Chat is not ready.",
            status_code=503,
            retryable=True,
        )
    return dependencies.graph


@router.post("/chat", response_model=FinalResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    principal: Annotated[Principal, Depends(current_principal)],
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
) -> FinalResponse:
    principal.require(Permission.TOOL_EXECUTE)
    with bind_graph_context(principal, request.state.trace_id):
        return await _graph(dependencies).run(_state(request, body, principal))


def _event(event_id: int, event: str, state: AgentState, data: dict[str, object]) -> str:
    envelope = {
        "schema_version": "1.0",
        "event_id": str(event_id),
        "event": event,
        "occurred_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_id": state["request_id"],
        "correlation_id": state["correlation_id"],
        "trace_id": state["trace_id"],
        "run_id": state["run_id"],
        "thread_id": state["thread_id"],
        "data": data,
    }
    return (
        f"id: {event_id}\nevent: {event}\ndata: {json.dumps(envelope, separators=(',', ':'))}\n\n"
    )


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    principal: Annotated[Principal, Depends(current_principal)],
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
) -> StreamingResponse:
    principal.require(Permission.TOOL_EXECUTE)
    graph = _graph(dependencies)
    state = _state(request, body, principal)

    async def stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        event_id = 1
        first = True
        with bind_graph_context(principal, state["trace_id"]):
            try:
                async for item in graph.astream(state):
                    if await request.is_disconnected():
                        return
                    event = str(item.get("event", ""))
                    data = item.get("data")
                    if not isinstance(data, dict):
                        data = {}
                    if first:
                        TTFT.observe(time.perf_counter() - started)
                        first = False
                    # Upgrade lightweight answer.completed payloads with full response when
                    # the graph already produced FinalResponse in a later values tick.
                    yield _event(event_id, event, state, data)
                    event_id += 1
                STREAMS.labels("success").inc()
            except asyncio.CancelledError:
                STREAMS.labels("cancelled").inc()
                raise
            except Exception:
                STREAMS.labels("server_error").inc()
                get_logger().exception(
                    "chat_stream_failed",
                    run_id=state["run_id"],
                    request_id=state["request_id"],
                    correlation_id=state["correlation_id"],
                )
                yield _event(
                    event_id,
                    "run.failed",
                    state,
                    {
                        "error": {
                            "code": "CHAT_RUN_FAILED",
                            "message": "The chat run could not be completed.",
                            "retryable": True,
                        }
                    },
                )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def feedback(
    body: FeedbackRequest,
    request: Request,
    response: Response,
    principal: Annotated[Principal, Depends(current_principal)],
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
) -> FeedbackResponse:
    if dependencies.feedback_store is None:
        raise AppError(
            code=ErrorCode.NOT_READY,
            message="Feedback storage is not ready.",
            status_code=503,
            retryable=True,
        )
    feedback_id, created = await dependencies.feedback_store.record(principal, body)
    FEEDBACK.labels(body.rating, str(body.cta_clicked).lower()).inc()
    request.app.state.langfuse.score(
        trace_id=body.run_id.hex,
        name="user_feedback",
        value=1.0 if body.rating == "up" else 0.0,
        comment=body.comment,
    )
    if body.cta_clicked:
        request.app.state.langfuse.score(
            trace_id=body.run_id.hex,
            name="cta_clicked",
            value=1.0,
        )
    if not created:
        response.status_code = 200
    return FeedbackResponse(feedback_id=feedback_id, created=created)
