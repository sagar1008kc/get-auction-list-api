"""Request-scoped graph identity that is deliberately excluded from checkpoints."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from get_auction_list_api.auth import Principal


@dataclass(frozen=True, slots=True)
class GraphRequestContext:
    principal: Principal
    trace_id: str


_CURRENT: ContextVar[GraphRequestContext | None] = ContextVar(
    "graph_request_context",
    default=None,
)


def current_graph_context() -> GraphRequestContext:
    value = _CURRENT.get()
    if value is None:
        raise RuntimeError("Graph request context is not bound.")
    return value


@contextmanager
def bind_graph_context(principal: Principal, trace_id: str) -> Iterator[None]:
    token = _CURRENT.set(GraphRequestContext(principal=principal, trace_id=trace_id))
    try:
        yield
    finally:
        _CURRENT.reset(token)
