"""Prometheus metrics with an explicit low-cardinality label contract."""

from collections.abc import Mapping
from typing import Final

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

router = APIRouter(tags=["operations"])

ROUTES: Final = frozenset(
    {
        "/health/live",
        "/health/ready",
        "/metrics",
        "/v1/chat",
        "/v1/chat/stream",
        "/v1/feedback",
        "/v1/ingestion/jobs",
        "/v1/ingestion/jobs/{job_id}",
        "/v1/sources/sync",
        "/mcp",
    }
)
METHODS: Final = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
OUTCOMES: Final = frozenset({"success", "client_error", "server_error", "cancelled"})
GRAPH_NODES: Final = frozenset(
    {
        "initialize_run",
        "validation",
        "routing",
        "extraction",
        "knowledge_rag",
        "sql_auction_search",
        "mcp_public_tools",
        "unsupported",
        "evidence_correlation",
        "grounding_verification",
        "compliance_disclaimer",
        "synthesis",
        "trace_finalization",
    }
)
SOURCES: Final = frozenset({"approved_policy", "auction_index", "county", "wcad", "unknown"})


def bounded(value: str, allowed: frozenset[str]) -> str:
    """Collapse any unapproved label value instead of creating a new series."""

    return value if value in allowed else "other"


HTTP_REQUESTS = Counter(
    "get_auction_list_http_requests_total",
    "Completed HTTP requests.",
    ("method", "route", "status_class"),
)
HTTP_DURATION = Histogram(
    "get_auction_list_http_request_duration_seconds",
    "HTTP request duration.",
    ("method", "route"),
)
STREAMS = Counter(
    "get_auction_list_streams_total",
    "Chat streams by terminal outcome.",
    ("outcome",),
)
TTFT = Histogram(
    "get_auction_list_stream_ttft_seconds",
    "Time from stream acceptance to first public event.",
)
GRAPH_NODE_DURATION = Histogram(
    "get_auction_list_graph_node_duration_seconds",
    "Graph node duration.",
    ("node", "outcome"),
)
RETRIEVAL_REQUESTS = Counter(
    "get_auction_list_retrieval_requests_total",
    "Hybrid retrieval requests.",
    ("outcome",),
)
RETRIEVAL_RESULTS = Histogram(
    "get_auction_list_retrieval_results",
    "Approved evidence items selected by retrieval.",
)
AUCTION_SEARCHES = Counter(
    "get_auction_list_auction_searches_total",
    "Structured auction searches.",
    ("outcome",),
)
TOOL_CALLS = Counter(
    "get_auction_list_tool_calls_total",
    "MCP and direct public-record tool calls.",
    ("tool", "outcome"),
)
SOURCE_HEALTH = Gauge(
    "get_auction_list_source_health",
    "Last observed source health (1 healthy, 0 unhealthy).",
    ("source",),
)
INGESTION_JOBS = Counter(
    "get_auction_list_ingestion_jobs_total",
    "Ingestion jobs and pipeline runs.",
    ("operation", "outcome"),
)
INGESTION_DURATION = Histogram(
    "get_auction_list_ingestion_duration_seconds",
    "Ingestion pipeline duration.",
    ("outcome",),
)
DB_POOL = Gauge(
    "get_auction_list_db_pool_connections",
    "PostgreSQL pool connections.",
    ("state",),
)
MODEL_TOKENS = Counter(
    "get_auction_list_model_tokens_total",
    "Model tokens consumed.",
    ("model", "direction"),
)
MODEL_COST = Counter(
    "get_auction_list_model_cost_usd_total",
    "Estimated model cost in USD.",
    ("model",),
)
FEEDBACK = Counter(
    "get_auction_list_feedback_total",
    "User feedback events.",
    ("rating", "cta_clicked"),
)

_TOOLS: Final = frozenset(
    {
        "county.discover_trustee_sale_sources",
        "county.search_foreclosure_records",
        "county.get_foreclosure_notice",
        "wcad.search_property",
        "wcad.get_property_details",
        "property.correlate_records",
    }
)
_MODELS: Final = frozenset({"classifier", "embedding", "synthesis", "unknown"})


def route_label(path: str, path_template: str | None = None) -> str:
    candidate = path_template or path
    if candidate.startswith("/mcp"):
        return "/mcp"
    return bounded(candidate, ROUTES)


def tool_label(tool: str) -> str:
    return bounded(tool, _TOOLS)


def source_label(source: str) -> str:
    return bounded(source, SOURCES)


def record_model_usage(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0,
) -> None:
    label = bounded(model, _MODELS)
    if input_tokens > 0:
        MODEL_TOKENS.labels(label, "input").inc(input_tokens)
    if output_tokens > 0:
        MODEL_TOKENS.labels(label, "output").inc(output_tokens)
    if cost_usd > 0:
        MODEL_COST.labels(label).inc(cost_usd)


def pool_sizes(pool: object) -> Mapping[str, float]:
    """Read only aggregate asyncpg pool state; never connection identifiers."""

    size = getattr(pool, "get_size", lambda: 0)()
    idle = getattr(pool, "get_idle_size", lambda: 0)()
    return {"open": float(size), "idle": float(idle), "in_use": float(max(0, size - idle))}


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
