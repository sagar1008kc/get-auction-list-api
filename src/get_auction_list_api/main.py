"""FastAPI application factory."""

import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from get_auction_list_api.api.health import router as health_router
from get_auction_list_api.api.metrics import (
    DB_POOL,
    HTTP_DURATION,
    HTTP_REQUESTS,
    METHODS,
    pool_sizes,
    route_label,
)
from get_auction_list_api.api.metrics import (
    router as metrics_router,
)
from get_auction_list_api.api.v1 import chat_router, operations_router
from get_auction_list_api.auth import SupabaseJWTValidator
from get_auction_list_api.config import Settings, get_settings
from get_auction_list_api.database import AsyncDatabase
from get_auction_list_api.dependencies import AppDependencies
from get_auction_list_api.errors import install_error_handlers
from get_auction_list_api.graph import ControlledAgentGraph
from get_auction_list_api.graph.adapters import build_graph_services
from get_auction_list_api.graph.checkpoint import PostgresCheckpointRuntime
from get_auction_list_api.graph.context import current_graph_context
from get_auction_list_api.graph.llm import OpenAIGroundedSynthesizer, OpenAIIntentClassifier
from get_auction_list_api.ingestion.operations import PostgresIngestionOperations
from get_auction_list_api.llm import (
    OpenAIEmbeddingProvider,
    OpenAIStructuredModel,
    create_embedding_transport,
    create_openai_client,
)
from get_auction_list_api.middleware import install_security_middleware
from get_auction_list_api.observability.langfuse import LangfuseTelemetry
from get_auction_list_api.observability.logging import configure_logging, get_logger
from get_auction_list_api.observability.telemetry import configure_telemetry
from get_auction_list_api.public_records import (
    ApprovedHttpClient,
    PublicRecordsService,
    create_mcp_server,
)
from get_auction_list_api.rag.repository import PostgresRAGRepository
from get_auction_list_api.rag.retrieval import HybridRetriever
from get_auction_list_api.repositories import PostgresAuctionRepository
from get_auction_list_api.runtime import InMemoryFeedbackStore
from get_auction_list_api.tools.auction_search import AuctionSearchService


def _safe_uuid(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    try:
        return str(UUID(value))
    except ValueError:
        return str(uuid4())


def _trace_id(traceparent: str | None) -> str:
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) == 4 and len(parts[1]) == 32:
            try:
                int(parts[1], 16)
            except ValueError:
                pass
            else:
                return parts[1].lower()
    return uuid4().hex


def create_app(
    settings: Settings | None = None,
    dependencies: AppDependencies | None = None,
) -> FastAPI:
    """Build an application with explicitly injectable dependencies."""

    resolved_settings = settings or (dependencies.settings if dependencies else get_settings())
    database = None
    openai_client = None
    checkpoint_runtime = None
    if dependencies is None and resolved_settings.database_url is not None:
        database = AsyncDatabase(
            resolved_settings.database_url.get_secret_value(),
            min_size=resolved_settings.database_pool_min_size,
            max_size=resolved_settings.database_pool_max_size,
            command_timeout=resolved_settings.database_command_timeout_seconds,
        )
    public_client = ApprovedHttpClient(
        resolved_settings.approved_source_hosts,
        timeout_seconds=resolved_settings.public_http_timeout_seconds,
        max_attempts=resolved_settings.public_http_max_attempts,
        max_response_bytes=resolved_settings.public_http_max_response_bytes,
        cache_ttl_seconds=resolved_settings.public_http_cache_ttl_seconds,
    )
    public_service = PublicRecordsService(public_client)
    if dependencies is None:
        langfuse = LangfuseTelemetry(resolved_settings)
        if resolved_settings.checkpoint_enabled and resolved_settings.database_url is not None:
            checkpoint_runtime = PostgresCheckpointRuntime(
                resolved_settings.database_url.get_secret_value(),
                min_size=resolved_settings.checkpoint_pool_min_size,
                max_size=resolved_settings.checkpoint_pool_max_size,
                setup_on_start=resolved_settings.checkpoint_setup_on_start,
            )
        authenticator = None
        if (
            resolved_settings.jwks_url is not None
            and resolved_settings.resolved_jwt_issuer is not None
        ):
            authenticator = SupabaseJWTValidator(
                jwks_url=resolved_settings.jwks_url,
                issuer=resolved_settings.resolved_jwt_issuer,
                audience=resolved_settings.jwt_audience,
                cache_ttl_seconds=resolved_settings.jwks_cache_ttl_seconds,
            )
        embeddings = None
        retriever = None
        auction_service = None
        classifier = None
        synthesizer = None
        if resolved_settings.openai_api_key is not None:
            openai_client = create_openai_client(
                api_key=resolved_settings.openai_api_key.get_secret_value(),
                base_url=resolved_settings.openai_base_url,
                timeout_seconds=resolved_settings.openai_timeout_seconds,
                max_retries=resolved_settings.openai_max_retries,
            )
            structured_model = OpenAIStructuredModel(
                model=resolved_settings.openai_chat_model,
                client=openai_client,
            )
            classifier = OpenAIIntentClassifier(structured_model)
            synthesizer = OpenAIGroundedSynthesizer(structured_model)
        if database is not None:
            auction_service = AuctionSearchService(PostgresAuctionRepository(database))
            if openai_client is not None:
                embeddings = OpenAIEmbeddingProvider(
                    model=resolved_settings.openai_embedding_model,
                    dimensions=resolved_settings.openai_embedding_dimensions,
                    transport=create_embedding_transport(openai_client),
                )
                retriever = HybridRetriever(
                    PostgresRAGRepository(
                        database,
                        dimensions=resolved_settings.openai_embedding_dimensions,
                    )
                )
        graph_services = build_graph_services(
            embeddings=embeddings,
            retriever=retriever,
            auction_service=auction_service,
            public_service=public_service,
            principal_resolver=lambda: current_graph_context().principal,
            trace_id_resolver=lambda: current_graph_context().trace_id,
            classifier=classifier,
            synthesizer=synthesizer,
        )
        resolved_dependencies = AppDependencies(
            settings=resolved_settings,
            database=database,
            authenticator=authenticator,
            graph=ControlledAgentGraph(
                services=graph_services,
                checkpointer=checkpoint_runtime.saver if checkpoint_runtime else None,
                thread_owner_store=checkpoint_runtime,
                telemetry=langfuse,
            ),
            feedback_store=InMemoryFeedbackStore(),
            ingestion_operations=(
                PostgresIngestionOperations(database) if database is not None else None
            ),
        )
    else:
        resolved_dependencies = dependencies
        langfuse = LangfuseTelemetry(resolved_settings)
    mcp_server = create_mcp_server(public_service)
    mcp_app = mcp_server.streamable_http_app()
    configure_logging(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.dependencies = resolved_dependencies
        if checkpoint_runtime is not None:
            await checkpoint_runtime.start()
        if resolved_dependencies.database is not None:
            await resolved_dependencies.database.connect()
            for state, value in pool_sizes(resolved_dependencies.database.pool).items():
                DB_POOL.labels(state).set(value)
        get_logger().info(
            "service_started",
            service=resolved_settings.service_name,
            version=resolved_settings.service_version,
            environment=resolved_settings.environment,
        )
        async with mcp_server.session_manager.run():
            try:
                yield
            finally:
                if resolved_dependencies.database is not None:
                    await resolved_dependencies.database.close()
                    for state in ("open", "idle", "in_use"):
                        DB_POOL.labels(state).set(0)
                if openai_client is not None:
                    await openai_client.close()
                if checkpoint_runtime is not None:
                    await checkpoint_runtime.close()
                langfuse.flush()
                get_logger().info("service_stopped", service=resolved_settings.service_name)

    app = FastAPI(
        title="GetAuctionList API",
        version=resolved_settings.service_version,
        lifespan=lifespan,
    )
    app.state.dependencies = resolved_dependencies
    app.state.langfuse = langfuse
    install_security_middleware(
        app,
        cors_origins=resolved_settings.cors_origins,
        trusted_hosts=resolved_settings.trusted_hosts,
        max_body_bytes=resolved_settings.max_request_body_bytes,
        rate_limit_requests=resolved_settings.rate_limit_requests,
        rate_limit_window_seconds=resolved_settings.rate_limit_window_seconds,
        concurrency_limit=resolved_settings.concurrency_limit,
        timeout_seconds=resolved_settings.request_timeout_seconds,
    )

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        request_id = _safe_uuid(request.headers.get("X-Request-ID"))
        correlation_id = _safe_uuid(request.headers.get("X-Correlation-ID"))
        trace_id = _trace_id(request.headers.get("traceparent"))
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        request.state.trace_id = trace_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
        )
        if request.url.path.startswith("/mcp") and resolved_settings.internal_mcp_token is not None:
            expected = f"Bearer {resolved_settings.internal_mcp_token.get_secret_value()}"
            supplied = request.headers.get("Authorization", "")
            if not secrets.compare_digest(supplied, expected):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {"code": "UNAUTHORIZED", "message": "Authentication required."}
                    },
                    headers={"Cache-Control": "no-store"},
                )
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Trace-ID"] = trace_id
            response.headers["Cache-Control"] = "no-store"
            return response
        finally:
            route = request.scope.get("route")
            path_template = getattr(route, "path", None)
            method = request.method if request.method in METHODS else "other"
            label = route_label(request.url.path, path_template)
            HTTP_REQUESTS.labels(method, label, f"{status_code // 100}xx").inc()
            HTTP_DURATION.labels(method, label).observe(time.perf_counter() - started)
            structlog.contextvars.clear_contextvars()

    install_error_handlers(app)
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(chat_router)
    app.include_router(operations_router)
    app.mount("/mcp", mcp_app, name="mcp")
    configure_telemetry(app, resolved_settings)
    return app


app = create_app()
