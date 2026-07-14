"""Failure-isolated OpenTelemetry setup and safe span helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from get_auction_list_api.config import Settings
from get_auction_list_api.observability.logging import get_logger, redact

_configured = False


def configure_telemetry(app: Any, settings: Settings) -> None:
    """Install W3C-propagating instrumentation; exporter faults stay off request paths."""

    global _configured
    if _configured:
        FastAPIInstrumentor.instrument_app(app)
        return
    try:
        if settings.otel_enabled:
            resource = Resource.create(
                {
                    "service.name": settings.service_name,
                    "service.version": settings.service_version,
                    "deployment.environment.name": settings.environment,
                }
            )
            provider = TracerProvider(resource=resource)
            if settings.otel_exporter_otlp_endpoint:
                provider.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
                    )
                )
            trace.set_tracer_provider(provider)
        propagate.set_global_textmap(propagate.get_global_textmap())
        HTTPXClientInstrumentor().instrument()
        AsyncPGInstrumentor().instrument()  # type: ignore[no-untyped-call]
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="/health/live,/metrics",
            http_capture_headers_server_request=[],
            http_capture_headers_server_response=[],
        )
        _configured = True
    except Exception as error:
        get_logger().warning(
            "telemetry_initialization_failed",
            exception_type=type(error).__name__,
        )


def safe_attributes(values: Mapping[str, object]) -> dict[str, str | int | float | bool]:
    """Keep only scalar, redacted values acceptable as span attributes."""

    result: dict[str, str | int | float | bool] = {}
    for key, value in redact(values).items():
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result


def tracer() -> trace.Tracer:
    return trace.get_tracer("get_auction_list_api")
