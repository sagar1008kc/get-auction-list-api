"""Optional Langfuse sink with redaction and complete failure isolation."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from types import TracebackType
from typing import Any, Literal, cast

from get_auction_list_api.config import Settings
from get_auction_list_api.observability.logging import get_logger, redact

ObservationType = Literal[
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
    "generation",
    "embedding",
]


class _SafeObservation(AbstractContextManager[Any]):
    def __init__(self, inner: AbstractContextManager[Any]) -> None:
        self._inner = inner

    def __enter__(self) -> Any:
        try:
            return self._inner.__enter__()
        except Exception as error:
            get_logger().warning(
                "langfuse_observation_enter_failed",
                exception_type=type(error).__name__,
            )
            return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            return bool(self._inner.__exit__(exc_type, exc_value, traceback))
        except Exception as error:
            get_logger().warning(
                "langfuse_observation_exit_failed",
                exception_type=type(error).__name__,
            )
            return False


def _mask(value: Any) -> Any:
    return redact(value)


class LangfuseTelemetry:
    """Thin adapter; disabled and unhealthy exporters always degrade to no-ops."""

    def __init__(self, settings: Settings) -> None:
        self._client: Any | None = None
        if not settings.langfuse_enabled:
            return
        if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
            get_logger().warning("langfuse_disabled_missing_credentials")
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                host=settings.langfuse_host,
                environment=settings.environment,
                release=settings.service_version,
                sample_rate=settings.langfuse_sample_rate,
                timeout=2,
                mask=cast(Any, _mask),
            )
        except Exception as error:
            get_logger().warning(
                "langfuse_initialization_failed",
                exception_type=type(error).__name__,
            )

    def observe(
        self,
        name: str,
        *,
        kind: ObservationType = "span",
        metadata: dict[str, object] | None = None,
        trace_id: str | None = None,
        model: str | None = None,
        usage: dict[str, int] | None = None,
        cost: dict[str, float] | None = None,
    ) -> AbstractContextManager[Any]:
        if self._client is None:
            return nullcontext()
        try:
            return _SafeObservation(
                cast(
                    AbstractContextManager[Any],
                    self._client.start_as_current_observation(
                        name=name,
                        as_type=kind,
                        trace_context={"trace_id": trace_id} if trace_id else None,
                        input=None,
                        output=None,
                        metadata=redact(metadata or {}),
                        model=model,
                        usage_details=usage,
                        cost_details=cost,
                    ),
                )
            )
        except Exception as error:
            get_logger().warning(
                "langfuse_observation_failed",
                exception_type=type(error).__name__,
            )
            return nullcontext()

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        comment: str | None = None,
    ) -> None:
        if self._client is None:
            return
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=str(redact(comment)) if comment else None,
            )
        except Exception as error:
            get_logger().warning(
                "langfuse_score_failed",
                exception_type=type(error).__name__,
            )

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as error:
            get_logger().warning(
                "langfuse_flush_failed",
                exception_type=type(error).__name__,
            )
