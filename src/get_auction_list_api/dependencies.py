"""Explicit dependency container and readiness probe contracts."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from fastapi import Request

from get_auction_list_api.auth import Principal
from get_auction_list_api.config import Settings
from get_auction_list_api.database import AsyncDatabase
from get_auction_list_api.graph.workflow import ControlledAgentGraph
from get_auction_list_api.observability.logging import get_logger
from get_auction_list_api.schemas import FeedbackRequest

ReadinessProbe = Callable[[], Awaitable[bool]]


class Authenticator(Protocol):
    async def validate(self, token: str) -> Principal: ...


class FeedbackStore(Protocol):
    async def record(
        self,
        principal: Principal,
        feedback: FeedbackRequest,
    ) -> tuple[UUID, bool]: ...


class IngestionOperations(Protocol):
    async def enqueue(
        self, principal: Principal, payload: dict[str, object]
    ) -> dict[str, object]: ...

    async def status(self, principal: Principal, job_id: UUID) -> dict[str, object] | None: ...

    async def sync(self, principal: Principal, payload: dict[str, object]) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class NamedReadinessProbe:
    """A bounded dependency check registered by an integration module."""

    name: str
    check: ReadinessProbe


@dataclass(frozen=True, slots=True)
class AppDependencies:
    """Dependencies shared by request handlers through application state."""

    settings: Settings
    database: AsyncDatabase | None = None
    readiness_probes: Sequence[NamedReadinessProbe] = ()
    readiness_timeout_seconds: float = 1.0
    authenticator: Authenticator | None = None
    graph: ControlledAgentGraph | None = None
    feedback_store: FeedbackStore | None = None
    ingestion_operations: IngestionOperations | None = None

    async def readiness(self) -> dict[str, bool]:
        """Execute registered checks concurrently with a shared timeout."""

        probes = list(self.readiness_probes)
        if self.database is not None:
            probes.append(NamedReadinessProbe(name="database", check=self.database.ping))
        if not probes:
            return {}

        async def run(probe: NamedReadinessProbe) -> tuple[str, bool]:
            try:
                result = await asyncio.wait_for(
                    probe.check(),
                    timeout=self.readiness_timeout_seconds,
                )
            except Exception as error:
                get_logger().warning(
                    "readiness_probe_failed",
                    probe=probe.name,
                    exception_type=type(error).__name__,
                )
                result = False
            return probe.name, result

        return dict(await asyncio.gather(*(run(probe) for probe in probes)))


def get_dependencies(request: Request) -> AppDependencies:
    """Resolve the app-scoped dependency container."""

    dependencies: AppDependencies = request.app.state.dependencies
    return dependencies
