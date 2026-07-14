"""Async PostgreSQL infrastructure with narrow, injectable query contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, Protocol, Self, cast

import asyncpg  # type: ignore[import-untyped]

DatabaseRow = Mapping[str, Any]


class QueryExecutor(Protocol):
    async def fetch(self, query: str, *args: object) -> Sequence[DatabaseRow]: ...

    async def fetchrow(self, query: str, *args: object) -> DatabaseRow | None: ...

    async def execute(self, query: str, *args: object) -> str: ...


class Transaction(AbstractAsyncContextManager[QueryExecutor]):
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._connection: Any | None = None
        self._transaction: Any | None = None

    async def __aenter__(self) -> QueryExecutor:
        self._connection = await self._pool.acquire()
        self._transaction = self._connection.transaction()
        await self._transaction.start()
        return cast(QueryExecutor, self._connection)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._connection is None or self._transaction is None:
            return None
        try:
            if exc_type is None:
                await self._transaction.commit()
            else:
                await self._transaction.rollback()
        finally:
            await self._pool.release(self._connection)
        return None


class AsyncDatabase:
    """Own an asyncpg pool; callers never interpolate values into SQL."""

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        command_timeout: float = 10,
    ) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout
        self._pool: Any | None = None

    async def connect(self) -> Self:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                command_timeout=self._command_timeout,
                server_settings={"application_name": "get-auction-list-api"},
            )
        return self

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("Database pool is not connected.")
        return self._pool

    @property
    def pool(self) -> Any:
        """Expose aggregate pool instrumentation without exposing the DSN."""

        return self._require_pool()

    async def fetch(self, query: str, *args: object) -> Sequence[DatabaseRow]:
        return cast(Sequence[DatabaseRow], await self._require_pool().fetch(query, *args))

    async def fetchrow(self, query: str, *args: object) -> DatabaseRow | None:
        return cast(DatabaseRow | None, await self._require_pool().fetchrow(query, *args))

    async def execute(self, query: str, *args: object) -> str:
        return cast(str, await self._require_pool().execute(query, *args))

    def transaction(self) -> Transaction:
        return Transaction(self._require_pool())

    async def ping(self) -> bool:
        value = cast(object, await self._require_pool().fetchval("select 1"))
        return value == 1
