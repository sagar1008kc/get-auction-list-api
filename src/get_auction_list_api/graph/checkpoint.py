"""Lifecycle and ownership controls for LangGraph PostgreSQL checkpoints."""

from typing import Any, cast
from uuid import UUID

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from get_auction_list_api.auth import forbidden


class PostgresCheckpointRuntime:
    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 4,
        setup_on_start: bool = True,
    ) -> None:
        self._pool = cast(
            AsyncConnectionPool[AsyncConnection[dict[str, Any]]],
            AsyncConnectionPool(
                conninfo=dsn,
                min_size=min_size,
                max_size=max_size,
                open=False,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "options": "-c search_path=langgraph,public",
                },
            ),
        )
        self.saver = AsyncPostgresSaver(self._pool)
        self._setup_on_start = setup_on_start

    async def start(self) -> None:
        await self._pool.open()
        await self._pool.wait()
        if self._setup_on_start:
            await self.saver.setup()

    async def close(self) -> None:
        await self._pool.close()

    async def ensure_owner(self, *, user_id: str, thread_id: str) -> None:
        user_uuid = UUID(user_id)
        thread_uuid = UUID(thread_id)
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                insert into checkpoint_thread_owners (thread_id, user_id, conversation_thread_id)
                values (%s, %s, null)
                on conflict (thread_id) do nothing
                """,
                (str(thread_uuid), user_uuid),
            )
            cursor = await connection.execute(
                "select user_id from checkpoint_thread_owners where thread_id = %s",
                (str(thread_uuid),),
            )
            row = await cursor.fetchone()
        if row is None or row["user_id"] != user_uuid:
            raise forbidden("Conversation thread belongs to a different user.")
