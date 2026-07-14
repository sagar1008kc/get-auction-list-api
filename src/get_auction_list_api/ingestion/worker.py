"""Durable ingestion worker runtime with an explicit source-handler boundary."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import signal
from collections.abc import Awaitable, Callable, Mapping
from datetime import timedelta
from typing import cast
from uuid import uuid4

from get_auction_list_api.config import Settings, get_settings
from get_auction_list_api.database import AsyncDatabase
from get_auction_list_api.ingestion.queue import IngestionJob, PostgresJobQueue
from get_auction_list_api.observability.logging import configure_logging, get_logger

JobHandler = Callable[[IngestionJob], Awaitable[Mapping[str, object]]]


def load_handler(reference: str) -> JobHandler:
    """Load ``package.module:async_callable`` from trusted deployment configuration."""

    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("ingestion_handler must use package.module:async_callable syntax.")
    candidate = getattr(importlib.import_module(module_name), attribute, None)
    if candidate is None or not callable(candidate) or not inspect.iscoroutinefunction(candidate):
        raise ValueError("ingestion_handler must identify an async callable.")
    return cast(JobHandler, candidate)


async def _handle_with_heartbeat(
    queue: PostgresJobQueue,
    handler: JobHandler,
    job: IngestionJob,
    *,
    lease_token: str,
    heartbeat_seconds: float,
) -> Mapping[str, object]:
    async def invoke() -> Mapping[str, object]:
        return await handler(job)

    task: asyncio.Task[Mapping[str, object]] = asyncio.create_task(invoke())
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
            if done:
                return await task
            if not await queue.heartbeat(job.id, lease_token=lease_token):
                task.cancel()
                raise RuntimeError("Ingestion lease was lost.")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def run_worker(settings: Settings, handler: JobHandler, stop: asyncio.Event) -> None:
    """Claim jobs until stopped, preserving retry and lease ownership semantics."""

    if settings.database_url is None:
        raise ValueError("The ingestion worker requires GET_AUCTION_LIST_DATABASE_URL.")
    database = AsyncDatabase(
        settings.database_url.get_secret_value(),
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
        command_timeout=settings.database_command_timeout_seconds,
    )
    await database.connect()
    queue = PostgresJobQueue(database)
    logger = get_logger()
    retry_delay = timedelta(seconds=settings.ingestion_retry_seconds)
    try:
        await queue.recover_stale(
            stale_after=timedelta(seconds=settings.ingestion_stale_seconds),
            max_attempts=settings.ingestion_max_attempts,
            retry_delay=retry_delay,
        )
        logger.info("ingestion_worker_started")
        while not stop.is_set():
            lease_token = uuid4().hex
            jobs = await queue.claim(lease_token=lease_token, limit=1)
            if not jobs:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=settings.ingestion_poll_seconds)
                except TimeoutError:
                    pass
                continue
            job = jobs[0]
            try:
                output = await _handle_with_heartbeat(
                    queue,
                    handler,
                    job,
                    lease_token=lease_token,
                    heartbeat_seconds=settings.ingestion_heartbeat_seconds,
                )
                if await queue.succeed(job.id, lease_token=lease_token, output=output) is None:
                    raise RuntimeError("Ingestion completion lost its lease.")
                logger.info("ingestion_job_succeeded", job_id=str(job.id))
            except Exception as error:
                await queue.fail(
                    job.id,
                    lease_token=lease_token,
                    max_attempts=settings.ingestion_max_attempts,
                    retry_delay=retry_delay,
                    error_code=type(error).__name__.casefold()[:80],
                    redacted_detail="The configured ingestion handler failed.",
                )
                logger.exception("ingestion_job_failed", job_id=str(job.id))
    finally:
        await database.close()
        logger.info("ingestion_worker_stopped")


async def _main() -> None:
    settings = get_settings()
    configure_logging(settings)
    if settings.ingestion_handler is None:
        raise SystemExit(
            "Set GET_AUCTION_LIST_INGESTION_HANDLER to package.module:async_callable; "
            "no jobs were claimed."
        )
    handler = load_handler(settings.ingestion_handler)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(handled_signal, stop.set)
    await run_worker(settings, handler, stop)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
