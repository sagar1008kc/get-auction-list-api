"""Bounded request middleware with no dependency on application handlers."""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                oversized = int(content_length) > self.max_bytes
            except ValueError:
                oversized = True
            if oversized:
                await JSONResponse({"error": {"code": "REQUEST_TOO_LARGE"}}, status_code=413)(
                    scope, receive, send
                )
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLarge:
            await JSONResponse({"error": {"code": "REQUEST_TOO_LARGE"}}, status_code=413)(
                scope, receive, send
            )


class _RequestTooLarge(Exception):
    pass


class TimeoutMiddleware:
    def __init__(self, app: ASGIApp, *, timeout_seconds: float) -> None:
        self.app = app
        self.timeout_seconds = timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await self.app(scope, receive, send)
        except TimeoutError:
            await JSONResponse({"error": {"code": "REQUEST_TIMEOUT"}}, status_code=504)(
                scope, receive, send
            )


class ConcurrencyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, limit: int) -> None:
        self.app = app
        self._semaphore = asyncio.Semaphore(limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if self._semaphore.locked():
            await JSONResponse({"error": {"code": "SERVER_BUSY"}}, status_code=503)(
                scope, receive, send
            )
            return
        async with self._semaphore:
            await self.app(scope, receive, send)


class RateLimitMiddleware:
    """Per-process fixed-window guard; distributed deployments replace its storage."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self._requests = requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._requests_by_client: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        key = client[0] if client else "unknown"
        now = self._clock()
        async with self._lock:
            entries = self._requests_by_client[key]
            while entries and entries[0] <= now - self._window_seconds:
                entries.popleft()
            if len(entries) >= self._requests:
                await JSONResponse(
                    {"error": {"code": "RATE_LIMITED"}},
                    status_code=429,
                    headers={"Retry-After": str(max(1, round(self._window_seconds)))},
                )(scope, receive, send)
                return
            entries.append(now)
        await self.app(scope, receive, send)


def install_security_middleware(
    app: FastAPI,
    *,
    cors_origins: tuple[str, ...],
    trusted_hosts: tuple[str, ...],
    max_body_bytes: int,
    rate_limit_requests: int,
    rate_limit_window_seconds: float,
    concurrency_limit: int,
    timeout_seconds: float,
) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Correlation-ID"],
        max_age=600,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(trusted_hosts))
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body_bytes)
    app.add_middleware(
        RateLimitMiddleware,
        requests=rate_limit_requests,
        window_seconds=rate_limit_window_seconds,
    )
    app.add_middleware(ConcurrencyLimitMiddleware, limit=concurrency_limit)
    app.add_middleware(TimeoutMiddleware, timeout_seconds=timeout_seconds)
