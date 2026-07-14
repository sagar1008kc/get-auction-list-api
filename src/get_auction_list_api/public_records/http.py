"""Bounded direct HTTP transport for explicitly approved public sources."""

import asyncio
import ipaddress
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from get_auction_list_api.api.metrics import SOURCE_HEALTH, source_label
from get_auction_list_api.public_records.models import PublicRecordToolError, ToolErrorCategory


@dataclass(frozen=True, slots=True)
class CachedResponse:
    body: bytes
    content_type: str
    final_url: str
    expires_at: float


class ApprovedHttpClient:
    """HTTP GET only, with destination validation before every network hop."""

    def __init__(
        self,
        approved_hosts: tuple[str, ...],
        *,
        timeout_seconds: float = 5,
        max_attempts: int = 2,
        max_response_bytes: int = 2_000_000,
        cache_ttl_seconds: float = 30,
        max_redirects: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._hosts = frozenset(host.casefold().rstrip(".") for host in approved_hosts)
        self._timeout = timeout_seconds
        self._attempts = max_attempts
        self._limit = max_response_bytes
        self._ttl = cache_ttl_seconds
        self._redirects = max_redirects
        self._transport = transport
        self._resolver = resolver or self._resolve
        self._cache: dict[str, CachedResponse] = {}

    @staticmethod
    def _resolve(host: str) -> list[str]:
        return list(
            {str(item[4][0]) for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
        )

    async def _validate(self, url: str) -> None:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold().rstrip(".")
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or host not in self._hosts
        ):
            raise PublicRecordToolError(
                ToolErrorCategory.FORBIDDEN_DESTINATION,
                "The requested destination is not approved.",
            )
        try:
            addresses = await asyncio.to_thread(self._resolver, host)
        except OSError as error:
            raise PublicRecordToolError(
                ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                "The approved source could not be resolved.",
                retryable=True,
            ) from error
        if not addresses:
            raise PublicRecordToolError(
                ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                "The approved source could not be resolved.",
                retryable=True,
            )
        for value in addresses:
            address = ipaddress.ip_address(value)
            if not address.is_global:
                raise PublicRecordToolError(
                    ToolErrorCategory.FORBIDDEN_DESTINATION,
                    "The destination resolved to a non-public address.",
                )

    async def get(self, url: str) -> tuple[CachedResponse, bool]:
        cached = self._cache.get(url)
        if cached is not None and cached.expires_at > time.monotonic():
            return cached, True
        current = url
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=httpx.Timeout(self._timeout),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            for redirect_count in range(self._redirects + 1):
                await self._validate(current)
                response: httpx.Response | None = None
                for attempt in range(self._attempts):
                    try:
                        request = client.build_request(
                            "GET",
                            current,
                            headers={"Accept": "application/json,text/html,application/pdf"},
                        )
                        response = await client.send(request, stream=True)
                    except httpx.TimeoutException as error:
                        if attempt + 1 == self._attempts:
                            raise PublicRecordToolError(
                                ToolErrorCategory.TIMEOUT,
                                "The approved source timed out.",
                                retryable=True,
                            ) from error
                        await asyncio.sleep(0.05 * (2**attempt))
                        continue
                    if response.status_code >= 500 and attempt + 1 < self._attempts:
                        await response.aclose()
                        await asyncio.sleep(0.05 * (2**attempt))
                        continue
                    break
                if response is None:
                    raise PublicRecordToolError(
                        ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                        "The approved source did not return a response.",
                        retryable=True,
                    )
                if response.is_redirect:
                    location = response.headers.get("location")
                    await response.aclose()
                    if location is None or redirect_count == self._redirects:
                        raise PublicRecordToolError(
                            ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                            "The approved source returned an invalid redirect.",
                            retryable=True,
                        )
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    status_code = response.status_code
                    await response.aclose()
                    raise PublicRecordToolError(
                        ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                        "The approved source request failed.",
                        retryable=status_code >= 500,
                    )
                declared_length = response.headers.get("content-length")
                if declared_length is not None and int(declared_length) > self._limit:
                    await response.aclose()
                    raise PublicRecordToolError(
                        ToolErrorCategory.RESPONSE_TOO_LARGE,
                        "The approved source response exceeded the size limit.",
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._limit:
                        await response.aclose()
                        raise PublicRecordToolError(
                            ToolErrorCategory.RESPONSE_TOO_LARGE,
                            "The approved source response exceeded the size limit.",
                        )
                await response.aclose()
                value = CachedResponse(
                    body=bytes(body),
                    content_type=response.headers.get("content-type", ""),
                    final_url=current,
                    expires_at=time.monotonic() + self._ttl,
                )
                self._cache[url] = value
                host = (urlsplit(current).hostname or "").casefold()
                source = "wcad" if "wcad" in host else "county"
                SOURCE_HEALTH.labels(source_label(source)).set(1)
                return value, False
        raise PublicRecordToolError(
            ToolErrorCategory.UPSTREAM_UNAVAILABLE,
            "The approved source redirect limit was exceeded.",
        )
