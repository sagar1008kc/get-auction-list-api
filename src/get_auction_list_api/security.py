"""Security utilities shared by retrieval and future read-only tools."""

import asyncio
import ipaddress
import random
import re
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from get_auction_list_api.observability.logging import REDACTED, redact

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
_URL = re.compile(r"https://[^\s<>()\]\"']+")
UNTRUSTED_EVIDENCE_START = "<UNTRUSTED_EVIDENCE>"
UNTRUSTED_EVIDENCE_END = "</UNTRUSTED_EVIDENCE>"

Resolver = Callable[[str, int], Awaitable[Sequence[str]]]


async def _resolve(host: str, port: int) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(sorted({cast(str, record[4][0]) for record in records}))


@dataclass(frozen=True, slots=True)
class AllowedSource:
    host: str
    path_prefixes: tuple[str, ...]


class URLPolicy:
    """Exact HTTPS host/path allowlist with DNS-based private-network rejection."""

    def __init__(
        self,
        sources: Sequence[AllowedSource],
        *,
        resolver: Resolver = _resolve,
    ) -> None:
        self._sources = {source.host.casefold(): source for source in sources}
        self._resolver = resolver

    async def validate(self, url: str) -> str:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.port not in (None, 443)
        ):
            raise ValueError("URL does not satisfy outbound source policy.")
        host = parsed.hostname.casefold().rstrip(".")
        source = self._sources.get(host)
        path = parsed.path or "/"
        if source is None or not any(
            path == prefix or path.startswith(prefix.rstrip("/") + "/")
            for prefix in source.path_prefixes
        ):
            raise ValueError("URL is not an approved source.")
        addresses = await self._resolver(host, 443)
        if not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        ):
            raise ValueError("Source DNS resolved outside the public network.")
        return urlunsplit(("https", host, path, parsed.query, ""))

    async def validate_redirect(self, previous_url: str, location: str) -> str:
        if not urlsplit(location).scheme:
            previous = urlsplit(previous_url)
            if not location.startswith("/"):
                base = previous.path.rsplit("/", 1)[0]
                location = f"{base}/{location}"
            location = urlunsplit(("https", previous.netloc, location, "", ""))
        return await self.validate(location)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0
    jitter_ratio: float = 0.2

    def delay(self, attempt: int, *, random_value: float | None = None) -> float:
        if attempt < 1:
            raise ValueError("Attempt is one-based.")
        value = min(self.max_delay_seconds, self.base_delay_seconds * (2 ** (attempt - 1)))
        sample = random.random() if random_value is None else random_value  # noqa: S311
        return float(max(0.0, value * (1 + self.jitter_ratio * (2 * sample - 1))))

    def should_retry(self, *, method: str, attempt: int, status_code: int | None) -> bool:
        if method.upper() not in {"GET", "HEAD"} or attempt >= self.max_attempts:
            return False
        return status_code is None or status_code in {408, 425, 429, 500, 502, 503, 504}


def mark_untrusted_evidence(content: str) -> str:
    """Fence evidence so prompts cannot confuse source text with instructions."""

    escaped = content.replace(UNTRUSTED_EVIDENCE_END, "&lt;/UNTRUSTED_EVIDENCE&gt;")
    return f"{UNTRUSTED_EVIDENCE_START}\n{escaped}\n{UNTRUSTED_EVIDENCE_END}"


def mask_telemetry(value: Any) -> Any:
    """Redact secrets first, then common direct-contact PII."""

    sanitized = redact(value)
    if isinstance(sanitized, Mapping):
        return {str(key): mask_telemetry(item) for key, item in sanitized.items()}
    if isinstance(sanitized, Sequence) and not isinstance(sanitized, (str, bytes, bytearray)):
        return [mask_telemetry(item) for item in sanitized]
    if isinstance(sanitized, str):
        return _PHONE.sub(REDACTED, _EMAIL.sub(REDACTED, sanitized))
    return sanitized


def validate_response_links(text: str, allowed_urls: Sequence[str]) -> tuple[str, ...]:
    """Ensure generated links exactly match verified source links."""

    links = tuple(match.rstrip(".,;:") for match in _URL.findall(text))
    allowed = frozenset(allowed_urls)
    if any(link not in allowed for link in links):
        raise ValueError("Response contains an unverified link.")
    return links


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"


class SideEffectApproval(Protocol):
    """Future write-capable tools must obtain an explicit, scoped decision."""

    async def request(
        self,
        *,
        user_id: str,
        tool_name: str,
        action_summary: str,
        idempotency_key: str,
    ) -> ApprovalDecision: ...
