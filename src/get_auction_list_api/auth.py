"""Supabase JWT validation and application authorization contracts."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
from urllib.request import Request, urlopen
from uuid import UUID

import jwt
from jwt import InvalidTokenError, PyJWK
from pydantic import BaseModel, ConfigDict, Field

from get_auction_list_api.errors import AppError, ErrorCode


class Permission(StrEnum):
    AUCTION_READ = "auction:read"
    DOCUMENT_READ = "document:read"
    INGESTION_WRITE = "ingestion:write"
    AUDIT_READ = "audit:read"
    TOOL_EXECUTE = "tool:execute"


_ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    # Authenticated chat users may search auctions and approved knowledge docs.
    # Ops/admin keep ingestion/audit boundaries via the full permission set.
    "user": frozenset({Permission.AUCTION_READ, Permission.DOCUMENT_READ, Permission.TOOL_EXECUTE}),
    "ops": frozenset(Permission),
    "admin": frozenset(Permission),
}


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: UUID
    roles: frozenset[str] = Field(default_factory=frozenset)
    permissions: frozenset[Permission] = Field(default_factory=frozenset)
    token_id: str | None = None

    def require(self, permission: Permission) -> None:
        if permission not in self.permissions:
            raise forbidden()


def unauthorized(message: str = "Authentication credentials are invalid.") -> AppError:
    return AppError(code=ErrorCode.UNAUTHORIZED, message=message, status_code=401)


def forbidden(message: str = "You do not have permission to perform this action.") -> AppError:
    return AppError(code=ErrorCode.FORBIDDEN, message=message, status_code=403)


JwksFetcher = Callable[[str], Awaitable[Mapping[str, Any]]]


async def _default_fetcher(url: str) -> Mapping[str, Any]:
    def fetch() -> Mapping[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urlopen(request, timeout=5) as response:  # noqa: S310
            if response.status != 200:
                raise OSError("JWKS request failed")
            value = json.loads(response.read(1_000_001))
        if not isinstance(value, dict):
            raise ValueError("JWKS response must be an object")
        return value

    return await asyncio.to_thread(fetch)


@dataclass(slots=True)
class _JwksCache:
    keys: dict[str, PyJWK]
    expires_at: float


class SupabaseJWTValidator:
    """Validate signed tokens, refreshing cached JWKS once for an unknown key id."""

    def __init__(
        self,
        *,
        jwks_url: str,
        issuer: str,
        audience: str,
        cache_ttl_seconds: float = 300,
        fetcher: JwksFetcher = _default_fetcher,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._jwks_url = jwks_url
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._cache_ttl_seconds = cache_ttl_seconds
        self._fetcher = fetcher
        self._clock = clock
        self._cache = _JwksCache(keys={}, expires_at=0)
        self._lock = asyncio.Lock()

    async def _keys(self, *, force: bool = False) -> dict[str, PyJWK]:
        if not force and self._cache.expires_at > self._clock():
            return self._cache.keys
        async with self._lock:
            if not force and self._cache.expires_at > self._clock():
                return self._cache.keys
            payload = await self._fetcher(self._jwks_url)
            raw_keys = payload.get("keys")
            if not isinstance(raw_keys, list):
                raise unauthorized()
            try:
                keys = {
                    str(raw["kid"]): PyJWK.from_dict(raw)
                    for raw in raw_keys
                    if isinstance(raw, dict) and raw.get("kid")
                }
            except (KeyError, InvalidTokenError, ValueError) as error:
                raise unauthorized() from error
            if not keys:
                raise unauthorized()
            self._cache = _JwksCache(
                keys=keys,
                expires_at=self._clock() + self._cache_ttl_seconds,
            )
            return keys

    async def validate(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
        except InvalidTokenError as error:
            raise unauthorized() from error
        kid = header.get("kid")
        algorithm = header.get("alg")
        if not isinstance(kid, str) or algorithm not in {"RS256", "ES256"}:
            raise unauthorized()

        keys = await self._keys()
        key = keys.get(kid)
        if key is None:
            key = (await self._keys(force=True)).get(kid)
        if key is None or key.algorithm_name != algorithm:
            raise unauthorized()

        try:
            claims = jwt.decode(
                token,
                key=key.key,
                algorithms=[algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["sub", "iss", "aud", "exp"]},
            )
            user_id = UUID(claims["sub"])
        except (InvalidTokenError, KeyError, TypeError, ValueError) as error:
            raise unauthorized() from error

        metadata = claims.get("app_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        raw_roles = metadata.get("roles", [])
        roles = (
            {str(role).casefold() for role in raw_roles} if isinstance(raw_roles, list) else set()
        )
        role = metadata.get("role")
        if isinstance(role, str):
            roles.add(role.casefold())
        if not roles:
            roles.add("user")
        permissions = frozenset(
            permission for role_name in roles for permission in _ROLE_PERMISSIONS.get(role_name, ())
        )
        return Principal(
            user_id=user_id,
            roles=frozenset(roles),
            permissions=permissions,
            token_id=claims.get("jti") if isinstance(claims.get("jti"), str) else None,
        )


class TokenValidator(Protocol):
    async def validate(self, token: str) -> Principal: ...


async def authenticate_authorization(
    authorization: str | None,
    validator: TokenValidator,
) -> Principal:
    """Parse the HTTP Bearer scheme without accepting ambiguous credentials."""

    if authorization is None:
        raise unauthorized("Authentication credentials are required.")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].casefold() != "bearer" or not parts[1]:
        raise unauthorized()
    return await validator.validate(parts[1])
