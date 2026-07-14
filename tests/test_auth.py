import json
import time
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from get_auction_list_api.auth import (
    Permission,
    SupabaseJWTValidator,
    authenticate_authorization,
)
from get_auction_list_api.errors import AppError

ISSUER = "https://project.supabase.co/auth/v1"
AUDIENCE = "authenticated"


def public_jwk(key: rsa.RSAPrivateKey, kid: str) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    value.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return value


def token(
    key: rsa.RSAPrivateKey,
    kid: str,
    *,
    audience: str = AUDIENCE,
    expires_at: int | None = None,
    app_metadata: Any = ...,
) -> str:
    claims: dict[str, object] = {
        "sub": str(uuid4()),
        "iss": ISSUER,
        "aud": audience,
        "exp": expires_at if expires_at is not None else int(time.time()) + 60,
    }
    if app_metadata is ...:
        claims["app_metadata"] = {"roles": ["ops"]}
    elif app_metadata is not None:
        claims["app_metadata"] = app_metadata
    return jwt.encode(
        claims,
        key,
        algorithm="RS256",
        headers={"kid": kid},
    )


@pytest.mark.asyncio
async def test_validates_signature_claims_and_permissions() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    async def fetcher(_url: str) -> Mapping[str, Any]:
        return {"keys": [public_jwk(key, "one")]}

    validator = SupabaseJWTValidator(
        jwks_url=f"{ISSUER}/.well-known/jwks.json",
        issuer=ISSUER,
        audience=AUDIENCE,
        fetcher=fetcher,
    )

    principal = await validator.validate(token(key, "one"))

    assert "ops" in principal.roles
    assert Permission.DOCUMENT_READ in principal.permissions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "app_metadata",
    [None, {}, {"roles": []}, {"role": "user"}, {"roles": ["user"]}],
)
async def test_default_and_explicit_user_role_includes_document_read(
    app_metadata: dict[str, object] | None,
) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    async def fetcher(_url: str) -> Mapping[str, Any]:
        return {"keys": [public_jwk(key, "one")]}

    validator = SupabaseJWTValidator(
        jwks_url=f"{ISSUER}/.well-known/jwks.json",
        issuer=ISSUER,
        audience=AUDIENCE,
        fetcher=fetcher,
    )

    principal = await validator.validate(token(key, "one", app_metadata=app_metadata))

    assert "user" in principal.roles
    assert principal.permissions == frozenset(
        {
            Permission.AUCTION_READ,
            Permission.DOCUMENT_READ,
            Permission.TOOL_EXECUTE,
        }
    )
    assert Permission.INGESTION_WRITE not in principal.permissions
    assert Permission.AUDIT_READ not in principal.permissions


@pytest.mark.asyncio
async def test_unknown_kid_forces_one_rotation_refresh() -> None:
    old_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    new_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    calls = 0

    async def fetcher(_url: str) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        key, kid = (old_key, "old") if calls == 1 else (new_key, "new")
        return {"keys": [public_jwk(key, kid)]}

    validator = SupabaseJWTValidator(
        jwks_url=f"{ISSUER}/.well-known/jwks.json",
        issuer=ISSUER,
        audience=AUDIENCE,
        fetcher=fetcher,
    )

    await validator.validate(token(new_key, "new"))

    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("audience", "expires_at"),
    [("wrong", None), (AUDIENCE, 1)],
)
async def test_rejects_wrong_audience_and_expiry(audience: str, expires_at: int | None) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    async def fetcher(_url: str) -> Mapping[str, Any]:
        return {"keys": [public_jwk(key, "one")]}

    validator = SupabaseJWTValidator(
        jwks_url=f"{ISSUER}/.well-known/jwks.json",
        issuer=ISSUER,
        audience=AUDIENCE,
        fetcher=fetcher,
    )

    with pytest.raises(AppError) as error:
        await validator.validate(token(key, "one", audience=audience, expires_at=expires_at))

    assert error.value.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [None, "", "Basic abc", "Bearer", "Bearer one two"])
async def test_rejects_missing_or_malformed_bearer_header(header: str | None) -> None:
    async def fetcher(_url: str) -> Mapping[str, Any]:
        raise AssertionError("Malformed authorization must not reach JWKS")

    validator = SupabaseJWTValidator(
        jwks_url=f"{ISSUER}/.well-known/jwks.json",
        issuer=ISSUER,
        audience=AUDIENCE,
        fetcher=fetcher,
    )

    with pytest.raises(AppError) as error:
        await authenticate_authorization(header, validator)

    assert error.value.status_code == 401
