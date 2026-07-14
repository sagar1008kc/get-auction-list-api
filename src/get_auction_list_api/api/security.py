"""Authentication dependencies shared by versioned routes."""

from typing import Annotated

from fastapi import Depends, Header

from get_auction_list_api.auth import Principal, authenticate_authorization, unauthorized
from get_auction_list_api.dependencies import AppDependencies, get_dependencies


async def current_principal(
    dependencies: Annotated[AppDependencies, Depends(get_dependencies)],
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    if dependencies.authenticator is None:
        raise unauthorized("Authentication is not configured.")
    return await authenticate_authorization(authorization, dependencies.authenticator)
