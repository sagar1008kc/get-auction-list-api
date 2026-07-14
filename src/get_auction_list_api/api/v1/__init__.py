"""Versioned API routers."""

from get_auction_list_api.api.v1.chat import router as chat_router
from get_auction_list_api.api.v1.operations import router as operations_router

__all__ = ["chat_router", "operations_router"]
