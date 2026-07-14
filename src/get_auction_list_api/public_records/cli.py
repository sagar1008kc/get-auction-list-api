"""Standalone, local-stdio MCP launcher."""

from get_auction_list_api.config import get_settings
from get_auction_list_api.observability.logging import configure_logging
from get_auction_list_api.public_records.http import ApprovedHttpClient
from get_auction_list_api.public_records.mcp import create_mcp_server
from get_auction_list_api.public_records.service import PublicRecordsService


def main() -> None:
    """Serve the read-only tools over stdio without opening a network listener."""

    settings = get_settings()
    configure_logging(settings)
    client = ApprovedHttpClient(
        settings.approved_source_hosts,
        timeout_seconds=settings.public_http_timeout_seconds,
        max_attempts=settings.public_http_max_attempts,
        max_response_bytes=settings.public_http_max_response_bytes,
        cache_ttl_seconds=settings.public_http_cache_ttl_seconds,
    )
    create_mcp_server(PublicRecordsService(client)).run(transport="stdio")


if __name__ == "__main__":
    main()
