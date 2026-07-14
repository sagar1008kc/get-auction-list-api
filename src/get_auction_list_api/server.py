"""Production ASGI launcher with bounded graceful shutdown."""

import uvicorn

from get_auction_list_api.config import get_settings


def main() -> None:
    """Run one API process; container platforms should scale process replicas."""

    settings = get_settings()
    uvicorn.run(
        "get_auction_list_api.main:app",
        host=settings.bind_host,
        port=settings.port,
        log_level=settings.log_level.casefold(),
        timeout_graceful_shutdown=settings.graceful_shutdown_seconds,
        timeout_keep_alive=settings.keep_alive_seconds,
        proxy_headers=True,
        server_header=False,
    )


if __name__ == "__main__":
    main()
