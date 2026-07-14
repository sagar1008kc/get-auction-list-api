# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.11.28 AS uv

FROM python:3.12-slim-bookworm AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GET_AUCTION_LIST_BIND_HOST=0.0.0.0 \
    GET_AUCTION_LIST_PORT=8000
WORKDIR /app
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --home-dir /nonexistent app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
USER app
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).read()"]
CMD ["auction-api"]
