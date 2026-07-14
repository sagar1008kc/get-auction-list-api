"""Application configuration loaded exclusively from process environment."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated service settings.

    Secret values use ``SecretStr`` and are never included in health responses or logs.
    No dotenv file is loaded implicitly.
    """

    model_config = SettingsConfigDict(
        env_prefix="GET_AUCTION_LIST_",
        env_file=None,
        extra="ignore",
        frozen=True,
    )

    service_name: str = "get-auction-list-api"
    service_version: str = "development"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    bind_host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    graceful_shutdown_seconds: int = Field(default=30, ge=1, le=300)
    keep_alive_seconds: int = Field(default=5, ge=1, le=120)

    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = Field(default=None, repr=False)
    database_url: SecretStr | None = Field(default=None, repr=False)
    jwt_issuer: str | None = None
    jwt_audience: str = "authenticated"
    jwks_cache_ttl_seconds: float = Field(default=300, ge=30, le=3600)

    database_pool_min_size: int = Field(default=1, ge=1, le=20)
    database_pool_max_size: int = Field(default=10, ge=1, le=100)
    database_command_timeout_seconds: float = Field(default=10, gt=0, le=120)
    checkpoint_enabled: bool = False
    checkpoint_setup_on_start: bool = True
    checkpoint_pool_min_size: int = Field(default=1, ge=1, le=10)
    checkpoint_pool_max_size: int = Field(default=4, ge=1, le=20)

    cors_origins: tuple[str, ...] = ()
    trusted_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "testserver")
    max_request_body_bytes: int = Field(default=1_048_576, ge=1024, le=25_000_000)
    rate_limit_requests: int = Field(default=120, ge=1, le=10_000)
    rate_limit_window_seconds: float = Field(default=60, ge=1, le=3600)
    concurrency_limit: int = Field(default=100, ge=1, le=1000)
    request_timeout_seconds: float = Field(default=30, gt=0, le=120)
    public_http_timeout_seconds: float = Field(default=5, gt=0, le=30)
    public_http_max_attempts: int = Field(default=2, ge=1, le=4)
    public_http_max_response_bytes: int = Field(default=2_000_000, ge=1024, le=10_000_000)
    public_http_cache_ttl_seconds: float = Field(default=30, ge=1, le=300)
    internal_mcp_token: SecretStr | None = Field(default=None, repr=False)
    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = Field(default=1536, ge=1, le=4096)
    openai_timeout_seconds: float = Field(default=10, gt=0, le=60)
    openai_max_retries: int = Field(default=1, ge=0, le=3)
    ingestion_handler: str | None = None
    ingestion_poll_seconds: float = Field(default=2, ge=0.1, le=60)
    ingestion_heartbeat_seconds: float = Field(default=15, ge=1, le=300)
    ingestion_stale_seconds: float = Field(default=120, ge=10, le=3600)
    ingestion_max_attempts: int = Field(default=5, ge=1, le=20)
    ingestion_retry_seconds: float = Field(default=30, ge=1, le=3600)
    otel_enabled: bool = True
    otel_exporter_otlp_endpoint: str | None = None
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = Field(default=None, repr=False)
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_sample_rate: float = Field(default=1.0, ge=0, le=1)

    approved_source_hosts: tuple[str, ...] = (
        "getauctionlist.com",
        "apps.wilco.org",
        "www.wilcotx.gov",
        "search.wcad.org",
    )

    @model_validator(mode="after")
    def validate_security_settings(self) -> "Settings":
        if self.database_pool_min_size > self.database_pool_max_size:
            raise ValueError("database_pool_min_size cannot exceed database_pool_max_size")
        if self.checkpoint_pool_min_size > self.checkpoint_pool_max_size:
            raise ValueError("checkpoint_pool_min_size cannot exceed checkpoint_pool_max_size")
        if self.checkpoint_enabled and self.database_url is None:
            raise ValueError("Checkpointing requires database_url.")
        if "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origins are not permitted.")
        if "*" in self.trusted_hosts:
            raise ValueError("Wildcard trusted hosts are not permitted.")
        if self.supabase_url is not None and not self.supabase_url.startswith("https://"):
            raise ValueError("supabase_url must use HTTPS.")
        for name, value in (
            ("otel_exporter_otlp_endpoint", self.otel_exporter_otlp_endpoint),
            ("langfuse_host", self.langfuse_host),
            ("openai_base_url", self.openai_base_url),
        ):
            if value is not None and not value.startswith(("https://", "http://localhost")):
                raise ValueError(f"{name} must use HTTPS (or localhost for development).")
        if self.langfuse_enabled and (
            self.langfuse_public_key is None or self.langfuse_secret_key is None
        ):
            raise ValueError("Enabled Langfuse requires public and secret keys.")
        if self.environment == "production" and not self.cors_origins:
            raise ValueError("Production requires at least one explicit CORS origin.")
        return self

    @property
    def resolved_jwt_issuer(self) -> str | None:
        return self.jwt_issuer or (
            self.supabase_url.rstrip("/") + "/auth/v1" if self.supabase_url else None
        )

    @property
    def jwks_url(self) -> str | None:
        return (
            self.supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
            if self.supabase_url
            else None
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
