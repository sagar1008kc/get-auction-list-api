from pathlib import Path

import pytest
from pydantic import SecretStr

from get_auction_list_api.config import Settings


def test_settings_load_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GET_AUCTION_LIST_ENVIRONMENT", "test")
    monkeypatch.setenv("GET_AUCTION_LIST_DATABASE_URL", "postgresql://private")

    settings = Settings()

    assert settings.environment == "test"
    assert isinstance(settings.database_url, SecretStr)
    assert "private" not in repr(settings)


def test_settings_do_not_load_dotenv_implicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GET_AUCTION_LIST_ENVIRONMENT=production\n",
        encoding="utf-8",
    )

    assert Settings().environment == "local"
