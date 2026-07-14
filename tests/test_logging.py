from get_auction_list_api.observability.logging import REDACTED, redact


def test_redact_recursively_removes_secret_fields_and_bearer_values() -> None:
    payload = {
        "authorization": "Bearer private-token",
        "nested": {
            "database_url": "postgresql://private",
            "message": "request used Bearer another-token",
        },
        "items": [{"api_key": "private-key"}],
    }

    redacted = redact(payload)

    assert redacted == {
        "authorization": REDACTED,
        "nested": {
            "database_url": REDACTED,
            "message": f"request used {REDACTED}",
        },
        "items": [{"api_key": REDACTED}],
    }
    assert "private" not in repr(redacted)


def test_redact_masks_query_style_credentials() -> None:
    assert redact("https://example.test/?token=private&safe=yes") == (
        f"https://example.test/?token={REDACTED}&safe=yes"
    )
