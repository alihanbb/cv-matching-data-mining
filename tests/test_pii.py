from src.preprocessing.pii import anonymize_text


def test_anonymize_masks_email_and_url() -> None:
    t = "Contact me at test@example.com or visit https://example.com/path"
    out = anonymize_text(t)
    assert "test@example.com" not in out
    assert "https://example.com" not in out
    assert "[REDACTED]" in out
