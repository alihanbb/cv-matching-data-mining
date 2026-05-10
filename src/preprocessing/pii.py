from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


_EMAIL = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b",
)
_URL = re.compile(
    r"\bhttps?://[^\s]+|\bwww\.[^\s]+",
    re.IGNORECASE,
)
_TR_ADDR = re.compile(
    r"\b(?:mah\.?|mahalle|cad\.?|cadde|sok\.?|sokak|no:?\s*\d+|daire:?\s*\d+|ilçe|şehir)\b[^\n\r]{0,120}",
    re.IGNORECASE,
)


def anonymize_text(text: str, *, mask: str = "[REDACTED]") -> str:
    """Mask common PII patterns so rankers do not exploit contact leakage."""
    if not text:
        return text
    t = _EMAIL.sub(mask, text)
    t = _URL.sub(mask, t)
    t = _PHONE.sub(mask, t)
    t = _TR_ADDR.sub(mask, t)
    return t


def anonymize_text_audited(text: str, *, mask: str = "[REDACTED]", doc_id: str = "") -> str:
    """Anonymize text and log how many PII patterns were masked.

    Use this variant when an audit trail is required (e.g. pipeline runs
    where KVKK / GDPR compliance must be demonstrated).
    """
    if not text:
        return text
    counts = {
        "email": len(_EMAIL.findall(text)),
        "url": len(_URL.findall(text)),
        "phone": len(_PHONE.findall(text)),
        "address": len(_TR_ADDR.findall(text)),
    }
    result = anonymize_text(text, mask=mask)
    total = sum(counts.values())
    if total > 0:
        logger.info(
            "PII masked doc_id=%r: email=%d url=%d phone=%d address=%d",
            doc_id or "<unknown>",
            counts["email"],
            counts["url"],
            counts["phone"],
            counts["address"],
        )
    return result
