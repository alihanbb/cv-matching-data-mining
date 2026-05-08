from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """ASCII-aware tokenizer used by TextCleaner (TF-IDF pipeline)."""
    if not text:
        return []
    return re.findall(r"[a-z0-9+#.]+", text.lower())


def tokenize_unicode(text: str, min_len: int = 2) -> list[str]:
    """Unicode-aware tokenizer for BM25 scoring.

    Unlike ``tokenize``, this preserves non-Latin characters so that
    multilingual CVs are not silently emptied.
    """
    if not text:
        return []
    tokens = re.findall(r"[\w#+.]+", text, flags=re.UNICODE)
    out: list[str] = []
    for token in tokens:
        cleaned = token.lower().strip(".")
        if len(cleaned) >= min_len:
            out.append(cleaned)
    return out
