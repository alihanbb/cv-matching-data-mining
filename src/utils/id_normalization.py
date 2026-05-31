from __future__ import annotations

import re


_FILE_EXT_RE = re.compile(r"\.(docx|doc|pdf|txt|md)$", flags=re.IGNORECASE)


def _norm_token(value: object) -> str:
    return "" if value is None else str(value).strip()


def _strip_corpus_prefix(token: str) -> str:
    out = token.strip()
    while out.lower().startswith("corpus_"):
        out = out[len("corpus_") :].strip()
    return out


def _slugify_for_match(token: str) -> str:
    out = _FILE_EXT_RE.sub("", token.strip())
    out = out.replace("-", "_").replace(" ", "_").lower()
    out = re.sub(r"_+", "_", out).strip("_")
    return out


def normalize_cv_id(cv_id: object) -> str:
    """Canonicalize CV identifiers across Bronze/Silver/Gold/evaluation."""
    out = _strip_corpus_prefix(_norm_token(cv_id))
    if not out:
        return out

    slug = _slugify_for_match(out)
    if not slug:
        return out

    m_vanetik = re.fullmatch(r"vanetik_cv_(\d+)", slug)
    if m_vanetik:
        return f"vanetik_cv_{int(m_vanetik.group(1)):03d}"

    # Keep this intentionally narrow to avoid rewriting generic ids like "cv1".
    m_cv = re.fullmatch(r"cv_?(\d{3,})", slug)
    if m_cv:
        return f"vanetik_cv_{int(m_cv.group(1)):03d}"

    m_digits = re.fullmatch(r"(\d+)", slug)
    if m_digits:
        return f"vanetik_cv_{int(m_digits.group(1)):03d}"

    return out


def normalize_job_id(job_id: object) -> str:
    """Canonicalize job identifiers used in ranking/evaluation joins."""
    out = _strip_corpus_prefix(_norm_token(job_id))
    if not out:
        return out

    slug = _slugify_for_match(out)
    if not slug:
        return out

    m_vanetik = re.fullmatch(r"vanetik_vacancy_(\d+)", slug)
    if m_vanetik:
        return f"vanetik_vacancy_{int(m_vanetik.group(1)):03d}"

    m_vacancy = re.fullmatch(r"vacancy_?(\d+)", slug)
    if m_vacancy:
        return f"vanetik_vacancy_{int(m_vacancy.group(1)):03d}"

    return out
