from __future__ import annotations


def _norm_token(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalize_cv_id(cv_id: object) -> str:
    """Canonicalize CV identifiers across Bronze/Silver/Gold/evaluation."""
    out = _norm_token(cv_id)
    # Repeated corpus prefixes may appear after multiple ingestion cycles.
    while out.lower().startswith("corpus_"):
        out = out[len("corpus_") :].strip()
    return out


def normalize_job_id(job_id: object) -> str:
    """Canonicalize job identifiers used in ranking/evaluation joins."""
    return _norm_token(job_id)

