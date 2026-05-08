from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.schemas.documents import CleanDocument

from .text_extract import extract_text_from_path

logger = logging.getLogger(__name__)

_RAW_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def _iter_raw_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in _RAW_EXTENSIONS:
            out.append(p)
    return out


def _doc_id_from_path(base: Path, file_path: Path) -> str:
    rel = file_path.relative_to(base)
    stem = rel.as_posix().replace("/", "__")
    return stem.rsplit(".", 1)[0] if "." in stem else stem


def build_processed_from_raw(
    raw_cvs_dir: Path,
    raw_jobs_dir: Path,
    out_cvs_csv: Path,
    out_jobs_csv: Path,
    *,
    extra_cv_rows: list[dict[str, str]] | None = None,
) -> tuple[int, int, int]:
    """
    Reads supported files under raw dirs and writes cleaned_*.csv (id, text).
    Optional ``extra_cv_rows`` (e.g. from JSONL corpus) appended after bronze CV files.
    Returns (n_cv_bronze, n_cv_extra, n_jobs).
    """
    cv_rows: list[dict] = []

    for fp in _iter_raw_files(raw_cvs_dir):
        try:
            text = extract_text_from_path(fp)
        except Exception as exc:
            logger.warning("Skipping CV file (extract failed) %s: %s: %s", fp.name, type(exc).__name__, exc)
            continue
        if not text.strip():
            logger.debug("Skipping CV file (empty text after extract): %s", fp.name)
            continue
        did = _doc_id_from_path(raw_cvs_dir, fp)
        CleanDocument(doc_id=did, text=text)
        cv_rows.append({"cv_id": did, "text": text})

    n_bronze = len(cv_rows)
    seen_ids = {r["cv_id"] for r in cv_rows}
    n_extra = 0
    if extra_cv_rows:
        for r in extra_cv_rows:
            cid = str(r.get("cv_id", "")).strip()
            txt = str(r.get("text", "")).strip()
            if not cid or not txt:
                continue
            if cid in seen_ids:
                continue
            try:
                CleanDocument(doc_id=cid, text=txt)
            except Exception as exc:
                logger.warning("Skipping extra CV row (validation failed) id=%s: %s", cid, exc)
                continue
            cv_rows.append({"cv_id": cid, "text": txt})
            seen_ids.add(cid)
            n_extra += 1

    job_rows: list[dict] = []
    for fp in _iter_raw_files(raw_jobs_dir):
        try:
            text = extract_text_from_path(fp)
        except Exception as exc:
            logger.warning("Skipping job file (extract failed) %s: %s: %s", fp.name, type(exc).__name__, exc)
            continue
        if not text.strip():
            logger.debug("Skipping job file (empty text after extract): %s", fp.name)
            continue
        did = _doc_id_from_path(raw_jobs_dir, fp)
        CleanDocument(doc_id=did, text=text)
        job_rows.append({"job_id": did, "text": text})

    out_cvs_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jobs_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows, columns=["cv_id", "text"]).to_csv(out_cvs_csv, index=False)
    pd.DataFrame(job_rows, columns=["job_id", "text"]).to_csv(out_jobs_csv, index=False)
    return n_bronze, n_extra, len(job_rows)
