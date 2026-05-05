from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schemas.documents import CleanDocument

from .text_extract import extract_text_from_path

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
) -> tuple[int, int]:
    """
    Reads supported files under raw dirs and writes cleaned_*.csv (id, text).
    Returns (n_cvs_written, n_jobs_written).
    """
    cv_rows: list[dict] = []
    job_rows: list[dict] = []

    for fp in _iter_raw_files(raw_cvs_dir):
        try:
            text = extract_text_from_path(fp)
        except Exception:
            continue
        if not text.strip():
            continue
        did = _doc_id_from_path(raw_cvs_dir, fp)
        CleanDocument(doc_id=did, text=text)
        cv_rows.append({"cv_id": did, "text": text})

    for fp in _iter_raw_files(raw_jobs_dir):
        try:
            text = extract_text_from_path(fp)
        except Exception:
            continue
        if not text.strip():
            continue
        did = _doc_id_from_path(raw_jobs_dir, fp)
        CleanDocument(doc_id=did, text=text)
        job_rows.append({"job_id": did, "text": text})

    out_cvs_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jobs_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(out_cvs_csv, index=False)
    pd.DataFrame(job_rows).to_csv(out_jobs_csv, index=False)
    return len(cv_rows), len(job_rows)
