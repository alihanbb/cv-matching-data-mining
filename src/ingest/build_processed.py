from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas.documents import CleanDocument

from .bronze_jsonl_io import bronze_jobs_path, bronze_resumes_path, iter_jsonl_objects
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


def _sources_pass_filter(source: str, ranking_sources: list[str]) -> bool:
    if not ranking_sources:
        return True
    return bool(source.strip() and source.strip() in ranking_sources)


def build_processed_from_raw(
    raw_cvs_dir: Path,
    raw_jobs_dir: Path,
    out_cvs_csv: Path,
    out_jobs_csv: Path,
    *,
    root: Path | None = None,
    ingest_cfg: dict[str, Any] | None = None,
    preprocessor_cfg: dict[str, Any] | None = None,
    extra_cv_rows: list[dict[str, str]] | None = None,
) -> tuple[int, int, int]:
    """Silver CSV ingest: prefers ``resumes_bronze.jsonl`` / ``jobs_bronze.jsonl`` when present.

    Returns (n_cv_from_primary, n_cv_extra_jsonl, n_jobs).
    """
    from src.preprocessing.cleaner import TextCleaner

    pre = preprocessor_cfg or {}
    cleaner = TextCleaner(
        remove_stopwords=pre.get("remove_stopwords", True),
        lemmatize=pre.get("lemmatize", True),
        language=pre.get("language", "en"),
    )

    ing = ingest_cfg or {}
    ranking_sources = [str(x).strip() for x in (ing.get("ranking_sources") or []) if str(x).strip()]

    cv_rows: list[dict[str, Any]] = []
    use_bronze_cvs = False
    bronze_resume_path: Path | None = None

    if root is not None:
        bronze_resume_path = bronze_resumes_path(root, ing)
        if bronze_resume_path.is_file():
            use_bronze_cvs = True
            skipped_filter = 0
            for obj in iter_jsonl_objects(bronze_resume_path):
                src = str(obj.get("source", "")).strip() or "unknown"
                if not _sources_pass_filter(src, ranking_sources):
                    skipped_filter += 1
                    continue
                rid = str(obj.get("resume_id", "") or obj.get("cv_id", "") or "").strip()
                raw_text = str(obj.get("raw_text", "") or obj.get("text", "") or "").strip()
                if not rid or not raw_text:
                    continue
                meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
                cleaned_text = cleaner.clean(raw_text)
                cv_rows.append(
                    {
                        "cv_id": rid,
                        "source": src,
                        "source_file": str(obj.get("source_file", "")).strip(),
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text,
                        "text": cleaned_text,
                    }
                )
            if skipped_filter and ranking_sources:
                logger.info(
                    "Bronze ingest: skipped %d resume rows outside ranking_sources=%s",
                    skipped_filter,
                    ranking_sources,
                )
            logger.info("Loaded %d CV rows from %s", len(cv_rows), bronze_resume_path)

    if not use_bronze_cvs:
        if ranking_sources:
            logger.warning(
                "ranking_sources is set but no Bronze resume JSONL at %s — folder ingest ignores source filter.",
                bronze_resume_path if root else "?",
            )
        for fp in _iter_raw_files(raw_cvs_dir):
            try:
                text = extract_text_from_path(fp)
            except Exception as exc:
                logger.warning(
                    "Skipping CV file (extract failed) %s: %s: %s", fp.name, type(exc).__name__, exc,
                )
                continue
            if not text.strip():
                logger.debug("Skipping CV file (empty text after extract): %s", fp.name)
                continue
            did = _doc_id_from_path(raw_cvs_dir, fp)
            cleaned_text = cleaner.clean(text)
            try:
                CleanDocument(doc_id=did, text=cleaned_text)
            except Exception:
                cleaned_text = text
            cv_rows.append(
                {
                    "cv_id": did,
                    "source": "bronze_folder",
                    "source_file": fp.name,
                    "raw_text": text,
                    "cleaned_text": cleaned_text,
                    "text": cleaned_text,
                }
            )

    n_primary = len(cv_rows)
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
            cleaned_txt = cleaner.clean(txt)
            try:
                CleanDocument(doc_id=cid, text=cleaned_txt)
            except Exception as exc:
                logger.warning("Skipping extra CV row (validation failed) id=%s: %s", cid, exc)
                continue
            cv_rows.append(
                {
                    "cv_id": cid,
                    "source": str(r.get("source", "") or "jsonl_corpus"),
                    "source_file": str(r.get("source_file", "") or ""),
                    "raw_text": txt,
                    "cleaned_text": cleaned_txt,
                    "text": cleaned_txt,
                }
            )
            seen_ids.add(cid)
            n_extra += 1

    job_rows: list[dict[str, Any]] = []
    use_bronze_jobs = False
    bronze_job_path: Path | None = None
    if root is not None:
        bronze_job_path = bronze_jobs_path(root, ing)
        if bronze_job_path.is_file():
            use_bronze_jobs = True
            for obj in iter_jsonl_objects(bronze_job_path):
                jid = str(obj.get("job_id", "") or "").strip()
                raw_text = str(obj.get("raw_text", "") or obj.get("text", "") or "").strip()
                title = str(obj.get("title", "") or jid or "").strip()
                if not jid or not raw_text:
                    continue
                src = str(obj.get("source", "")).strip() or "unknown"
                cleaned_job = cleaner.clean(raw_text)
                job_rows.append(
                    {
                        "job_id": jid,
                        "source": src,
                        "source_file": str(obj.get("source_file", "")).strip(),
                        "title": title,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_job,
                        "text": cleaned_job,
                    }
                )

    if not use_bronze_jobs:
        for fp in _iter_raw_files(raw_jobs_dir):
            try:
                text = extract_text_from_path(fp)
            except Exception as exc:
                logger.warning(
                    "Skipping job file (extract failed) %s: %s: %s", fp.name, type(exc).__name__, exc,
                )
                continue
            if not text.strip():
                logger.debug("Skipping job file (empty text after extract): %s", fp.name)
                continue
            did = _doc_id_from_path(raw_jobs_dir, fp)
            cleaned_job = cleaner.clean(text)
            try:
                CleanDocument(doc_id=did, text=cleaned_job)
            except Exception:
                cleaned_job = text
            job_rows.append(
                {
                    "job_id": did,
                    "source": "bronze_folder",
                    "source_file": fp.name,
                    "title": "",
                    "raw_text": text,
                    "cleaned_text": cleaned_job,
                    "text": cleaned_job,
                }
            )

    out_cvs_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jobs_csv.parent.mkdir(parents=True, exist_ok=True)

    ccols = ["cv_id", "source", "source_file", "raw_text", "cleaned_text", "text"]
    pd.DataFrame(cv_rows, columns=ccols).to_csv(out_cvs_csv, index=False)
    jcols = ["job_id", "source", "source_file", "title", "raw_text", "cleaned_text", "text"]
    pd.DataFrame(job_rows, columns=jcols).to_csv(out_jobs_csv, index=False)
    return n_primary, n_extra, len(job_rows)
