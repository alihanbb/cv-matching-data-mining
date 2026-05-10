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
SHORT_TEXT_CHARS = 40


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


def _passes_ranking_filter(source: str, ranking_sources: list[str]) -> bool:
    if not ranking_sources:
        return True
    src = source.strip()
    return bool(src and src in ranking_sources)


def _bronze_resume_row(
    obj: dict[str, Any], cleaner, src_default: str
) -> dict[str, Any] | None:
    rid = str(obj.get("resume_id", "") or obj.get("cv_id", "") or "").strip()
    raw_text = str(obj.get("raw_text", "") or obj.get("text", "") or "").strip()
    if not rid or not raw_text:
        return None
    src = str(obj.get("source", "")).strip() or src_default
    cleaned_text = cleaner.clean(raw_text)
    return {
        "cv_id": rid,
        "source": src,
        "source_file": str(obj.get("source_file", "")).strip(),
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "text": cleaned_text,
    }


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
    pipeline_cfg: dict[str, Any] | None = None,
    write_silver: bool | None = None,
) -> tuple[int, int, int]:
    """Silver ingest: prefers Bronze JSONL, then folders. Writes enriched Silver files + profiles."""
    from src.preprocessing.cleaner import TextCleaner
    from src.silver.build import write_silver_artifacts

    pre = preprocessor_cfg or {}
    cleaner = TextCleaner(
        remove_stopwords=pre.get("remove_stopwords", True),
        lemmatize=pre.get("lemmatize", True),
        language=pre.get("language", "en"),
    )

    ing = ingest_cfg or {}
    ranking_sources = [
        str(x).strip() for x in (ing.get("ranking_sources") or []) if str(x).strip()
    ]

    bronze_profile_rows: list[dict[str, Any]] = []
    ranking_cv_rows: list[dict[str, Any]] = []
    use_bronze_cvs = False
    bronze_resume_path: Path | None = None

    if root is not None:
        bronze_resume_path = bronze_resumes_path(root, ing)
        if bronze_resume_path.is_file():
            use_bronze_cvs = True
            skipped_ranking = 0
            for obj in iter_jsonl_objects(bronze_resume_path):
                if not isinstance(obj, dict):
                    continue
                row = _bronze_resume_row(obj, cleaner, src_default="unknown")
                if not row:
                    continue
                bronze_profile_rows.append(dict(row))
                if _passes_ranking_filter(row["source"], ranking_sources):
                    if len(row["raw_text"].strip()) >= SHORT_TEXT_CHARS:
                        ranking_cv_rows.append(dict(row))
                else:
                    skipped_ranking += 1
            if skipped_ranking and ranking_sources:
                logger.info(
                    "Bronze: %d resume rows omitted from ranking (not in ranking_sources)",
                    skipped_ranking,
                )
            logger.info(
                "Bronze ingest: %d profile rows from %s, %d ranking-eligible resumes",
                len(bronze_profile_rows),
                bronze_resume_path.name,
                len(ranking_cv_rows),
            )

    if not use_bronze_cvs:
        if ranking_sources:
            logger.warning(
                "ranking_sources is set but Bronze resume JSONL missing at %s — "
                "folder ingest uses source bronze_folder.",
                bronze_resume_path if root else "?",
            )
        for fp in _iter_raw_files(raw_cvs_dir):
            try:
                text = extract_text_from_path(fp)
            except Exception as exc:
                logger.warning(
                    "Skipping CV file (extract failed) %s: %s: %s",
                    fp.name,
                    type(exc).__name__,
                    exc,
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
            row = {
                "cv_id": did,
                "source": "bronze_folder",
                "source_file": fp.name,
                "raw_text": text,
                "cleaned_text": cleaned_text,
                "text": cleaned_text,
            }
            bronze_profile_rows.append(row)
            if _passes_ranking_filter(row["source"], ranking_sources):
                if len(text.strip()) >= SHORT_TEXT_CHARS:
                    ranking_cv_rows.append(dict(row))

    ranking_ordered: list[dict[str, Any]] = []
    seen_rid: set[str] = set()
    for row in ranking_cv_rows:
        if len(row["raw_text"].strip()) < SHORT_TEXT_CHARS:
            continue
        cid = row["cv_id"]
        if cid in seen_rid:
            continue
        seen_rid.add(cid)
        ranking_ordered.append(dict(row))

    n_primary_ranking = len(ranking_ordered)

    ranking_final: dict[str, dict[str, Any]] = {r["cv_id"]: r for r in ranking_ordered}

    short_extra_rank = 0
    dup_extra_rank = 0
    if extra_cv_rows:
        for r in extra_cv_rows:
            cid = str(r.get("cv_id", "")).strip()
            txt = str(r.get("text", "")).strip()
            if not cid or not txt:
                continue
            if len(txt) < SHORT_TEXT_CHARS:
                short_extra_rank += 1
                continue
            cleaned_txt = cleaner.clean(txt)
            try:
                CleanDocument(doc_id=cid, text=cleaned_txt)
            except Exception as exc:
                logger.warning(
                    "Skipping extra CV row (validation failed) id=%s: %s", cid, exc
                )
                continue
            nrow = {
                "cv_id": cid,
                "source": str(r.get("source", "") or "jsonl_corpus"),
                "source_file": str(r.get("source_file", "") or ""),
                "raw_text": txt,
                "cleaned_text": cleaned_txt,
                "text": cleaned_txt,
            }
            bronze_profile_rows.append(nrow)
            if cid not in ranking_final:
                ranking_final[cid] = nrow
            else:
                dup_extra_rank += 1

    merged_ranking = list(ranking_final.values())
    n_extra_ranking = max(0, len(merged_ranking) - n_primary_ranking)

    if root is not None:
        from src.ingest.cv_corpus import ner_profile_extensions_from_jsonl

        for r in ner_profile_extensions_from_jsonl(root, ing):
            bronze_profile_rows.append(r)

    if short_extra_rank:
        logger.info(
            "Skipped %d corpus rows shorter than %d chars (ranking corpus)",
            short_extra_rank,
            SHORT_TEXT_CHARS,
        )

    job_rows: list[dict[str, Any]] = []
    use_bronze_jobs = False
    bronze_job_path: Path | None = None
    if root is not None:
        bronze_job_path = bronze_jobs_path(root, ing)
        if bronze_job_path.is_file():
            use_bronze_jobs = True
            for obj in iter_jsonl_objects(bronze_job_path):
                if not isinstance(obj, dict):
                    continue
                jid = str(obj.get("job_id", "") or "").strip()
                raw_text = str(
                    obj.get("raw_text", "") or obj.get("text", "") or ""
                ).strip()
                title = str(obj.get("title", "") or jid or "").strip()
                if not jid or not raw_text:
                    continue
                src = str(obj.get("source", "")).strip() or "unknown"
                cleaned_job = cleaner.clean(raw_text)
                row = {
                    "job_id": jid,
                    "source": src,
                    "source_file": str(obj.get("source_file", "")).strip(),
                    "title": title,
                    "raw_text": raw_text,
                    "cleaned_text": cleaned_job,
                    "text": cleaned_job,
                }
                job_rows.append(row)

    if not use_bronze_jobs:
        for fp in _iter_raw_files(raw_jobs_dir):
            try:
                text = extract_text_from_path(fp)
            except Exception as exc:
                logger.warning(
                    "Skipping job file (extract failed) %s: %s: %s",
                    fp.name,
                    type(exc).__name__,
                    exc,
                )
                continue
            if not text.strip():
                logger.debug(
                    "Skipping job file (empty text after extract): %s", fp.name
                )
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
    pd.DataFrame(merged_ranking, columns=ccols).to_csv(out_cvs_csv, index=False)
    jcols = [
        "job_id",
        "source",
        "source_file",
        "title",
        "raw_text",
        "cleaned_text",
        "text",
    ]
    pd.DataFrame(job_rows, columns=jcols).to_csv(out_jobs_csv, index=False)

    if write_silver is None:
        write_silver = bool(
            (pipeline_cfg or {}).get("silver", {}).get("write_silver_on_ingest", True)
        )
    if root and pipeline_cfg and write_silver:
        bronze_profile_unique: dict[str, dict[str, Any]] = {}
        for r in bronze_profile_rows:
            bronze_profile_unique.setdefault(r["cv_id"], r)
        write_silver_artifacts(
            root,
            pipeline_cfg,
            profile_cv_rows=list(bronze_profile_unique.values()),
            job_rows_in=job_rows,
        )

    if dup_extra_rank:
        logger.info(
            "Ranking corpus: %d corpus rows skipped (cv_id already in ranking set)",
            dup_extra_rank,
        )

    return n_primary_ranking, n_extra_ranking, len(job_rows)
