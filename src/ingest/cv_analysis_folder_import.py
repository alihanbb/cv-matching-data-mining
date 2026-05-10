"""Import ``cv_analysis/data/data/<CATEGORY>/*`` PDF/DOCX/TXT into Bronze ``resumes_bronze.jsonl``.

Layout (sibling ``cv_analysis`` workspace)::

    ../data/data/ACCOUNTANT/foo.pdf
    ../data/data/SALES/bar.pdf

Runtime pipeline still reads only ``data/bronze/`` JSONL; this is an import-time helper.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

from src.ingest.external_bronze_import import (
    default_outputs,
    load_jsonl_by_id,
    stats_for_records,
    write_jsonl,
    write_stats,
)
from src.ingest.text_extract import extract_text_from_path
from src.utils.id_normalization import normalize_cv_id

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE_TAG = "cv_analysis_pdf_corpus"
_RAW_EXT = {".pdf", ".docx", ".txt", ".md"}


def normalize_category_token(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def iter_corpus_files(corpus_root: Path) -> list[tuple[Path, Path]]:
    """Return ``(file_path, category_dir)`` sorted for stable IDs."""
    out: list[tuple[Path, Path]] = []
    if not corpus_root.is_dir():
        return out
    for cat_dir in sorted([p for p in corpus_root.iterdir() if p.is_dir()]):
        for fp in sorted(cat_dir.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in _RAW_EXT:
                out.append((fp, cat_dir))
    return out


def build_resume_row(
    fp: Path,
    cat_dir: Path,
    *,
    source_tag: str,
) -> dict[str, Any] | None:
    rel = fp.relative_to(cat_dir)
    try:
        text = extract_text_from_path(fp)
    except Exception as exc:
        logger.warning("Extract failed %s: %s", fp, exc)
        return None
    text = text.strip()
    if not text:
        logger.debug("Skipping empty text: %s", fp)
        return None
    cat_slug = normalize_category_token(cat_dir.name)
    rel_stem = normalize_category_token(rel.with_suffix("").as_posix().replace("/", "_")) or "doc"
    rid = normalize_cv_id(f"cv_pdf_{cat_slug}_{rel_stem}")
    source_rel = f"{cat_dir.name}/{rel.as_posix()}"
    return {
        "resume_id": rid,
        "source": source_tag,
        "source_file": source_rel,
        "raw_text": text,
        "language": "en",
        "metadata": {
            "category": cat_dir.name,
            "format": fp.suffix.lower().lstrip("."),
        },
    }


def merge_cv_analysis_pdf_corpus(
    *,
    project_root: Path,
    corpus_root: Path,
    overwrite: bool,
    source_tag: str = _DEFAULT_SOURCE_TAG,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Merge extracted PDF/DOCX corpus into ``resumes_bronze.jsonl`` (by ``resume_id``)."""
    outs = default_outputs(project_root)
    resumes = load_jsonl_by_id(outs.resumes_path, "resume_id")

    if overwrite:
        strip = {source_tag.strip()}
        resumes = {
            k: v
            for k, v in resumes.items()
            if str(v.get("source", "")).strip() not in strip
        }

    pairs = iter_corpus_files(corpus_root)
    if max_files is not None:
        pairs = pairs[: max(0, max_files)]

    added = 0
    skipped_short = 0
    failed = 0
    for fp, cat_dir in pairs:
        row = build_resume_row(fp, cat_dir, source_tag=source_tag)
        if row is None:
            failed += 1
            continue
        # align with build_processed minimum signal
        if len(row["raw_text"].strip()) < 40:
            skipped_short += 1
            continue
        resumes[row["resume_id"]] = row
        added += 1
        if added % 500 == 0:
            logger.info("Imported %d resumes from corpus...", added)

    write_jsonl(outs.resumes_path, sorted(resumes.values(), key=lambda x: x["resume_id"]))
    write_stats(outs.resumes_stats, stats_for_records(resumes.values()))

    return {
        "corpus_root": str(corpus_root),
        "source_tag": source_tag,
        "files_scanned": len(pairs),
        "resume_rows_written_delta": added,
        "skipped_short_text": skipped_short,
        "extract_failures": failed,
        "total_resumes_bronze": len(resumes),
    }


def run_cli(argv: list[str] | None = None, *, default_project_root: Path | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Merge cv_analysis sibling folder corpus (PDF by category) into Bronze resumes JSONL.",
    )
    ap.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Folder whose subfolders are categories (default: ../data/data next to cv-matching-data-mining).",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing data/bronze (default: auto).",
    )
    ap.add_argument(
        "--source-tag",
        default=_DEFAULT_SOURCE_TAG,
        help=f"Bronze ``source`` field (default: {_DEFAULT_SOURCE_TAG}).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Remove existing Bronze rows whose source matches --source-tag, then re-import.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap files processed (debug / sampling).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count corpus files matching extensions; no writes.",
    )
    ap.add_argument("--verbose", action="store_true", help="DEBUG logging.")
    args = ap.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    project_root = (args.project_root or default_project_root or Path.cwd()).resolve()
    corpus_default = project_root.parent / "data" / "data"
    corpus_root = (args.corpus_root or corpus_default).expanduser().resolve()

    if not corpus_root.is_dir():
        logger.error("Corpus root is not a directory: %s", corpus_root)
        sys.exit(1)

    pairs = iter_corpus_files(corpus_root)
    if args.max_files:
        pairs = pairs[: max(0, args.max_files)]

    if args.dry_run:
        by_cat: dict[str, int] = {}
        for fp, cat_dir in pairs:
            by_cat[cat_dir.name] = by_cat.get(cat_dir.name, 0) + 1
        print(f"Corpus root: {corpus_root}")
        print(f"Files (--extensions {sorted(_RAW_EXT)}): {len(pairs)}")
        print(f"Categories: {len(by_cat)}")
        if args.verbose:
            for k in sorted(by_cat.keys()):
                print(f"  {k}: {by_cat[k]}")
        return

    try:
        summary = merge_cv_analysis_pdf_corpus(
            project_root=project_root,
            corpus_root=corpus_root,
            overwrite=bool(args.overwrite),
            source_tag=str(args.source_tag).strip(),
            max_files=args.max_files,
        )
    except Exception as exc:
        logger.exception("Import failed: %s", exc)
        sys.exit(1)

    print(
        f"Bronze resumes merge done: +{summary['resume_rows_written_delta']} rows from corpus "
        f"({summary['files_scanned']} files scanned, {summary['skipped_short_text']} too short text, "
        f"{summary['extract_failures']} extract failures); total resumes in Bronze: "
        f"{summary['total_resumes_bronze']}"
    )
