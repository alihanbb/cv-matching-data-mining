from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.helpers import load_config, project_root, resolve_path

from .build_processed import build_processed_from_raw
from .cv_corpus import extra_cvs_from_ingest_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Silver CSVs from data/bronze (Bronze → Silver).")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None, help="Project root (default: auto)")
    args = ap.parse_args()
    root = args.root or project_root()
    cfg = load_config(args.config) if args.config else load_config()
    ing = cfg.get("ingest", {})
    raw_cvs = resolve_path(root, ing.get("raw_cvs_dir", "data/bronze/cvs"))
    raw_jobs = resolve_path(root, ing.get("raw_jobs_dir", "data/bronze/job_descriptions"))
    paths = cfg["paths"]
    out_cvs = resolve_path(root, paths["processed_cvs"])
    out_jobs = resolve_path(root, paths["processed_jobs"])
    extra = extra_cvs_from_ingest_config(root, ing)
    n_bronze, n_corpus, n_job = build_processed_from_raw(
        raw_cvs, raw_jobs, out_cvs, out_jobs, extra_cv_rows=extra or None
    )
    print(
        f"Ingest complete: {n_bronze} bronze + {n_corpus} corpus = {n_bronze + n_corpus} CV rows → {out_cvs}, "
        f"{n_job} job rows → {out_jobs}"
    )


if __name__ == "__main__":
    main()
