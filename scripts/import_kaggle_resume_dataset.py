#!/usr/bin/env python3
"""Import a Kaggle Resume Dataset zip into data/bronze/cvs.

Requires Kaggle API credentials (~/.kaggle/kaggle.json) and: pip install -e ".[kaggle_import]"

Usage:
  python scripts/import_kaggle_resume_dataset.py --dataset <owner/name>
Example dataset slugs: datasnaek/resumes or similar — pick an active resume corpus on Kaggle.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise SystemExit(
            'pip install -e ".[kaggle_import]" and configure ~/.kaggle/kaggle.json'
        ) from e

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--dataset", type=str, required=True, help="Kaggle dataset slug owner/name")
    args = ap.parse_args()
    root: Path = args.root
    dl_dir = root / "data" / "bronze" / "_kaggle_download"
    out_cv = root / "data" / "bronze" / "cvs"
    dl_dir.mkdir(parents=True, exist_ok=True)
    out_cv.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(args.dataset, path=str(dl_dir), unzip=True)
    for z in dl_dir.glob("*.zip"):
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dl_dir)
    moved = 0
    for p in dl_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}:
            target = out_cv / p.name
            target.write_bytes(p.read_bytes())
            moved += 1
    print(f"Copied {moved} files into {out_cv}. Review and de-duplicate filenames if needed.")


if __name__ == "__main__":
    main()
