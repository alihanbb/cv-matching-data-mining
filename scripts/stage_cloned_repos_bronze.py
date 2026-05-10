#!/usr/bin/env python3
"""Stage NataliaVanetik/vacancy-resume-matching-dataset into bronze (DOCX CVs + vacancy texts)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Copy Vanetik CV DOCX and vacancy rows into data/bronze (sibling clone under workspace)."
    )
    ap.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Folder containing vacancy-resume-matching-dataset (default: parent of project root).",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (cv-matching-data-mining).",
    )
    args = ap.parse_args()
    project: Path = args.root
    workspace: Path = args.workspace or project.parent
    vanetik = workspace / "vacancy-resume-matching-dataset"
    cv_src = vanetik / "CV"
    vac_csv = vanetik / "5_vacancies.csv"
    if not cv_src.is_dir():
        raise SystemExit(f"Missing CV directory: {cv_src}")
    if not vac_csv.is_file():
        raise SystemExit(f"Missing vacancies CSV: {vac_csv}")

    bronze_cv = project / "data" / "bronze" / "cvs"
    bronze_jobs = project / "data" / "bronze" / "job_descriptions"
    eval_dir = project / "data" / "evaluation"
    for p in (bronze_cv, bronze_jobs, eval_dir):
        p.mkdir(parents=True, exist_ok=True)

    n_cv = 0
    for docx in sorted(cv_src.glob("*.docx")):
        dest = bronze_cv / f"vanetik_{docx.name}"
        shutil.copy2(docx, dest)
        n_cv += 1

    n_jobs = 0
    with vac_csv.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = str(row.get("id", "")).strip() or str(row.get("uid", "")).strip()
            if not jid:
                continue
            title = (row.get("job_title") or "").strip()
            body = (row.get("job_description") or "").strip()
            text = f"{title}\n\n{body}".strip() if title else body
            if not text:
                continue
            (bronze_jobs / f"vanetik_vacancy_{jid}.txt").write_text(text, encoding="utf-8")
            n_jobs += 1

    note = {
        "source": str(vanetik),
        "cv_docx_copied": n_cv,
        "vacancy_txt_written": n_jobs,
        "annotations_file": str(vanetik / "annotations-for-the-first-30-vacancies.txt"),
    }
    (eval_dir / "stage_vanetik_bronze.json").write_text(
        json.dumps(note, indent=2), encoding="utf-8"
    )
    print(f"Bronze: {n_cv} Vanetik DOCX -> {bronze_cv}")
    print(f"Bronze: {n_jobs} vacancy TXT -> {bronze_jobs}")


if __name__ == "__main__":
    main()
