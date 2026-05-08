#!/usr/bin/env python3
"""Import NataliaVanetik/vacancy-resume-matching-dataset into bronze + optional ground_truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install -e \".[data_imports]\"") from e

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root: Path = args.root
    bronze_cv = root / "data" / "bronze" / "cvs"
    bronze_job = root / "data" / "bronze" / "job_descriptions"
    eval_dir = root / "data" / "evaluation"
    for p in (bronze_cv, bronze_job, eval_dir):
        p.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("NataliaVanetik/vacancy-resume-matching-dataset", split="train")
    gt_rows: list[dict] = []
    for i, row in enumerate(ds):
        r = dict(row)
        cid = str(r.get("resume_id", r.get("cv_id", f"cv_{i}")))
        jid = str(r.get("vacancy_id", r.get("job_id", f"job_{i}")))
        cv_text = str(r.get("resume") or r.get("cv_text") or "")
        job_text = str(r.get("vacancy") or r.get("job_text") or "")
        if cv_text.strip():
            (bronze_cv / f"{cid}.txt").write_text(cv_text, encoding="utf-8")
        if job_text.strip():
            (bronze_job / f"{jid}.txt").write_text(job_text, encoding="utf-8")
        if "label" in r or "match" in r:
            rel = int(float(r.get("label", r.get("match", 0))))
            gt_rows.append({"job_id": jid, "cv_id": cid, "relevance": max(0, min(3, rel))})

    if gt_rows:
        import pandas as pd

        pd.DataFrame(gt_rows).drop_duplicates().to_csv(eval_dir / "ground_truth.csv", index=False)
    (eval_dir / "import_vacancy_note.json").write_text(json.dumps({"samples": len(ds)}, indent=2), encoding="utf-8")
    print(f"Imported {len(ds)} samples. Ground-truth rows: {len(gt_rows)}")


if __name__ == "__main__":
    main()
