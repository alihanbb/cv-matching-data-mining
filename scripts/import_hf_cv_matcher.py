#!/usr/bin/env python3
"""Import JeremiahOnu/cv-matcher-data from Hugging Face into data/bronze/ structure.

Requires: pip install -e ".[data_imports]"

Dataset card: https://huggingface.co/datasets/JeremiahOnu/cv-matcher-data
Field names vary by revision — adjust column mapping below if load fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit('Install optional extra: pip install -e ".[data_imports]"') from e

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--split", type=str, default="train")
    args = ap.parse_args()
    root: Path = args.root
    bronze_cv = root / "data" / "bronze" / "cvs"
    bronze_job = root / "data" / "bronze" / "job_descriptions"
    eval_dir = root / "data" / "evaluation"
    for p in (bronze_cv, bronze_job, eval_dir):
        p.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("JeremiahOnu/cv-matcher-data", split=args.split)
    gt_rows: list[dict] = []
    for i, row in enumerate(ds):
        r = dict(row)
        rid = str(r.get("id", i))
        cv_text = str(r.get("resume_text") or r.get("cv") or r.get("text") or "")
        job_text = str(r.get("job_description") or r.get("job") or "")
        if cv_text.strip():
            (bronze_cv / f"cv__{rid}.txt").write_text(cv_text, encoding="utf-8")
        if job_text.strip():
            (bronze_job / f"job__{rid}.txt").write_text(job_text, encoding="utf-8")
        if "label" in r or "match_score" in r or "relevance" in r:
            rel = int(r.get("relevance", r.get("label", r.get("match_score", 0))))
            gt_rows.append(
                {
                    "job_id": f"job__{rid}",
                    "cv_id": f"cv__{rid}",
                    "relevance": max(0, min(3, rel)),
                }
            )

    if gt_rows:
        import pandas as pd

        pd.DataFrame(gt_rows).drop_duplicates().to_csv(eval_dir / "ground_truth.csv", index=False)
    (eval_dir / "import_hf_cv_matcher_note.json").write_text(
        json.dumps(
            {
                "rows_imported": len(ds),
                "hint": "Verify column names match the dataset schema.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Done — CV/job TXT under {bronze_cv} and {bronze_job}. Ground truth rows: {len(gt_rows)}"
    )


if __name__ == "__main__":
    main()
