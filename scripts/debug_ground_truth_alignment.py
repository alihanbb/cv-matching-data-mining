#!/usr/bin/env python3
"""Debug alignment between evaluation ground truth and explained candidate scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.id_normalization import normalize_cv_id, normalize_job_id


def _resolve(path_like: str, root: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (root / p)


def _sample(values: pd.Series, n: int = 10) -> list[str]:
    uniq = sorted({str(v) for v in values.dropna().astype(str).tolist() if str(v).strip()})
    return uniq[:n]


def build_alignment_report(
    ground_truth_path: Path,
    scores_path: Path,
    unmatched_out: Path,
) -> dict[str, Any]:
    gt = pd.read_csv(ground_truth_path)
    scores = pd.read_csv(scores_path)

    if "cv_id" not in gt.columns and "resume_id" in gt.columns:
        gt = gt.copy()
        gt["cv_id"] = gt["resume_id"]
    required_gt = {"job_id", "cv_id"}
    missing_gt = required_gt - set(gt.columns)
    if missing_gt:
        raise ValueError(
            f"ground_truth missing required columns: {sorted(missing_gt)} "
            "(cv_id can be provided via resume_id alias)."
        )

    required_scores = {"job_id", "cv_id"}
    missing_scores = required_scores - set(scores.columns)
    if missing_scores:
        raise ValueError(f"scores missing required columns: {sorted(missing_scores)}")

    gt = gt.copy()
    scores = scores.copy()
    gt["job_id_norm"] = gt["job_id"].map(normalize_job_id)
    gt["cv_id_norm"] = gt["cv_id"].map(normalize_cv_id)
    scores["job_id_norm"] = scores["job_id"].map(normalize_job_id)
    scores["cv_id_norm"] = scores["cv_id"].map(normalize_cv_id)

    gt = gt[(gt["job_id_norm"] != "") & (gt["cv_id_norm"] != "")]
    scores = scores[(scores["job_id_norm"] != "") & (scores["cv_id_norm"] != "")]

    score_pairs = scores[["job_id_norm", "cv_id_norm"]].drop_duplicates()
    merged = gt.merge(
        score_pairs,
        on=["job_id_norm", "cv_id_norm"],
        how="left",
        indicator=True,
    )

    total_gt = int(len(merged))
    matched_gt = int((merged["_merge"] == "both").sum())
    unmatched_gt = total_gt - matched_gt
    match_ratio = (matched_gt / total_gt) if total_gt else 0.0

    unmatched = merged[merged["_merge"] != "both"].copy()
    unmatched_cols = [
        c
        for c in [
            "job_id",
            "cv_id",
            "resume_id",
            "relevance",
            "relevant",
            "source",
            "job_id_norm",
            "cv_id_norm",
        ]
        if c in unmatched.columns
    ]
    unmatched_out.parent.mkdir(parents=True, exist_ok=True)
    unmatched[unmatched_cols].to_csv(unmatched_out, index=False)

    return {
        "total_ground_truth_rows": total_gt,
        "matched_ground_truth_rows": matched_gt,
        "unmatched_ground_truth_rows": unmatched_gt,
        "match_ratio": match_ratio,
        "unique_ground_truth_job_ids": int(gt["job_id_norm"].nunique()),
        "unique_score_job_ids": int(scores["job_id_norm"].nunique()),
        "sample_ground_truth_cv_ids": _sample(gt["cv_id_norm"]),
        "sample_score_cv_ids": _sample(scores["cv_id_norm"]),
        "unmatched_ground_truth_examples_csv": str(unmatched_out),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ground-truth",
        default="data/evaluation/ground_truth.csv",
        help="Ground-truth CSV path.",
    )
    ap.add_argument(
        "--scores",
        default="data/gold/rankings/candidate_scores_explained.csv",
        help="Candidate scores explained CSV path.",
    )
    ap.add_argument(
        "--unmatched-out",
        default="data/evaluation/unmatched_ground_truth_examples.csv",
        help="Output CSV path for unmatched ground-truth rows.",
    )
    args = ap.parse_args()

    report = build_alignment_report(
        _resolve(args.ground_truth, root),
        _resolve(args.scores, root),
        _resolve(args.unmatched_out, root),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
