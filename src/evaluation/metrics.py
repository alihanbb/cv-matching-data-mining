from __future__ import annotations

from collections import defaultdict

import pandas as pd


def _relevant_pairs(df: pd.DataFrame) -> set[tuple]:
    """Return relevant (cv_id, job_id) pairs.

    Treats any positive grade (>=1) as relevant; this works for both binary
    (0/1) and graded (0..3) ground truth.
    """
    sub = df[df["relevant"].astype(int) >= 1]
    return set(zip(sub["cv_id"].tolist(), sub["job_id"].tolist()))


def top_k_accuracy(ranked: pd.DataFrame, ground_truth: pd.DataFrame, k: int) -> float:
    """
    Fraction of jobs for which at least one relevant (cv, job) appears in top-k.
    ground_truth columns: cv_id, job_id, relevant (0/1).
    ranked columns: job_id, cv_id, rank_for_job, score.
    """
    rel = _relevant_pairs(ground_truth)
    if not rel:
        return 0.0
    jobs = ranked["job_id"].unique()
    hits = 0
    for job in jobs:
        top = ranked[(ranked["job_id"] == job) & (ranked["rank_for_job"] <= k)]
        cvs = set(top["cv_id"].tolist())
        if any((cv, job) in rel for cv in cvs):
            hits += 1
    return hits / max(len(jobs), 1)


def precision_at_k(ranked: pd.DataFrame, ground_truth: pd.DataFrame, k: int) -> float:
    rel = _relevant_pairs(ground_truth)
    if not rel:
        return 0.0
    by_job: dict = defaultdict(list)
    for _, row in ranked.iterrows():
        if int(row["rank_for_job"]) <= k:
            by_job[row["job_id"]].append(row["cv_id"])
    precisions: list[float] = []
    for job, cvs in by_job.items():
        if not cvs:
            continue
        tp = sum(1 for cv in cvs if (cv, job) in rel)
        precisions.append(tp / len(cvs))
    return sum(precisions) / max(len(precisions), 1)
