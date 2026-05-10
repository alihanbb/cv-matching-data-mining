from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd


def _relevance_by_job(ground_truth: pd.DataFrame) -> dict[str, set[str]]:
    """job_id -> {cv_id} for binary metrics; any grade >= 1 is relevant."""
    sub = ground_truth[ground_truth["relevant"].astype(int) >= 1]
    m: dict[str, set[str]] = defaultdict(set)
    for _, r in sub.iterrows():
        m[str(r["job_id"])].add(str(r["cv_id"]))
    return dict(m)


def _graded_by_job(ground_truth: pd.DataFrame) -> dict[str, dict[str, int]]:
    """job_id -> {cv_id: grade}; includes all graded relevance rows (0..3)."""
    m: dict[str, dict[str, int]] = defaultdict(dict)
    for _, r in ground_truth.iterrows():
        m[str(r["job_id"])][str(r["cv_id"])] = int(r["relevant"])
    return dict(m)


def mean_reciprocal_rank(ranked: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    rel = _relevance_by_job(ground_truth)
    if not rel:
        return 0.0
    rrs: list[float] = []
    for job, relevant in rel.items():
        sub = ranked[ranked["job_id"].astype(str) == job].sort_values("rank_for_job")
        for _, row in sub.iterrows():
            if str(row["cv_id"]) in relevant:
                rrs.append(1.0 / int(row["rank_for_job"]))
                break
        else:
            rrs.append(0.0)
    return sum(rrs) / max(len(rrs), 1)


def _dcg(rels: list[int]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked: pd.DataFrame, ground_truth: pd.DataFrame, k: int) -> float:
    """Graded NDCG@K. Supports binary (0/1) and graded (0..3) labels."""
    grades = _graded_by_job(ground_truth)
    if not grades:
        return 0.0
    scores: list[float] = []
    for job, cv_grade in grades.items():
        sub = (
            ranked[ranked["job_id"].astype(str) == job]
            .sort_values("rank_for_job")
            .head(k)
        )
        gained = [int(cv_grade.get(str(row["cv_id"]), 0)) for _, row in sub.iterrows()]
        dcg = _dcg(gained)
        ideal_grades = sorted(cv_grade.values(), reverse=True)[:k]
        ideal = ideal_grades + [0] * max(0, k - len(ideal_grades))
        idcg = _dcg(ideal)
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / max(len(scores), 1)


def mean_average_precision(ranked: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    rel = _relevance_by_job(ground_truth)
    if not rel:
        return 0.0
    aps: list[float] = []
    for job, relevant in rel.items():
        sub = ranked[ranked["job_id"].astype(str) == job].sort_values("rank_for_job")
        hits = 0
        precisions: list[float] = []
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            if str(row["cv_id"]) in relevant:
                hits += 1
                precisions.append(hits / rank)
        if not relevant:
            continue
        aps.append(sum(precisions) / len(relevant))
    return sum(aps) / max(len(aps), 1)
