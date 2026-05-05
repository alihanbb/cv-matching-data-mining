from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd


def _relevance_by_job(ground_truth: pd.DataFrame) -> dict[str, set[str]]:
    sub = ground_truth[ground_truth["relevant"].astype(int) == 1]
    m: dict[str, set[str]] = defaultdict(set)
    for _, r in sub.iterrows():
        m[str(r["job_id"])].add(str(r["cv_id"]))
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
    rel = _relevance_by_job(ground_truth)
    if not rel:
        return 0.0
    scores: list[float] = []
    for job, relevant in rel.items():
        sub = ranked[ranked["job_id"].astype(str) == job].sort_values("rank_for_job").head(k)
        gained = [1 if str(row["cv_id"]) in relevant else 0 for _, row in sub.iterrows()]
        dcg = _dcg(gained)
        ideal_len = min(len(relevant), k)
        ideal = [1] * ideal_len + [0] * (k - ideal_len)
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
