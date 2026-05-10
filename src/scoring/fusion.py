from __future__ import annotations

from typing import Any

import numpy as np


def minmax_per_column(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each column into [0, 1] (per job)."""
    out = np.zeros_like(matrix, dtype=np.float64)
    for j in range(matrix.shape[1]):
        col = matrix[:, j].astype(np.float64)
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < eps:
            out[:, j] = 1.0
        else:
            out[:, j] = (col - lo) / (hi - lo)
    return out


def skill_jaccard_matrix(
    cv_sets: list[set[str]], job_sets: list[set[str]]
) -> np.ndarray:
    n_c, n_j = len(cv_sets), len(job_sets)
    s = np.zeros((n_c, n_j), dtype=np.float64)
    for i in range(n_c):
        a = cv_sets[i]
        for j in range(n_j):
            b = job_sets[j]
            if not a and not b:
                s[i, j] = 1.0
            elif not a or not b:
                s[i, j] = 0.0
            else:
                inter = len(a & b)
                union = len(a | b)
                s[i, j] = inter / union if union else 0.0
    return s


def experience_match_matrix(
    cv_years_max: list[float],
    job_required: list[float | None],
) -> np.ndarray:
    n_c, n_j = len(cv_years_max), len(job_required)
    m = np.zeros((n_c, n_j), dtype=np.float64)
    for i in range(n_c):
        cy = float(cv_years_max[i])
        for j in range(n_j):
            req = job_required[j]
            if req is None or req <= 0:
                m[i, j] = 1.0
            elif cy >= req:
                m[i, j] = 1.0
            else:
                m[i, j] = max(0.0, cy / req)
    return m


def fuse_scores(
    tfidf: np.ndarray,
    dense: np.ndarray | None,
    skills: np.ndarray,
    experience: np.ndarray,
    weights: dict[str, float],
    dense_enabled: bool,
    bm25: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Late fusion on min-max normalized channels."""
    w = dict(weights)
    if not dense_enabled or dense is None:
        w["dense"] = 0.0
    w.setdefault("bm25", 0.0)
    if bm25 is None:
        w["bm25"] = 0.0
    tf = minmax_per_column(tfidf)
    sk = minmax_per_column(skills)
    ex = minmax_per_column(experience)
    if dense_enabled and dense is not None:
        de = minmax_per_column(dense)
    else:
        de = None
    bm = minmax_per_column(bm25) if bm25 is not None and w.get("bm25", 0) > 0 else None
    total = sum(max(0.0, float(v)) for v in w.values())
    if total <= 0:
        raise ValueError("Fusion weights sum to zero.")
    w = {k: max(0.0, float(v)) / total for k, v in w.items()}
    fused = w["tfidf"] * tf + w["skills"] * sk + w["experience"] * ex
    if de is not None:
        fused = fused + w["dense"] * de
    if bm is not None:
        fused = fused + w["bm25"] * bm
    return fused, w


def fuse_weighted_raw(
    tfidf: np.ndarray,
    dense: np.ndarray | None,
    skills: np.ndarray,
    experience: np.ndarray,
    weights: dict[str, float],
    dense_enabled: bool,
    bm25: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Weighted sum on **raw** (unnormalized) matrices with normalized weights."""
    w = {k: max(0.0, float(v)) for k, v in weights.items()}
    if not dense_enabled or dense is None:
        w["dense"] = 0.0
    w.setdefault("bm25", 0.0)
    if bm25 is None:
        w["bm25"] = 0.0
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Fusion weights sum to zero.")
    w = {k: v / total for k, v in w.items()}
    fused = w["tfidf"] * tfidf + w["skills"] * skills + w["experience"] * experience
    if dense is not None and w.get("dense", 0) > 0:
        fused = fused + w["dense"] * dense
    if bm25 is not None and w.get("bm25", 0) > 0:
        fused = fused + w["bm25"] * bm25
    return fused, w


def component_dict(
    tfidf: np.ndarray,
    dense: np.ndarray | None,
    skills: np.ndarray,
    experience: np.ndarray,
    bm25: np.ndarray | None,
    *,
    dense_enabled: bool,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        "tfidf": tfidf,
        "skills": skills,
        "experience": experience,
    }
    if dense_enabled and dense is not None:
        out["dense"] = dense
    else:
        out["dense"] = np.zeros_like(tfidf)
    if bm25 is not None:
        out["bm25"] = bm25
    else:
        out["bm25"] = np.zeros_like(tfidf)
    return out
