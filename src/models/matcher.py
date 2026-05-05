from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring.explain import explain_pair


def rank_candidates_for_jobs(
    sim_matrix: np.ndarray,
    cv_ids: list,
    job_ids: list,
    top_k: int,
    component_matrices: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    sim_matrix shape: (n_cvs, n_jobs) — similarity[cv_idx, job_idx].
    Returns long table: job_id, cv_id, score, rank_for_job, optional channel scores.
    """
    rows: list[dict] = []
    n_cvs, n_jobs = sim_matrix.shape
    comps = component_matrices or {}
    for j in range(n_jobs):
        col = sim_matrix[:, j]
        order = np.argsort(-col)[:top_k]
        for rank, i in enumerate(order, start=1):
            row: dict = {
                "job_id": job_ids[j],
                "cv_id": cv_ids[i],
                "score": float(col[i]),
                "rank_for_job": rank,
            }
            for name, mat in comps.items():
                row[f"score_{name}"] = float(mat[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def enrich_with_explanations(
    ranked: pd.DataFrame,
    cv_ids: list,
    job_ids: list,
    cv_skill_sets: list[set[str]],
    job_skill_sets: list[set[str]],
    cv_years: list[float],
    job_required: list[float | None],
) -> pd.DataFrame:
    cv_pos = {str(cid): idx for idx, cid in enumerate(cv_ids)}
    job_pos = {str(jid): idx for idx, jid in enumerate(job_ids)}
    matched: list[str] = []
    missing: list[str] = []
    notes: list[str] = []
    for _, r in ranked.iterrows():
        i = cv_pos[str(r["cv_id"])]
        j = job_pos[str(r["job_id"])]
        ex = explain_pair(cv_skill_sets[i], job_skill_sets[j], cv_years[i], job_required[j])
        matched.append(ex["matched_skills"])
        missing.append(ex["missing_skills"])
        notes.append(ex["experience_note"])
    out = ranked.copy()
    out["matched_skills"] = matched
    out["missing_skills"] = missing
    out["experience_note"] = notes
    return out
