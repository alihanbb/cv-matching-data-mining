from __future__ import annotations

import pandas as pd

from src.utils.id_normalization import normalize_cv_id, normalize_job_id


def resolve_score_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for col in (
        "ranking_score",
        "final_score_v2_bm25",
        "final_score_v1",
        "final_score",
        "score",
    ):
        if col in df.columns:
            return col
    return preferred


def dedupe_candidates_by_canonical_cv_id(
    df: pd.DataFrame,
    *,
    score_column: str,
    keep_canonical_column: bool = True,
) -> pd.DataFrame:
    """Normalize IDs and keep best row per (job_id, canonical_cv_id)."""
    if df.empty:
        out = df.copy()
        if keep_canonical_column and "canonical_cv_id" not in out.columns:
            out["canonical_cv_id"] = []
        return out

    out = df.copy()
    out["job_id"] = out["job_id"].map(normalize_job_id)
    out["canonical_cv_id"] = out["cv_id"].map(normalize_cv_id)
    out = out[(out["job_id"] != "") & (out["canonical_cv_id"] != "")]
    if out.empty:
        return out

    score_col = resolve_score_column(out, score_column)
    out[score_col] = pd.to_numeric(out.get(score_col), errors="coerce")
    out = out.dropna(subset=[score_col])
    if out.empty:
        return out

    if "rank_for_job" in out.columns:
        out["_rank_old"] = pd.to_numeric(out["rank_for_job"], errors="coerce").fillna(10**9)
    else:
        out["_rank_old"] = 10**9

    out = out.sort_values(
        by=["job_id", score_col, "_rank_old", "canonical_cv_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=["job_id", "canonical_cv_id"], keep="first")
    out["cv_id"] = out["canonical_cv_id"]

    out = out.sort_values(
        by=["job_id", score_col, "_rank_old", "cv_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    out["rank_for_job"] = out.groupby("job_id").cumcount() + 1
    out = out.drop(columns=["_rank_old"], errors="ignore")
    if not keep_canonical_column:
        out = out.drop(columns=["canonical_cv_id"], errors="ignore")
    return out


def dedupe_candidates_by_job_cv_id(
    df: pd.DataFrame,
    *,
    score_column: str,
) -> pd.DataFrame:
    """Keep best row per exact (job_id, cv_id) pair and recompute rank_for_job."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out["job_id"] = out["job_id"].map(normalize_job_id)
    out["cv_id"] = out["cv_id"].map(normalize_cv_id)
    out = out[(out["job_id"] != "") & (out["cv_id"] != "")]
    if out.empty:
        return out

    score_col = resolve_score_column(out, score_column)
    out[score_col] = pd.to_numeric(out.get(score_col), errors="coerce")
    out = out.dropna(subset=[score_col])
    if out.empty:
        return out

    if "rank_for_job" in out.columns:
        out["_rank_old"] = pd.to_numeric(out["rank_for_job"], errors="coerce").fillna(10**9)
    else:
        out["_rank_old"] = 10**9

    out = out.sort_values(
        by=["job_id", score_col, "_rank_old", "cv_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=["job_id", "cv_id"], keep="first")
    out = out.sort_values(
        by=["job_id", score_col, "_rank_old", "cv_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    out["rank_for_job"] = out.groupby("job_id").cumcount() + 1
    out = out.drop(columns=["_rank_old"], errors="ignore")
    return out
