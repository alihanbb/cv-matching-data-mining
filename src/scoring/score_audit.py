from __future__ import annotations

import pandas as pd

from src.config.defaults import SCORE_AUDIT_WARN_THRESHOLD


def weighted_fusion_v1_row(
    row: pd.Series, weights_v1: dict[str, float], *, has_semantic: bool
) -> float:
    """Raw weighted V1 sum for a single (cv, job) pair from component columns."""
    w_v1 = {k: float(v) for k, v in weights_v1.items()}
    if not has_semantic:
        w_v1["dense"] = 0.0
    w_v1.pop("bm25", None)
    total = sum(max(0.0, v) for v in w_v1.values())
    w_v1 = {k: max(0.0, v) / total for k, v in w_v1.items()}
    tf = float(row.get("tfidf_score", 0.0))
    sem = float(row.get("semantic_score", 0.0)) if has_semantic else 0.0
    sk = float(row.get("skill_score", 0.0))
    ex = float(row.get("experience_score", 0.0))
    return (
        w_v1["tfidf"] * tf
        + w_v1.get("dense", 0.0) * sem
        + w_v1["skills"] * sk
        + w_v1["experience"] * ex
    )


def add_score_audit_columns(
    df: pd.DataFrame,
    weights_v1: dict[str, float],
    *,
    has_semantic: bool,
    has_bm25: bool,
) -> pd.DataFrame:
    """Append audit columns; expects raw component columns present."""
    out = df.copy()
    out["final_score_raw"] = out.apply(
        lambda row: weighted_fusion_v1_row(row, weights_v1, has_semantic=has_semantic),
        axis=1,
    )
    out["final_score_v1"] = out["final_score_raw"]
    out["score_check"] = out["final_score_raw"]
    if "final_score" in out.columns:
        out["score_diff"] = (
            out["final_score"].astype(float) - out["score_check"].astype(float)
        ).abs()
        out["score_warning"] = out["score_diff"].apply(
            lambda x: "CHECK>0.01" if float(x) > SCORE_AUDIT_WARN_THRESHOLD else ""
        )
    else:
        out["score_diff"] = 0.0
        out["score_warning"] = ""
    return out


def semantic_zero_ratio(df: pd.DataFrame) -> float:
    if df.empty or "semantic_score" not in df.columns:
        return 0.0
    zero = (df["semantic_score"].astype(float).abs() < 1e-9).mean()
    return float(zero)
