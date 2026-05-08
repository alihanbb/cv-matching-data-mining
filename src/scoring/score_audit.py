from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.defaults import SCORE_AUDIT_WARN_THRESHOLD


def add_score_audit_columns(
    df: pd.DataFrame,
    weights_v1: dict[str, float],
    *,
    has_semantic: bool,
    has_bm25: bool,
    weights_v2: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Append audit columns; expects raw component columns present."""
    out = df.copy()
    w_v1 = {k: float(v) for k, v in weights_v1.items()}
    if not has_semantic:
        w_v1["dense"] = 0.0
    w_v1.pop("bm25", None)
    total = sum(max(0.0, v) for v in w_v1.values())
    w_v1 = {k: max(0.0, v) / total for k, v in w_v1.items()}

    def row_raw_sum(row: pd.Series) -> float:
        tf = float(row.get("tfidf_score", 0.0))
        sem = float(row.get("semantic_score", 0.0)) if has_semantic else 0.0
        sk = float(row.get("skill_score", 0.0))
        ex = float(row.get("experience_score", 0.0))
        return w_v1["tfidf"] * tf + w_v1.get("dense", 0.0) * sem + w_v1["skills"] * sk + w_v1["experience"] * ex

    out["final_score_raw"] = out.apply(row_raw_sum, axis=1)
    if "final_score_normalized" not in out.columns:
        out["final_score_normalized"] = np.nan
    out["score_check"] = out["final_score_raw"]
    if "final_score" in out.columns:
        out["score_diff"] = (out["final_score"].astype(float) - out["score_check"].astype(float)).abs()
        out["score_warning"] = out["score_diff"].apply(lambda x: "CHECK>0.01" if x > SCORE_AUDIT_WARN_THRESHOLD else "")
    else:
        out["score_diff"] = 0.0
        out["score_warning"] = ""
    if weights_v2 and has_bm25:
        w2 = {k: float(v) for k, v in weights_v2.items()}
        if not has_semantic:
            w2["dense"] = 0.0
        tot2 = sum(max(0.0, v) for v in w2.values())
        w2 = {k: max(0.0, v) / tot2 for k, v in w2.items()}

        def row_v2(row: pd.Series) -> float:
            return (
                w2["tfidf"] * float(row.get("tfidf_score", 0.0))
                + w2.get("dense", 0.0) * float(row.get("semantic_score", 0.0))
                + w2.get("bm25", 0.0) * float(row.get("bm25_score", 0.0))
                + w2["skills"] * float(row.get("skill_score", 0.0))
                + w2["experience"] * float(row.get("experience_score", 0.0))
            )

        out["final_score_v2_bm25"] = out.apply(row_v2, axis=1)
    return out


def semantic_zero_ratio(df: pd.DataFrame) -> float:
    if df.empty or "semantic_score" not in df.columns:
        return 0.0
    zero = (df["semantic_score"].astype(float).abs() < 1e-9).mean()
    return float(zero)
