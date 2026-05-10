from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.defaults import CROSS_ENCODER_BLEND, CROSS_ENCODER_MODEL_NAME

logger = logging.getLogger(__name__)


def rerank_with_cross_encoder(
    root: Path,
    cfg: dict[str, Any],
    explained_path: Path,
    *,
    top_n: int = 20,
    model_name: str = CROSS_ENCODER_MODEL_NAME,
) -> pd.DataFrame:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers required for cross-encoder reranking."
        ) from e

    df = pd.read_csv(explained_path)
    if df.empty:
        return df

    proc_cvs = root / "data/silver/cleaned_cvs.csv"
    proc_jobs = root / "data/silver/cleaned_jobs.csv"
    if not proc_cvs.is_file() or not proc_jobs.is_file():
        raise FileNotFoundError("Silver processed CV/job CSVs required for reranking.")

    cvs = pd.read_csv(proc_cvs)
    jobs = pd.read_csv(proc_jobs)
    cv_text = {str(r["cv_id"]): str(r["text"]) for _, r in cvs.iterrows()}
    job_text = {str(r["job_id"]): str(r["text"]) for _, r in jobs.iterrows()}

    model = CrossEncoder(model_name)
    pairs: list[tuple[str, str]] = []
    meta: list[tuple[int, str, str]] = []
    for idx, row in df.iterrows():
        if int(row.get("rank_for_job", 99)) > top_n:
            continue
        jid, cid = str(row["job_id"]), str(row["cv_id"])
        jt, ct = job_text.get(jid, ""), cv_text.get(cid, "")
        pairs.append((jt[:8000], ct[:8000]))
        meta.append((idx, jid, cid))

    if not pairs:
        return df

    logits = model.predict(pairs, show_progress_bar=False)
    scores = np.asarray(logits, dtype=np.float64)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    out = df.copy()
    out["cross_encoder_score"] = np.nan
    if "final_score_v2_bm25" in out.columns:
        out["final_rerank_score"] = out["final_score_v2_bm25"].fillna(
            out["final_score"]
        )
    else:
        out["final_rerank_score"] = out["final_score"].copy()
    for (row_idx, _, _), s in zip(meta, scores_norm, strict=False):
        out.loc[row_idx, "cross_encoder_score"] = float(s)
        base = float(out.loc[row_idx, "final_rerank_score"])
        out.loc[row_idx, "final_rerank_score"] = (
            1 - CROSS_ENCODER_BLEND
        ) * base + CROSS_ENCODER_BLEND * float(s)

    # re-order per job within top_n
    parts: list[pd.DataFrame] = []
    for jid in out["job_id"].astype(str).unique():
        block = out[out["job_id"].astype(str) == jid].copy()
        sub = block[block["rank_for_job"] <= top_n].sort_values(
            "final_rerank_score", ascending=False
        )
        sub = sub.assign(rank_for_job=range(1, len(sub) + 1))
        rest = block[block["rank_for_job"] > top_n]
        parts.append(pd.concat([sub, rest], axis=0))
    return pd.concat(parts, axis=0).sort_index()
