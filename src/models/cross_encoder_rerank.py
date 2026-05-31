"""Cross-encoder based reranking for improved candidate selection.

Phase 2 Upgrade: Enhanced reranking with better models and adaptive strategies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.defaults import CROSS_ENCODER_BLEND, CROSS_ENCODER_MODEL_NAME

logger = logging.getLogger(__name__)


# Phase 2 Upgrade: Better cross-encoder models
# Priority: performance vs speed trade-off
CROSS_ENCODER_MODELS = {
    "fast": "cross-encoder/ms-marco-TinyBERT-L-2-v2",  # Fast, low memory
    "balanced": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Balance (default)
    "accurate": "cross-encoder/gtr-t5-base",  # More accurate, slower
    "msmarco_v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # V2 improvements
}


def rerank_with_cross_encoder(
    root: Path,
    cfg: dict[str, Any],
    explained_path: Path,
    *,
    top_n: int = 20,
    model_name: str = CROSS_ENCODER_MODEL_NAME,
    model_variant: str = "balanced",
    use_adaptive: bool = True,
) -> pd.DataFrame:
    """Rerank candidates using cross-encoder model.
    
    Phase 2 Upgrades:
    - Multiple model variants (fast/balanced/accurate)
    - Adaptive reranking strategy
    - Improved score normalization
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError("sentence-transformers required for cross-encoder reranking.") from e

    # Select model variant if specified
    if model_variant in CROSS_ENCODER_MODELS:
        actual_model = CROSS_ENCODER_MODELS[model_variant]
    else:
        actual_model = model_name
    
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

    logger.info("Loading cross-encoder model: %s", actual_model)
    model = CrossEncoder(actual_model, max_length=512)

    # Initialise meta lists so both branches are always defined.
    meta: list[tuple[int, str, str]] = []
    meta_second: list[tuple[int, str, str]] = []
    scores = np.array([], dtype=np.float64)

    # Phase 2 Upgrade: Adaptive candidate selection
    # First stage: rerank top-k candidates more thoroughly
    if use_adaptive:
        # Two-stage reranking: rough first, then detailed
        pairs_first_stage: list[tuple[str, str]] = []
        meta_first_stage: list[tuple[int, str, str, float]] = []
        
        for idx, row in df.iterrows():
            # More candidates in first stage
            if int(row.get("rank_for_job", 99)) > min(top_n * 2, 50):
                continue
            jid, cid = str(row["job_id"]), str(row["cv_id"])
            jt, ct = job_text.get(jid, ""), cv_text.get(cid, "")
            # Truncate to reduce memory
            pairs_first_stage.append((jt[:4000], ct[:4000]))
            meta_first_stage.append((idx, jid, cid, float(row.get("ranking_score", 0))))
        
        if pairs_first_stage:
            logger.info("First stage reranking: %d candidates", len(pairs_first_stage))
            logits_first = model.predict(pairs_first_stage, show_progress_bar=True, batch_size=32)
            
            # Get top candidates for second stage
            first_scores = np.asarray(logits_first, dtype=np.float64)
            top_indices = np.argsort(-first_scores)[:top_n * 2]
            
            # Second stage: detailed reranking with full text
            pairs_second: list[tuple[str, str]] = []
            
            for i in top_indices:
                idx, jid, cid, _ = meta_first_stage[i]
                jt, ct = job_text.get(jid, ""), cv_text.get(cid, "")
                pairs_second.append((jt[:8000], ct[:8000]))
                meta_second.append((idx, jid, cid))
            
            logger.info("Second stage reranking: %d candidates", len(pairs_second))
            logits_second = model.predict(pairs_second, show_progress_bar=True, batch_size=16)
            scores = np.asarray(logits_second, dtype=np.float64)
            
        else:
            scores = np.array([], dtype=np.float64)
    else:
        # Original single-stage approach
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

    # Phase 2 Upgrade: Improved score normalization
    if len(scores) > 0:
        # Sigmoid normalization for better distribution
        scores_norm = 1 / (1 + np.exp(-scores))  # Sigmoid
        
        # Min-max normalization as backup
        score_min, score_max = scores_norm.min(), scores_norm.max()
        if score_max - score_min > 1e-9:
            scores_norm = (scores_norm - score_min) / (score_max - score_min)
    else:
        scores_norm = np.array([])

    # Apply reranking scores
    out = df.copy()
    out["cross_encoder_score"] = np.nan
    
    # Select base score for blending
    if "final_score_v2_bm25" in out.columns:
        out["final_rerank_score"] = out["final_score_v2_bm25"].fillna(out["final_score"])
    else:
        out["final_rerank_score"] = out["final_score"].copy()
    
    # Apply cross-encoder scores
    if use_adaptive and len(meta_second) > 0:
        for (idx, _, _), s in zip(meta_second, scores_norm, strict=False):
            out.loc[idx, "cross_encoder_score"] = float(s)
            base = float(out.loc[idx, "final_rerank_score"])
            out.loc[idx, "final_rerank_score"] = (
                1 - CROSS_ENCODER_BLEND
            ) * base + CROSS_ENCODER_BLEND * float(s)
    elif len(meta) > 0:
        for (row_idx, _, _), s in zip(meta, scores_norm, strict=False):
            out.loc[row_idx, "cross_encoder_score"] = float(s)
            base = float(out.loc[row_idx, "final_rerank_score"])
            out.loc[row_idx, "final_rerank_score"] = (
                1 - CROSS_ENCODER_BLEND
            ) * base + CROSS_ENCODER_BLEND * float(s)

    # Re-order per job
    parts: list[pd.DataFrame] = []
    rerank_top_n = top_n if not use_adaptive else top_n * 2
    
    for jid in out["job_id"].astype(str).unique():
        block = out[out["job_id"].astype(str) == jid].copy()
        sub = block[block["rank_for_job"] <= rerank_top_n].sort_values(
            "final_rerank_score", ascending=False
        )
        sub = sub.assign(rank_for_job=range(1, len(sub) + 1))
        rest = block[block["rank_for_job"] > rerank_top_n]
        parts.append(pd.concat([sub, rest], axis=0))
    
    return pd.concat(parts, axis=0).sort_index()


def get_cross_encoder_info() -> dict[str, Any]:
    """Get information about available cross-encoder models."""
    return {
        "models": CROSS_ENCODER_MODELS,
        "default": CROSS_ENCODER_MODEL_NAME,
        "description": {
            "fast": "Best for low latency, ~80% accuracy",
            "balanced": "Good balance of speed and accuracy (recommended)",
            "accurate": "Highest accuracy, slower inference",
            "msmarco_v2": "Improved MS MARCO model, good general performance",
        },
    }