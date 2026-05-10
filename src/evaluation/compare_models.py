from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.defaults import FUSION_V1_WEIGHTS, FUSION_V2_WEIGHTS
from src.evaluation.metrics import precision_at_k, recall_at_k, top_k_accuracy
from src.evaluation.ranking_metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
)
from src.pipeline.matching_inputs import (
    MatchingMatrices,
    build_matching_matrices,
    rankings_from_fused,
)
from src.schemas.documents import validate_ground_truth_df
from src.scoring.fusion import fuse_weighted_raw
from src.utils.helpers import ensure_parent, resolve_path

logger = logging.getLogger(__name__)


def _rank_tfidf_baseline(m: MatchingMatrices, top_k: int) -> pd.DataFrame:
    w = {"tfidf": 1.0, "dense": 0.0, "skills": 0.0, "experience": 0.0}
    fused, _ = fuse_weighted_raw(m.sim_lex, None, m.skill_score, m.exp_mat, w, False, bm25=None)
    return rankings_from_fused(fused, m.cv_ids, m.job_ids, top_k)


def _rank_semantic_only(m: MatchingMatrices, top_k: int) -> pd.DataFrame:
    w = {"tfidf": 0.0, "dense": 1.0, "skills": 0.0, "experience": 0.0}
    fused, _ = fuse_weighted_raw(
        m.sim_lex, m.dense_sim, m.skill_score, m.exp_mat, w, m.dense_enabled, bm25=None
    )
    return rankings_from_fused(fused, m.cv_ids, m.job_ids, top_k)


def _rank_hybrid_v1(m: MatchingMatrices, top_k: int, cfg: dict[str, Any]) -> pd.DataFrame:
    w = dict(cfg.get("fusion", {}).get("weights", {})) or dict(FUSION_V1_WEIGHTS)
    fused, _ = fuse_weighted_raw(
        m.sim_lex, m.dense_sim, m.skill_score, m.exp_mat, w, m.dense_enabled, bm25=None
    )
    return rankings_from_fused(fused, m.cv_ids, m.job_ids, top_k)


def _rank_hybrid_v2(m: MatchingMatrices, top_k: int, cfg: dict[str, Any]) -> pd.DataFrame:
    w = dict(cfg.get("fusion_v2", {}).get("weights", {})) or dict(FUSION_V2_WEIGHTS)
    if not m.bm25_enabled or m.bm25 is None:
        return _rank_hybrid_v1(m, top_k, cfg)
    fused, _ = fuse_weighted_raw(
        m.sim_lex,
        m.dense_sim,
        m.skill_score,
        m.exp_mat,
        w,
        m.dense_enabled,
        bm25=m.bm25,
    )
    return rankings_from_fused(fused, m.cv_ids, m.job_ids, top_k)


def _rank_optimized(m: MatchingMatrices, top_k: int, root: Path) -> pd.DataFrame:
    art = resolve_path(root, "artifacts/best_fusion_weights.json")
    with open(art, encoding="utf-8") as f:
        payload = json.load(f)
    w = dict(payload.get("weights", {}))
    bm25_w = float(w.pop("bm25", 0.0))
    if m.bm25 is not None and m.bm25_enabled and bm25_w > 0:
        w_full = {**w, "bm25": bm25_w}
        fused, _ = fuse_weighted_raw(
            m.sim_lex,
            m.dense_sim,
            m.skill_score,
            m.exp_mat,
            w_full,
            m.dense_enabled,
            bm25=m.bm25,
        )
    else:
        fused, _ = fuse_weighted_raw(
            m.sim_lex,
            m.dense_sim,
            m.skill_score,
            m.exp_mat,
            w,
            m.dense_enabled,
            bm25=None,
        )
    return rankings_from_fused(fused, m.cv_ids, m.job_ids, top_k)


def evaluate_models(
    root: Path,
    cfg: dict[str, Any],
    *,
    semantic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = cfg.get("paths", {})
    gt_rel = paths_cfg.get("ground_truth", "data/evaluation/ground_truth.csv")
    gt_path = resolve_path(root, gt_rel)
    if not gt_path.is_file():
        logger.warning(
            "Evaluation skipped: %s not found.",
            gt_rel,
        )
        return pd.DataFrame(), pd.DataFrame()
    gt = validate_ground_truth_df(pd.read_csv(gt_path))
    top_k = int(cfg.get("matching", {}).get("top_k", 10))
    ks = [int(k) for k in cfg.get("evaluation", {}).get("top_k_values", [1, 3, 5])]

    m_base = build_matching_matrices(root, cfg, semantic=False, bm25=False)
    m_sem = build_matching_matrices(root, cfg, semantic=semantic, bm25=False)
    m_bm25 = build_matching_matrices(root, cfg, semantic=semantic, bm25=True)

    spec: list[tuple[str, pd.DataFrame]] = [
        ("TF-IDF Baseline", _rank_tfidf_baseline(m_base, top_k)),
        ("Semantic Only", _rank_semantic_only(m_sem, top_k)),
        ("Hybrid V1", _rank_hybrid_v1(m_sem, top_k, cfg)),
        ("Hybrid V2 + BM25", _rank_hybrid_v2(m_bm25, top_k, cfg)),
    ]
    art = resolve_path(root, "artifacts/best_fusion_weights.json")
    if art.is_file():
        spec.append(("Optimized Fusion", _rank_optimized(m_bm25, top_k, root)))

    eval_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for name, ranked in spec:
        row: dict[str, Any] = {"model": name}
        for k in ks:
            row[f"precision_at_{k}"] = precision_at_k(ranked, gt, k)
            row[f"recall_at_{k}"] = recall_at_k(ranked, gt, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(ranked, gt, k)
            row[f"topk_hit_rate_{k}"] = top_k_accuracy(ranked, gt, k)
        row["mrr"] = mean_reciprocal_rank(ranked, gt)
        row["map"] = mean_average_precision(ranked, gt)
        eval_rows.append(row)
        comp_rows.append({"model": name, **{f"ndcg_at_{k}": row[f"ndcg_at_{k}"] for k in ks}})

    eval_df = pd.DataFrame(eval_rows)
    comp_df = pd.DataFrame(comp_rows)
    eval_out = resolve_path(
        root,
        paths_cfg.get("evaluation_results_csv", "data/gold/evaluation/evaluation_results.csv"),
    )
    comp_out = resolve_path(
        root,
        paths_cfg.get("model_comparison_csv", "data/gold/evaluation/model_comparison.csv"),
    )
    ensure_parent(eval_out)
    ensure_parent(comp_out)
    eval_df.to_csv(eval_out, index=False)
    comp_df.to_csv(comp_out, index=False)
    logger.info("Wrote evaluation exports: %s, %s", eval_out, comp_out)
    return eval_df, comp_df
