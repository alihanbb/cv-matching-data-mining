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
from src.models.learned_fusion import (
    DEFAULT_FEATURE_COLS,
    export_learned_fusion_weights_json,
    predict_learned_fusion,
    save_learned_fusion_model,
    train_learned_fusion,
)
from src.pipeline.matching_inputs import (
    MatchingMatrices,
    build_matching_matrices,
    rankings_from_fused,
)
from src.schemas.documents import validate_ground_truth_df
from src.scoring.fusion import fuse_weighted_raw
from src.utils.candidate_dedup import dedupe_candidates_by_canonical_cv_id
from src.utils.helpers import ensure_parent, resolve_path
from src.utils.id_normalization import normalize_cv_id, normalize_job_id

logger = logging.getLogger(__name__)


def _normalize_ranked_ids(ranked: pd.DataFrame) -> pd.DataFrame:
    out = ranked.copy()
    out["cv_id"] = out["cv_id"].map(normalize_cv_id)
    out["job_id"] = out["job_id"].map(normalize_job_id)
    out = out[(out["cv_id"] != "") & (out["job_id"] != "")]
    if out.empty:
        return out

    if "score" in out.columns:
        score_col = "score"
    elif "rank_for_job" in out.columns:
        out["score"] = -pd.to_numeric(out["rank_for_job"], errors="coerce")
        score_col = "score"
    else:
        out["score"] = 0.0
        score_col = "score"
    return dedupe_candidates_by_canonical_cv_id(
        out,
        score_column=score_col,
        keep_canonical_column=False,
    )


def _log_gt_alignment(model_name: str, ranked: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
    gt_pairs = ground_truth[["job_id", "cv_id"]].drop_duplicates()
    ranked_pairs = ranked[["job_id", "cv_id"]].drop_duplicates()
    merged = gt_pairs.merge(ranked_pairs, on=["job_id", "cv_id"], how="left", indicator=True)
    total = int(len(merged))
    matched = int((merged["_merge"] == "both").sum())
    unmatched = total - matched
    logger.info(
        "%s evaluation alignment: total_ground_truth_rows=%d matched_ground_truth_rows=%d unmatched_ground_truth_rows=%d",
        model_name,
        total,
        matched,
        unmatched,
    )
    if total > 0 and (matched / total) < 0.10:
        logger.warning(
            "%s: ground-truth alignment is very low (matched_ground_truth_rows=%d/%d). Check cv_id/job_id normalization and source consistency.",
            model_name,
            matched,
            total,
        )


def _rank_tfidf_baseline(m: MatchingMatrices, top_k: int) -> pd.DataFrame:
    w = {"tfidf": 1.0, "dense": 0.0, "skills": 0.0, "experience": 0.0}
    fused, _ = fuse_weighted_raw(m.sim_lex, None, m.skill_score, m.exp_mat, w, False, bm25=None)
    return rankings_from_fused(fused, m.cv_ids, m.job_ids, top_k)


def _rank_semantic_only(m: MatchingMatrices, top_k: int) -> pd.DataFrame:
    if not m.dense_enabled or m.dense_sim is None:
        return _rank_tfidf_baseline(m, top_k)
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


def _rank_from_score_df(scored_df: pd.DataFrame, score_col: str, top_k: int) -> pd.DataFrame:
    if score_col not in scored_df.columns:
        return pd.DataFrame(columns=["job_id", "cv_id", "score", "rank_for_job"])
    ranked = scored_df[["job_id", "cv_id", score_col]].rename(columns={score_col: "score"}).copy()
    ranked = _normalize_ranked_ids(ranked)
    ranked = ranked.sort_values(["job_id", "score"], ascending=[True, False], kind="mergesort")
    ranked["rank_for_job"] = ranked.groupby("job_id").cumcount() + 1
    ranked = ranked.groupby("job_id", sort=False).head(top_k).reset_index(drop=True)
    return ranked


def _maybe_rank_learned_fusion(
    root: Path,
    cfg: dict[str, Any],
    gt: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame | None:
    paths_cfg = cfg.get("paths", {})
    explained_path = resolve_path(
        root,
        paths_cfg.get("output_explanations", "data/gold/rankings/candidate_scores_explained.csv"),
    )
    if not explained_path.is_file():
        return None

    scores_df = pd.read_csv(explained_path)
    if "learned_fusion_score" not in scores_df.columns or scores_df["learned_fusion_score"].isna().all():
        try:
            model = train_learned_fusion(
                scores_df,
                gt,
                feature_cols=list(DEFAULT_FEATURE_COLS),
                target_col="relevant",
                epochs=100,
                lr=0.01,
            )
        except (ValueError, ImportError) as exc:
            logger.warning("Learned fusion evaluation skipped: %s", exc)
            return None

        model_path = resolve_path(root, "artifacts/models/learned_fusion.pt")
        weights_path = resolve_path(root, "artifacts/models/learned_fusion_weights.json")
        save_learned_fusion_model(model, model_path)
        export_learned_fusion_weights_json(model, list(DEFAULT_FEATURE_COLS), weights_path)
        scores_df["learned_fusion_score"] = predict_learned_fusion(
            scores_df, model, feature_cols=list(DEFAULT_FEATURE_COLS)
        )
        scores_df.to_csv(explained_path, index=False)

    if scores_df["learned_fusion_score"].isna().all():
        return None
    return _rank_from_score_df(scores_df, "learned_fusion_score", top_k)


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
    # eval_top_k: evaluation uses a larger pool than the pipeline output top_k.
    # This ensures ground-truth CVs ranked 11-50 are still reachable for metrics.
    matching_top_k = int(cfg.get("matching", {}).get("top_k", 10))
    top_k = int(cfg.get("evaluation", {}).get("eval_top_k", max(matching_top_k, 50)))
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
    lf_ranked = _maybe_rank_learned_fusion(root, cfg, gt, top_k)
    if lf_ranked is not None and not lf_ranked.empty:
        spec.append(("Learned Fusion", lf_ranked))

    eval_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for name, ranked in spec:
        ranked = _normalize_ranked_ids(ranked)
        _log_gt_alignment(name, ranked, gt)
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
