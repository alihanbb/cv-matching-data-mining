from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.defaults import WEIGHT_SEARCH_N_TRIALS
from src.evaluation.ranking_metrics import ndcg_at_k
from src.pipeline.matching_inputs import (
    MatchingMatrices,
    build_matching_matrices,
    rankings_from_fused,
)
from src.schemas.documents import validate_ground_truth_df
from src.scoring.fusion import fuse_scores
from src.utils.helpers import ensure_parent, resolve_path

logger = logging.getLogger(__name__)


def _random_weight_candidates(
    channel_keys: list[str],
    n_trials: int,
    *,
    rng_seed: int | None = None,
) -> list[dict[str, float]]:
    """Return ``n_trials`` weight dicts sampled uniformly from the simplex.

    Uses Dirichlet(alpha=1) which is equivalent to a uniform distribution over
    the probability simplex — no new dependencies required (numpy only).
    """
    rng = np.random.default_rng(rng_seed)
    n = len(channel_keys)
    # alpha=1 → uniform distribution over the n-simplex
    samples = rng.dirichlet(np.ones(n), size=n_trials)
    return [dict(zip(channel_keys, row.tolist())) for row in samples]


def optimize_weights(
    root: Path,
    cfg: dict[str, Any],
    *,
    semantic: bool = True,
    use_bm25: bool = False,
    k_target: int = 5,
    n_trials: int | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    gt_path = resolve_path(
        root, cfg["paths"].get("ground_truth", "data/evaluation/ground_truth.csv")
    )
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth required for weight search: {gt_path}")

    trials = (
        n_trials
        if n_trials is not None
        else int(cfg.get("weight_search", {}).get("n_trials", WEIGHT_SEARCH_N_TRIALS))
    )
    gt = validate_ground_truth_df(pd.read_csv(gt_path))
    mats = build_matching_matrices(root, cfg, semantic=semantic, bm25=use_bm25)
    top_k = max(int(cfg.get("matching", {}).get("top_k", 10)), k_target)

    if use_bm25 and mats.bm25_enabled and mats.bm25 is not None:
        channel_keys = ["tfidf", "dense", "bm25", "skills", "experience"]
    else:
        channel_keys = ["tfidf", "dense", "skills", "experience"]

    candidates = _random_weight_candidates(channel_keys, trials)
    logger.info("Random weight search: %d candidates, channels=%s", trials, channel_keys)

    rows: list[dict[str, Any]] = []
    best_w: dict[str, float] | None = None
    best_ndcg = -1.0

    for w in candidates:
        if use_bm25 and mats.bm25_enabled and mats.bm25 is not None:
            w_full = dict(w)
            fused, _ = fuse_scores(
                mats.sim_lex,
                mats.dense_sim,
                mats.skill_score,
                mats.exp_mat,
                w_full,
                mats.dense_enabled,
                bm25=mats.bm25,
            )
        else:
            w_full = {**w, "bm25": 0.0}
            fused, _ = fuse_scores(
                mats.sim_lex,
                mats.dense_sim,
                mats.skill_score,
                mats.exp_mat,
                w_full,
                mats.dense_enabled,
                bm25=None,
            )
        ranked = rankings_from_fused(fused, mats.cv_ids, mats.job_ids, top_k)
        nd = float(ndcg_at_k(ranked, gt, k_target))
        rows.append({**w_full, "ndcg_at_k": nd, "k": k_target})
        if nd > best_ndcg:
            best_ndcg = nd
            best_w = w_full

    if best_w is None:
        raise RuntimeError("Weight search failed — no candidates evaluated.")

    df = pd.DataFrame(rows).sort_values("ndcg_at_k", ascending=False)
    art = resolve_path(root, "artifacts/best_fusion_weights.json")
    ensure_parent(art)
    with open(art, "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": best_w,
                "ndcg_at_5": best_ndcg,
                "bm25": use_bm25,
                "n_trials": trials,
            },
            f,
            indent=2,
        )
    ev_dir = resolve_path(root, "data/gold/evaluation")
    ev_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ev_dir / "weight_search_results.csv", index=False)
    logger.info("Best weights %s  NDCG@%d=%.4f", best_w, k_target, best_ndcg)
    return best_w, df
