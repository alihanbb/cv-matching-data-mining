#!/usr/bin/env python3
"""CV Matching Data Mining — multi-channel scoring pipeline entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.compare_models import evaluate_models
from src.models.cross_encoder_rerank import rerank_with_cross_encoder
from src.pipeline.orchestrator import run_full_pipeline
from src.training.learned_fusion import train_learned_fusion
from src.training.weight_optimizer import optimize_weights
from src.utils.helpers import load_config
from src.utils.logging_config import setup_logging


def _apply_best_weights(cfg: dict, root: Path) -> None:
    art = root / "artifacts" / "best_fusion_weights.json"
    if not art.is_file():
        raise FileNotFoundError(
            f"Optimized weights not found: {art}\nRun: python main.py --optimize-weights"
        )
    with open(art, encoding="utf-8") as f:
        payload = json.load(f)
    w = payload.get("weights", {})
    if not w:
        raise ValueError(f"Invalid weights file: {art}")
    cfg.setdefault("fusion", {})["weights"] = {k: float(v) for k, v in w.items() if k != "bm25"}
    bm25_v = float(w.get("bm25", 0.0) or 0.0)
    if bm25_v > 0:
        cfg.setdefault("bm25", {})["enabled"] = True
        cfg.setdefault("fusion_v2", {})["weights"] = {k: float(v) for k, v in w.items()}


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="CV–job matching pipeline")
    ap.add_argument("--config", type=Path, default=None, help="YAML config path")

    ap.add_argument(
        "--ingest",
        action="store_true",
        help="Rebuild Silver CSVs from data/bronze (PDF/DOCX/TXT/MD)",
    )

    # --- Semantic channel -------------------------------------------------
    semantic_group = ap.add_mutually_exclusive_group()
    semantic_group.add_argument(
        "--semantic",
        action="store_true",
        help="Force-enable dense embedding channel (sentence-transformers)",
    )
    semantic_group.add_argument(
        "--no-semantic",
        action="store_true",
        help="Force-disable dense embeddings (TF-IDF + structured channels only)",
    )

    # --- Evaluation toggle -----------------------------------------------
    eval_group = ap.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run offline metrics when paths.ground_truth exists (skipped with log if missing)",
    )
    eval_group.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip offline evaluation even if ground_truth.csv exists",
    )

    ap.add_argument(
        "--bm25",
        action="store_true",
        help="Enable BM25 channel and Hybrid V2 ranking fusion",
    )
    ap.add_argument(
        "--optimize-weights",
        action="store_true",
        help="Grid-search fusion weights to maximize NDCG@5 (needs ground truth)",
    )
    ap.add_argument(
        "--use-best-weights",
        action="store_true",
        help="Load artifacts/best_fusion_weights.json into fusion config before run",
    )
    ap.add_argument(
        "--train-fusion",
        action="store_true",
        help="Train softmax-constrained fusion on ground truth (PyTorch)",
    )
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="Cross-encoder rerank top-20 from explained CSV",
    )
    ap.add_argument(
        "--export-eval-csv",
        action="store_true",
        help="Write data/gold/evaluation/*.csv model comparison without full re-run",
    )

    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    # Validate config against schema at startup — catches typos / bad values early.
    from src.config.schema import PipelineConfig
    from pydantic import ValidationError

    try:
        PipelineConfig.model_validate(cfg)
    except ValidationError as exc:
        import sys

        print(f"[CONFIG ERROR] config.yaml failed validation:\n{exc}", file=sys.stderr)
        sys.exit(1)

    if args.use_best_weights:
        _apply_best_weights(cfg, root)

    if args.semantic:
        semantic = True
    elif args.no_semantic:
        semantic = False
    else:
        semantic = bool(cfg.get("embeddings", {}).get("enabled", True))

    if args.evaluate:
        evaluate_flag: bool | None = True
    elif args.no_evaluate:
        evaluate_flag = False
    else:
        evaluate_flag = None

    if args.optimize_weights:
        optimize_weights(root, cfg, semantic=semantic, use_bm25=args.bm25)
        return

    if args.train_fusion:
        train_learned_fusion(root, cfg, semantic=semantic)
        return

    if args.export_eval_csv:
        evaluate_models(root, cfg, semantic=semantic)
        return

    if args.rerank:
        explain_path = root / "data" / "gold" / "rankings" / "candidate_scores_explained.csv"
        if not explain_path.is_file():
            raise FileNotFoundError(
                f"Missing {explain_path}. Run python main.py (with --bm25 recommended) first."
            )
        out = rerank_with_cross_encoder(root, cfg, explain_path)
        out.to_csv(explain_path, index=False)
        return

    run_full_pipeline(
        root,
        cfg,
        ingest=args.ingest,
        semantic=semantic,
        evaluate=evaluate_flag,
        bm25=args.bm25,
    )


if __name__ == "__main__":
    main()
