#!/usr/bin/env python3
"""CV Matching Data Mining — multi-channel scoring pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.orchestrator import run_full_pipeline
from src.utils.helpers import load_config
from src.utils.logging_config import setup_logging


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="CV–job matching pipeline")
    ap.add_argument("--config", type=Path, default=None, help="YAML config path")
    ap.add_argument(
        "--ingest",
        action="store_true",
        help="Rebuild Silver CSVs from data/bronze (PDF/DOCX/TXT/MD)",
    )

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

    eval_group = ap.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Force-on offline evaluation (requires ground_truth.csv)",
    )
    eval_group.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip offline evaluation even if ground_truth.csv exists",
    )

    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    if args.semantic:
        semantic = True
    elif args.no_semantic:
        semantic = False
    else:
        semantic = bool(cfg.get("embeddings", {}).get("enabled", True))

    if args.evaluate:
        evaluate: bool | None = True
    elif args.no_evaluate:
        evaluate = False
    else:
        evaluate = None

    run_full_pipeline(
        root,
        cfg,
        ingest=args.ingest,
        semantic=semantic,
        evaluate=evaluate,
    )


if __name__ == "__main__":
    main()
