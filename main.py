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
    ap.add_argument("--ingest", action="store_true", help="Rebuild Silver CSVs from data/bronze")
    ap.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable dense embeddings (TF-IDF + structured channels only)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    run_full_pipeline(root, cfg, ingest=args.ingest, semantic=not args.no_semantic)


if __name__ == "__main__":
    main()
