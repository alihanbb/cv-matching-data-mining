#!/usr/bin/env python3
"""CLI entry: sibling cloned CV/NER repos → ``data/bronze/*.jsonl``.

Example from project root (``cv-matching-data-mining``)::

    python scripts/import_external_repos_to_bronze.py --source-root .. --all --overwrite
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingest.external_bronze_import import run_cli

if __name__ == "__main__":
    run_cli(default_project_root=_ROOT)
