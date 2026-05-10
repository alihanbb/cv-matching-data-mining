#!/usr/bin/env python3
"""Merge ``cv_analysis/data/data/<category>/*`` into Bronze ``resumes_bronze.jsonl``.

Default corpus path is the sibling layout::

    cv_analysis/
      cv-matching-data-mining/   <- run from here
      data/data/ACCOUNTANT/*.pdf

Examples::

    python scripts/import_cv_analysis_data_to_bronze.py --dry-run --verbose
    python scripts/import_cv_analysis_data_to_bronze.py --overwrite
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingest.cv_analysis_folder_import import run_cli  # noqa: E402

if __name__ == "__main__":
    run_cli(default_project_root=_ROOT)
