"""Read canonical Bronze-layer JSONL under ``data/bronze/{resumes,jobs,annotations}/``.

The matching pipeline consumes only these standard paths — not cloned external repos.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from json import JSONDecodeError
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BRONZE_RESUMES_JSONL = Path("data/bronze/resumes/resumes_bronze.jsonl")
DEFAULT_BRONZE_JOBS_JSONL = Path("data/bronze/jobs/jobs_bronze.jsonl")
DEFAULT_BRONZE_NER_JSONL = Path("data/bronze/annotations/ner_annotations_bronze.jsonl")


def bronze_resumes_path(root: Path, cfg: dict[str, Any] | None = None) -> Path:
    rel = (cfg or {}).get("bronze_resumes_jsonl") or str(DEFAULT_BRONZE_RESUMES_JSONL)
    return root / rel


def bronze_jobs_path(root: Path, cfg: dict[str, Any] | None = None) -> Path:
    rel = (cfg or {}).get("bronze_jobs_jsonl") or str(DEFAULT_BRONZE_JOBS_JSONL)
    return root / rel


def bronze_ner_annotations_path(root: Path, cfg: dict[str, Any] | None = None) -> Path:
    rel = (cfg or {}).get("bronze_ner_annotations_jsonl") or str(DEFAULT_BRONZE_NER_JSONL)
    return root / rel


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", errors="replace", newline="\n") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d in %s", line_no, path)
                continue
            if isinstance(obj, dict):
                yield obj
