from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.helpers import resolve_path

logger = logging.getLogger(__name__)


def load_cv_rows_from_jsonl(
    path: Path,
    *,
    id_field: str = "record_id",
    text_field: str = "text",
    id_prefix: str = "corpus_",
    max_rows: int | None = None,
) -> list[dict[str, str]]:
    """Load CV text rows from JSONL (e.g. Indeed / NER resume dumps)."""
    if not path.is_file():
        logger.warning("CV corpus JSONL not found, skipped: %s", path)
        return []
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_id = obj.get(id_field)
            if raw_id is None:
                raw_id = obj.get("cv_id") or obj.get("record_id") or obj.get("id")
            if raw_id is None:
                continue
            text = obj.get(text_field)
            if text is None:
                for ak in ("cleaned_text", "raw_text", "resume_text", "cv_text", "text"):
                    v = obj.get(ak)
                    if v and str(v).strip():
                        text = v
                        break
            stext = str(text).strip() if text is not None else ""
            if len(stext) < 40:
                continue
            cid = f"{id_prefix}{raw_id}"
            rows.append({"cv_id": cid, "text": stext})
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    logger.info("Loaded %d CV rows from JSONL corpus: %s", len(rows), path)
    return rows


def extra_cvs_from_ingest_config(root: Path, ingest_cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Optional JSONL résumé corpus (paths relative to project root)."""
    ds = ingest_cfg.get("cv_corpus_jsonl") or {}
    if not ds.get("enabled", False):
        return []
    rel = ds.get("path")
    if not rel:
        return []
    path = resolve_path(root, str(rel))
    return load_cv_rows_from_jsonl(
        path,
        id_field=str(ds.get("id_field", "record_id")),
        text_field=str(ds.get("text_field", "text")),
        id_prefix=str(ds.get("id_prefix", "corpus_")),
        max_rows=ds.get("max_rows"),
    )
