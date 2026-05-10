from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.helpers import resolve_path
from src.utils.id_normalization import normalize_cv_id

logger = logging.getLogger(__name__)


def load_cv_rows_from_jsonl(
    path: Path,
    *,
    id_field: str = "record_id",
    text_field: str = "text",
    id_prefix: str = "corpus_",
    max_rows: int | None = None,
    source_allowlist: list[str] | None = None,
) -> list[dict[str, str]]:
    """Load CV text rows from JSONL; optional ``source_allowlist`` filter."""
    if not path.is_file():
        logger.warning("CV corpus JSONL not found, skipped: %s", path)
        return []
    allow_src = {str(s).strip() for s in (source_allowlist or []) if str(s).strip()} or None
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
            if allow_src is not None:
                src = str(obj.get("source", "")).strip()
                if not src or src not in allow_src:
                    continue
            raw_id = obj.get(id_field)
            if raw_id is None:
                raw_id = (
                    obj.get("resume_id")
                    or obj.get("cv_id")
                    or obj.get("record_id")
                    or obj.get("id")
                )
            if raw_id is None:
                continue
            text = obj.get(text_field)
            if text is None:
                for ak in (
                    "cleaned_text",
                    "raw_text",
                    "resume_text",
                    "cv_text",
                    "text",
                ):
                    v = obj.get(ak)
                    if v and str(v).strip():
                        text = v
                        break
            stext = str(text).strip() if text is not None else ""
            if len(stext) < 40:
                continue
            rid = normalize_cv_id(raw_id)
            if not rid:
                continue
            cid = normalize_cv_id(f"{id_prefix}{rid}" if id_prefix else rid)
            if not cid:
                continue
            sfile = str(obj.get("source_file", "") or "").strip()
            sour = str(obj.get("source", "") or "").strip()
            rows.append({"cv_id": cid, "text": stext, "source": sour, "source_file": sfile})
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
    rk = ingest_cfg.get("ranking_sources")
    rk_list: list[str] | None = None
    if isinstance(rk, list) and rk:
        rk_list = [str(x).strip() for x in rk if str(x).strip()]
    return load_cv_rows_from_jsonl(
        path,
        id_field=str(ds.get("id_field", "record_id")),
        text_field=str(ds.get("text_field", "text")),
        id_prefix=str(ds.get("id_prefix", "corpus_")),
        max_rows=ds.get("max_rows"),
        source_allowlist=rk_list,
    )


def ner_profile_extensions_from_jsonl(
    root: Path, ingest_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load NER-labelled corpus rows (not for ranking) for Silver profiling."""
    ds = ingest_cfg.get("cv_corpus_jsonl") or {}
    if not ds.get("enabled", False):
        return []
    tags = ingest_cfg.get("ner_corpus_sources") or []
    ner_only: list[str] = []
    if isinstance(tags, list):
        for x in tags:
            s = str(x).strip()
            if s:
                ner_only.append(s)
    if not ner_only:
        return []
    rel = ds.get("path")
    if not rel:
        return []
    path = resolve_path(root, str(rel))
    rows_plain = load_cv_rows_from_jsonl(
        path,
        id_field=str(ds.get("id_field", "record_id")),
        text_field=str(ds.get("text_field", "text")),
        id_prefix=str(ds.get("id_prefix", "corpus_")),
        max_rows=None,
        source_allowlist=ner_only,
    )
    bronze_rows: list[dict[str, Any]] = []
    for r in rows_plain:
        t = str(r.get("text", "")).strip()
        cid = str(r.get("cv_id", "")).strip()
        if not cid:
            continue
        bronze_rows.append(
            {
                "cv_id": cid,
                "source": str(r.get("source", "") or ""),
                "source_file": str(r.get("source_file", "") or ""),
                "resume_id": cid,
                "raw_text": t,
                "cleaned_text": t,
                "text": t,
            }
        )
    return bronze_rows
