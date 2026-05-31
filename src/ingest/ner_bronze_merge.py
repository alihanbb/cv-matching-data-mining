"""Map ``ner_annotations_bronze.jsonl`` rows onto normalized ``cv_id`` for Silver ingest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.id_normalization import normalize_cv_id

logger = logging.getLogger(__name__)


def resume_id_from_annotation_id(annotation_id: str) -> str | None:
    """Derive synthetic resume id used by ``external_bronze_import`` (``*_ann_*`` infix)."""
    aid = str(annotation_id or "").strip()
    if not aid:
        return None
    if "_ann_" in aid:
        return aid.replace("_ann_", "_", 1)
    return None


def _cv_id_for_ner_row(obj: dict[str, Any]) -> str | None:
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    for key in ("resume_id", "cv_id"):
        raw = meta.get(key)
        if raw is not None and str(raw).strip():
            cid = normalize_cv_id(str(raw).strip())
            if cid:
                return cid
    rid = resume_id_from_annotation_id(str(obj.get("annotation_id", "") or ""))
    if rid:
        return normalize_cv_id(rid)
    return None


def load_ner_entities_by_resume_id(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Index NER annotation lines by normalized ``cv_id`` (last row wins per id for duplicates)."""
    if not path.is_file():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    n_lines = 0
    n_skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            if not isinstance(obj, dict):
                n_skipped += 1
                continue
            cid = _cv_id_for_ner_row(obj)
            if not cid:
                n_skipped += 1
                continue
            raw_ents = obj.get("entities")
            if not isinstance(raw_ents, list):
                n_skipped += 1
                continue
            norm_ents: list[dict[str, Any]] = []
            for e in raw_ents:
                if isinstance(e, dict) and e.get("label") is not None:
                    norm_ents.append(e)
            if norm_ents:
                out[cid] = norm_ents
    if out:
        logger.info(
            "Bronze NER annotations: indexed %d cv_ids from %s (%d lines, %d skipped)",
            len(out),
            path.name,
            n_lines,
            n_skipped,
        )
    elif n_lines:
        logger.warning(
            "Bronze NER annotations: no rows indexed from %s (%d lines, %d skipped)",
            path,
            n_lines,
            n_skipped,
        )
    return out


def merge_ner_labels_into_profile_rows(
    rows: list[dict[str, Any]], ner_by_cv: dict[str, list[dict[str, Any]]]
) -> int:
    """Fill ``row['labels']['entities']`` from ``ner_by_cv`` when missing or empty. Returns merge count."""
    merged = 0
    for r in rows:
        cid = normalize_cv_id(str(r.get("cv_id", "") or ""))
        if not cid:
            continue
        ents = ner_by_cv.get(cid)
        if not ents:
            continue
        lab = r.get("labels")
        if not isinstance(lab, dict):
            lab = {}
        existing = lab.get("entities")
        if isinstance(existing, list) and existing:
            continue
        r["labels"] = {**lab, "entities": list(ents)}
        merged += 1
    return merged
