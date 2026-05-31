#!/usr/bin/env python3
"""Root dizinindeki yerel datasetleri Bronze pipeline'a aktarır.

Desteklenen kaynaklar
---------------------
1. ``resume_dataset_2.csv`` — 2.000 satırlık yapılandırılmış CV verisi
   (Name, Email, Phone, University, Graduation_Year, Years_Experience,
    Job_Role, Skills, Resume_Text)

2. ``train.json`` — 5.960 adet NER annotasyonlu CV
   Format: [{text, annotations: [[start, end, label], ...]}]
   Zaten Bronze'da bulunan Dataturks kayıtları (metin parmak iziyle)
   tekrar eklenmez.

3. ``Entity Recognition in Resumes.json`` — farklı annotation formatında
   ek NER kayıtları
   Format: [{content, annotation: [{label, points:[{start,end,text}]}]}]

Kullanım
--------
    # Projenin kökünden çalıştırın (cv-matching-data-mining/)
    python scripts/import_local_datasets_to_bronze.py
    python scripts/import_local_datasets_to_bronze.py --dry-run --verbose
    python scripts/import_local_datasets_to_bronze.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path düzeltmesi — projenin kökü sys.path'te olmazsa ekle
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingest.external_bronze_import import (  # noqa: E402
    load_jsonl_by_id,
    stats_for_records,
    write_jsonl,
    write_stats,
    _utf8_json_safe,
)
from src.utils.id_normalization import normalize_cv_id  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
_SOURCE_CSV = "kaggle_resume_dataset_2"
_SOURCE_TRAIN_JSON = "dataturks_resume_ner_local_train"
_SOURCE_ENTITY_JSON = "dataturks_resume_ner_entity_recognition"

_BRONZE_RESUMES = _PROJECT_ROOT / "data" / "bronze" / "resumes" / "resumes_bronze.jsonl"
_BRONZE_RESUMES_STATS = _PROJECT_ROOT / "data" / "bronze" / "resumes" / "resumes_bronze.stats.json"
_BRONZE_NER = _PROJECT_ROOT / "data" / "bronze" / "annotations" / "ner_annotations_bronze.jsonl"
_BRONZE_NER_STATS = _PROJECT_ROOT / "data" / "bronze" / "annotations" / "ner_annotations_bronze.stats.json"

# Kök dizindeki kaynak dosyalar
_DATA_ROOT = _PROJECT_ROOT.parent  # cv_analysis/
_CSV_FILE = _DATA_ROOT / "resume_dataset_2.csv"
_TRAIN_JSON = _DATA_ROOT / "train.json"
_ENTITY_JSON = _DATA_ROOT / "Entity Recognition in Resumes.json"


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _text_fingerprint(text: str) -> str:
    """Metin parmak izi — ilk 300 karakterin MD5'i (büyük/küçük harf normalize)."""
    norm = re.sub(r"\s+", " ", text[:300]).strip().lower()
    return hashlib.md5(norm.encode("utf-8", errors="replace")).hexdigest()


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


# ---------------------------------------------------------------------------
# 1. resume_dataset_2.csv  →  Bronze resumes
# ---------------------------------------------------------------------------

def import_resume_csv(
    *,
    existing_ids: set[str],
    existing_fingerprints: set[str],
    dry_run: bool,
) -> list[dict[str, Any]]:
    """CSV'yi okuyup yeni bronze kayıtlarını döndürür."""
    if not _CSV_FILE.is_file():
        logger.warning("Kaynak dosya bulunamadı: %s", _CSV_FILE)
        return []

    new_records: list[dict[str, Any]] = []
    skipped_dup = 0
    skipped_short = 0

    with _CSV_FILE.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            name = _safe_str(row.get("Name", "")).replace(" ", "_").lower()
            resume_text = _safe_str(row.get("Resume_Text", ""))

            # Metin çok kısaysa atla
            if len(resume_text) < 50:
                skipped_short += 1
                continue

            fingerprint = _text_fingerprint(resume_text)
            if fingerprint in existing_fingerprints:
                skipped_dup += 1
                continue

            slug = _slug(name) if name else f"row_{row_idx:05d}"
            rid = normalize_cv_id(f"cv_csv2_{slug}_{row_idx:05d}")

            # ID çakışması çok nadir ama kontrol et
            if rid in existing_ids:
                rid = normalize_cv_id(f"cv_csv2_{slug}_{row_idx:05d}_b")

            record: dict[str, Any] = {
                "resume_id": rid,
                "source": _SOURCE_CSV,
                "source_file": "resume_dataset_2.csv",
                "raw_text": resume_text,
                "language": "en",
                "metadata": {
                    "name": _safe_str(row.get("Name")),
                    "university": _safe_str(row.get("University")),
                    "graduation_year": _safe_str(row.get("Graduation_Year")),
                    "years_experience": _safe_str(row.get("Years_Experience")),
                    "job_role": _safe_str(row.get("Job_Role")),
                    "skills_raw": _safe_str(row.get("Skills")),
                    # PII (e-posta, telefon) ham formatta sakla ama maskeleme
                    # Silver katmanında yapılacak (src/preprocessing/pii.py).
                    "format": "csv",
                },
            }

            new_records.append(record)
            existing_ids.add(rid)
            existing_fingerprints.add(fingerprint)

    logger.info(
        "[CSV] %d yeni kayıt | %d zaten var (parmak izi) | %d çok kısa",
        len(new_records), skipped_dup, skipped_short,
    )
    return new_records


# ---------------------------------------------------------------------------
# 2. train.json  →  Bronze resumes + NER annotations
# ---------------------------------------------------------------------------

def import_train_json(
    *,
    existing_resume_ids: set[str],
    existing_resume_fps: set[str],
    existing_ann_ids: set[str],
    existing_ann_fps: set[str],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """train.json'ı okuyup (yeni_resumes, yeni_annotation) çiftini döndürür."""
    if not _TRAIN_JSON.is_file():
        logger.warning("Kaynak dosya bulunamadı: %s", _TRAIN_JSON)
        return [], []

    raw = _TRAIN_JSON.read_text(encoding="utf-8", errors="replace")
    data: list = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            data = parsed
        elif isinstance(parsed, dict):
            data = [parsed]
    except json.JSONDecodeError:
        logger.debug("train.json JSON array parse başarısız, JSONL olarak deneniyor…")
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    data.append(obj)
                elif isinstance(obj, list):
                    data.extend(obj)
            except json.JSONDecodeError:
                logger.warning("train.json geçersiz satır %d atlandı", line_no)

    if not data:
        logger.error("train.json'dan hiç kayıt okunamadı")
        return [], []

    new_resumes: list[dict[str, Any]] = []
    new_annotations: list[dict[str, Any]] = []
    skipped_dup_res = skipped_dup_ann = skipped_short = 0

    for idx, item in enumerate(data):
        text = _safe_str(item.get("text", item.get("content", "")))
        raw_annotations = item.get("annotations", [])

        if len(text) < 50:
            skipped_short += 1
            continue

        fp = _text_fingerprint(text)

        # --- Bronze resume kaydı ---
        if fp not in existing_resume_fps:
            slug = _slug(text[:30])
            rid = normalize_cv_id(f"cv_train_{idx:06d}_{slug}")
            if rid in existing_resume_ids:
                rid = normalize_cv_id(f"cv_train_{idx:06d}_{slug}_b")

            new_resumes.append({
                "resume_id": rid,
                "source": _SOURCE_TRAIN_JSON,
                "source_file": "train.json",
                "raw_text": text,
                "language": "en",
                "metadata": {"format": "json", "original_index": idx},
            })
            existing_resume_ids.add(rid)
            existing_resume_fps.add(fp)
        else:
            skipped_dup_res += 1

        # --- NER annotation kaydı ---
        if fp not in existing_ann_fps and raw_annotations:
            ann_id = f"train_ann_{idx:06d}"
            if ann_id in existing_ann_ids:
                ann_id = f"train_ann_{idx:06d}_b"

            # [start, end, label] formatını standart entity dict'e çevir
            entities = []
            for ann in raw_annotations:
                if isinstance(ann, (list, tuple)) and len(ann) >= 3:
                    start, end, label = int(ann[0]), int(ann[1]), str(ann[2])
                    entities.append({
                        "start": start,
                        "end": end,
                        "label": label,
                        "text": text[start:end],
                    })

            new_annotations.append({
                "annotation_id": ann_id,
                "source": _SOURCE_TRAIN_JSON,
                "source_file": "train.json",
                "text": text,
                "entities": entities,
                "metadata": {"original_format": "json", "original_index": idx},
            })
            existing_ann_ids.add(ann_id)
            existing_ann_fps.add(fp)
        else:
            skipped_dup_ann += 1

    logger.info(
        "[train.json] %d/%d yeni resume | %d zaten var | %d çok kısa",
        len(new_resumes), len(data), skipped_dup_res, skipped_short,
    )
    logger.info(
        "[train.json] %d/%d yeni annotation | %d zaten var",
        len(new_annotations), len(data), skipped_dup_ann,
    )
    return new_resumes, new_annotations


# ---------------------------------------------------------------------------
# 3. Entity Recognition in Resumes.json  →  NER annotations (+ resumes)
# ---------------------------------------------------------------------------

def import_entity_json(
    *,
    existing_resume_ids: set[str],
    existing_resume_fps: set[str],
    existing_ann_ids: set[str],
    existing_ann_fps: set[str],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Entity Recognition JSON'ı okuyup (yeni_resumes, yeni_annotation) döndürür."""
    if not _ENTITY_JSON.is_file():
        logger.warning("Kaynak dosya bulunamadı: %s", _ENTITY_JSON)
        return [], []

    # Bu dosya bazen JSONL (satır başına bir nesne), bazen JSON array olabilir.
    # Her iki formatı da destekle.
    data: list = []
    raw = _ENTITY_JSON.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            data = parsed
        elif isinstance(parsed, dict):
            data = [parsed]
        else:
            logger.error("Entity Recognition JSON beklenmedik tip: %s", type(parsed))
            return [], []
    except json.JSONDecodeError:
        # JSON array parse başarısız → JSONL olarak dene
        logger.debug("JSON array parse başarısız, JSONL olarak deneniyor…")
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    data.append(obj)
                elif isinstance(obj, list):
                    data.extend(obj)
            except json.JSONDecodeError:
                logger.warning("Geçersiz JSONL satırı %d atlandı", line_no)

    if not data:
        logger.error("Entity Recognition JSON'dan hiç kayıt okunamadı")
        return [], []

    new_resumes: list[dict[str, Any]] = []
    new_annotations: list[dict[str, Any]] = []
    skipped_dup = skipped_short = 0

    for idx, item in enumerate(data):
        # Bu formatta alan adı "content" veya "text" olabilir
        text = _safe_str(item.get("content", item.get("text", "")))
        raw_annotations = item.get("annotation", [])

        if len(text) < 50:
            skipped_short += 1
            continue

        fp = _text_fingerprint(text)

        # --- Bronze resume ---
        if fp not in existing_resume_fps:
            slug = _slug(text[:30])
            rid = normalize_cv_id(f"cv_entity_{idx:06d}_{slug}")
            if rid in existing_resume_ids:
                rid = normalize_cv_id(f"cv_entity_{idx:06d}_{slug}_b")

            new_resumes.append({
                "resume_id": rid,
                "source": _SOURCE_ENTITY_JSON,
                "source_file": "Entity Recognition in Resumes.json",
                "raw_text": text,
                "language": "en",
                "metadata": {"format": "json", "original_index": idx},
            })
            existing_resume_ids.add(rid)
            existing_resume_fps.add(fp)
        else:
            skipped_dup += 1

        # --- NER annotation  ---
        # Format: [{label: [...], points: [{start, end, text}]}]
        if fp not in existing_ann_fps and raw_annotations:
            ann_id = f"entity_ann_{idx:06d}"
            if ann_id in existing_ann_ids:
                ann_id = f"entity_ann_{idx:06d}_b"

            entities = []
            for ann_group in raw_annotations:
                labels = ann_group.get("label", [])
                label_str = labels[0] if labels else "Unknown"
                for point in ann_group.get("points", []):
                    entities.append({
                        "start": int(point.get("start", 0)),
                        "end": int(point.get("end", 0)),
                        "label": label_str,
                        "text": _safe_str(point.get("text", "")),
                    })

            new_annotations.append({
                "annotation_id": ann_id,
                "source": _SOURCE_ENTITY_JSON,
                "source_file": "Entity Recognition in Resumes.json",
                "text": text,
                "entities": entities,
                "metadata": {"original_format": "dataturks_v2", "original_index": idx},
            })
            existing_ann_ids.add(ann_id)
            existing_ann_fps.add(fp)

    logger.info(
        "[entity_json] %d/%d yeni resume | %d zaten var | %d çok kısa",
        len(new_resumes), len(data), skipped_dup, skipped_short,
    )
    logger.info(
        "[entity_json] %d/%d yeni annotation",
        len(new_annotations), len(data),
    )
    return new_resumes, new_annotations


# ---------------------------------------------------------------------------
# Ana çalıştırıcı
# ---------------------------------------------------------------------------

def run(
    *,
    dry_run: bool = False,
    overwrite_sources: set[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Tüm kaynakları okuyup Bronze'a yazar; özet dict döndürür."""
    overwrite_sources = overwrite_sources or set()

    # Mevcut Bronze kayıtlarını yükle
    logger.info("Mevcut Bronze resumes yükleniyor…")
    existing_resumes: dict[str, dict] = load_jsonl_by_id(_BRONZE_RESUMES, "resume_id")

    if overwrite_sources:
        before = len(existing_resumes)
        existing_resumes = {
            k: v for k, v in existing_resumes.items()
            if _safe_str(v.get("source")) not in overwrite_sources
        }
        logger.info("Overwrite: %d kayıt silindi (%s)", before - len(existing_resumes), overwrite_sources)

    existing_resume_ids: set[str] = set(existing_resumes.keys())
    existing_resume_fps: set[str] = {
        _text_fingerprint(_safe_str(v.get("raw_text", "")))
        for v in existing_resumes.values()
    }

    logger.info("Mevcut Bronze NER annotations yükleniyor…")
    existing_anns: dict[str, dict] = load_jsonl_by_id(_BRONZE_NER, "annotation_id")
    if overwrite_sources:
        existing_anns = {
            k: v for k, v in existing_anns.items()
            if _safe_str(v.get("source")) not in overwrite_sources
        }
    existing_ann_ids: set[str] = set(existing_anns.keys())
    existing_ann_fps: set[str] = {
        _text_fingerprint(_safe_str(v.get("text", "")))
        for v in existing_anns.values()
    }

    summary: dict[str, Any] = {}

    # ---- 1. CSV ----
    csv_records = import_resume_csv(
        existing_ids=existing_resume_ids,
        existing_fingerprints=existing_resume_fps,
        dry_run=dry_run,
    )
    summary["csv_new_resumes"] = len(csv_records)

    # ---- 2. train.json ----
    train_resumes, train_anns = import_train_json(
        existing_resume_ids=existing_resume_ids,
        existing_resume_fps=existing_resume_fps,
        existing_ann_ids=existing_ann_ids,
        existing_ann_fps=existing_ann_fps,
        dry_run=dry_run,
    )
    summary["train_new_resumes"] = len(train_resumes)
    summary["train_new_annotations"] = len(train_anns)

    # ---- 3. Entity Recognition JSON ----
    entity_resumes, entity_anns = import_entity_json(
        existing_resume_ids=existing_resume_ids,
        existing_resume_fps=existing_resume_fps,
        existing_ann_ids=existing_ann_ids,
        existing_ann_fps=existing_ann_fps,
        dry_run=dry_run,
    )
    summary["entity_new_resumes"] = len(entity_resumes)
    summary["entity_new_annotations"] = len(entity_anns)

    if dry_run:
        logger.info("[DRY-RUN] Yazma atlandı.")
        summary["dry_run"] = True
        return summary

    # ---- Bronze resumes yaz ----
    all_resumes = list(existing_resumes.values()) + csv_records + train_resumes + entity_resumes
    all_resumes_sorted = sorted(all_resumes, key=lambda r: _safe_str(r.get("resume_id")))
    written_resumes = write_jsonl(_BRONZE_RESUMES, all_resumes_sorted)
    write_stats(_BRONZE_RESUMES_STATS, stats_for_records(all_resumes_sorted))
    summary["total_resumes_bronze"] = written_resumes

    # ---- Bronze NER annotations yaz ----
    all_anns = list(existing_anns.values()) + train_anns + entity_anns
    all_anns_sorted = sorted(all_anns, key=lambda a: _safe_str(a.get("annotation_id")))
    written_anns = write_jsonl(_BRONZE_NER, all_anns_sorted)
    write_stats(_BRONZE_NER_STATS, stats_for_records(all_anns_sorted))
    summary["total_annotations_bronze"] = written_anns

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Root dizinindeki yerel CSV/JSON datasetlerini Bronze JSONL'a aktarır.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Yalnızca sayım yapar; dosya yazmaz.",
    )
    ap.add_argument(
        "--overwrite",
        nargs="*",
        metavar="SOURCE_TAG",
        default=None,
        help=(
            "Belirtilen source tag'lere ait mevcut Bronze kayıtları silinip yeniden yazılır. "
            "Argümansız kullanılırsa tüm local kayıtlar sıfırlanır: "
            f"{_SOURCE_CSV!r}, {_SOURCE_TRAIN_JSON!r}, {_SOURCE_ENTITY_JSON!r}."
        ),
    )
    ap.add_argument("--verbose", action="store_true", help="DEBUG log seviyesi.")
    args = ap.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    overwrite_sources: set[str] = set()
    if args.overwrite is not None:
        if args.overwrite:
            overwrite_sources = set(args.overwrite)
        else:
            # Argümansız --overwrite → hepsini sıfırla
            overwrite_sources = {_SOURCE_CSV, _SOURCE_TRAIN_JSON, _SOURCE_ENTITY_JSON}

    summary = run(
        dry_run=args.dry_run,
        overwrite_sources=overwrite_sources,
        verbose=args.verbose,
    )

    print("\n=== İmport Özeti ===")
    print(f"  CSV (resume_dataset_2.csv)          : +{summary.get('csv_new_resumes', 0):,} resume")
    print(f"  train.json                          : +{summary.get('train_new_resumes', 0):,} resume | +{summary.get('train_new_annotations', 0):,} annotation")
    print(f"  Entity Recognition in Resumes.json  : +{summary.get('entity_new_resumes', 0):,} resume | +{summary.get('entity_new_annotations', 0):,} annotation")
    if not args.dry_run:
        print(f"  Bronze Resumes TOPLAM               : {summary.get('total_resumes_bronze', 0):,}")
        print(f"  Bronze NER Annotations TOPLAM       : {summary.get('total_annotations_bronze', 0):,}")
    else:
        print("  [DRY-RUN] Dosyalara hiçbir şey yazılmadı.")
    print()


if __name__ == "__main__":
    main()
