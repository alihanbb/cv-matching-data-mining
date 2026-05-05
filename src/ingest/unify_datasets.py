from __future__ import annotations

import argparse
import csv
import hashlib
import json
from json import JSONDecodeError
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.ingest.text_extract import extract_text_from_path


def _stable_id(*parts: str) -> str:
    raw = "::".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _make_record(
    *,
    record_id: str,
    source: str,
    source_file: Path,
    source_row: int | None,
    text: str,
    category: str | None = None,
    labels: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "source": source,
        "source_file": str(source_file),
        "source_row": source_row,
        "document_type": "resume",
        "category": category,
        "language": "en",
        "text": text,
        "text_length": len(text),
        "labels": labels or {},
        "metadata": metadata or {},
        "extraction_status": status,
        "error": error,
    }


def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
            return

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        yield data


def iter_ner_json(path: Path, source_name: str) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    for idx, item in enumerate(_iter_json_records(path)):
        text = _safe_text(item.get("text"))
        annotations = item.get("annotations") if isinstance(item.get("annotations"), list) else []
        yield _make_record(
            record_id=_stable_id(source_name, str(idx), text[:100]),
            source=source_name,
            source_file=path,
            source_row=idx,
            text=text,
            labels={"entities": annotations},
            metadata={"annotation_count": len(annotations)},
            status="ok" if text else "empty_text",
            error=None if text else "empty_text",
        )


def iter_structured_resume_csv(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = _safe_text(row.get("Resume_Text"))
            skills = [
                x.strip()
                for x in _safe_text(row.get("Skills")).split(",")
                if x.strip()
            ]
            metadata = {
                "name": _safe_text(row.get("Name")),
                "email": _safe_text(row.get("Email")),
                "phone": _safe_text(row.get("Phone")),
                "university": _safe_text(row.get("University")),
                "graduation_year": _safe_text(row.get("Graduation_Year")),
                "years_experience": _safe_text(row.get("Years_Experience")),
            }
            yield _make_record(
                record_id=_stable_id("resume_dataset_2_csv", str(idx), metadata["email"], text[:100]),
                source="resume_dataset_2_csv",
                source_file=path,
                source_row=idx,
                text=text,
                category=_safe_text(row.get("Job_Role")) or None,
                labels={"skills": skills, "job_role": _safe_text(row.get("Job_Role"))},
                metadata=metadata,
                status="ok" if text else "empty_text",
                error=None if text else "empty_text",
            )


def iter_resume_corpus_csv(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = _safe_text(row.get("Resume_str"))
            external_id = _safe_text(row.get("ID"))
            category = _safe_text(row.get("Category")) or None
            yield _make_record(
                record_id=_stable_id("resume_corpus_csv", external_id or str(idx), text[:100]),
                source="resume_corpus_csv",
                source_file=path,
                source_row=idx,
                text=text,
                category=category,
                labels={"category": category},
                metadata={"external_id": external_id, "has_html": bool(_safe_text(row.get("Resume_html")))},
                status="ok" if text else "empty_text",
                error=None if text else "empty_text",
            )


def iter_category_pdfs(pdf_root: Path, limit: int = 0) -> Iterable[dict[str, Any]]:
    if not pdf_root.is_dir():
        return
    count = 0
    for path in sorted(pdf_root.rglob("*.pdf")):
        if limit and count >= limit:
            break
        category = path.parent.name
        count += 1
        try:
            text = extract_text_from_path(path)
            status = "ok" if text else "empty_text"
            error = None if text else "empty_text"
        except Exception as exc:
            text = ""
            status = "extract_error"
            error = f"{type(exc).__name__}: {exc}"
        yield _make_record(
            record_id=_stable_id("category_pdf", category, path.stem),
            source="category_pdf",
            source_file=path,
            source_row=None,
            text=text,
            category=category,
            labels={"category": category},
            metadata={"filename": path.name},
            status=status,
            error=error,
        )


def write_unified_dataset(
    *,
    source_root: Path,
    output_path: Path,
    include_pdfs: bool,
    pdf_limit: int,
) -> dict[str, Any]:
    sources = [
        iter_ner_json(source_root / "train.json", "train_json_ner"),
        iter_ner_json(source_root / "sample.json", "sample_json_ner"),
        iter_ner_json(source_root / "Entity Recognition in Resumes.json", "entity_recognition_json"),
        iter_structured_resume_csv(source_root / "resume_dataset_2.csv"),
        iter_resume_corpus_csv(source_root / "Resume" / "Resume.csv"),
    ]
    if include_pdfs:
        sources.append(iter_category_pdfs(source_root / "data" / "data", limit=pdf_limit))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "output_path": str(output_path),
        "source_root": str(source_root),
        "include_pdfs": include_pdfs,
        "pdf_limit": pdf_limit,
        "records": 0,
        "empty_or_error_records": 0,
        "by_source": Counter(),
        "by_status": Counter(),
        "by_category": Counter(),
    }

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for iterator in sources:
            for record in iterator:
                # Some public resume datasets contain unpaired surrogate characters.
                # ASCII escaping keeps JSONL valid without dropping the record.
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                stats["records"] += 1
                stats["by_source"][record["source"]] += 1
                stats["by_status"][record["extraction_status"]] += 1
                if record["category"]:
                    stats["by_category"][record["category"]] += 1
                if record["extraction_status"] != "ok":
                    stats["empty_or_error_records"] += 1

    # Convert Counters for JSON serialization.
    stats["by_source"] = dict(stats["by_source"])
    stats["by_status"] = dict(stats["by_status"])
    stats["by_category"] = dict(stats["by_category"])

    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    stats["stats_path"] = str(stats_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify all resume datasets into one canonical JSONL file.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(".."),
        help="Root containing train.json, resume CSVs, and data/data PDFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/silver/unified_resumes.jsonl"),
        help="Canonical output JSONL path.",
    )
    parser.add_argument("--include-pdfs", action="store_true", help="Also extract and include data/data/*.pdf.")
    parser.add_argument("--pdf-limit", type=int, default=0, help="Limit PDF extraction count (0 = no limit).")
    args = parser.parse_args()

    stats = write_unified_dataset(
        source_root=args.source_root.resolve(),
        output_path=args.output,
        include_pdfs=args.include_pdfs,
        pdf_limit=args.pdf_limit,
    )
    print(
        "Unified dataset written: "
        f"{stats['records']} records, "
        f"{stats['empty_or_error_records']} empty/error -> {stats['output_path']}"
    )
    print(f"Stats: {stats['stats_path']}")


if __name__ == "__main__":
    main()
