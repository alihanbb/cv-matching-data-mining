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
        raw_ann = item.get("annotations")
        annotations: list[Any]
        if (
            isinstance(raw_ann, list)
            and raw_ann
            and isinstance(raw_ann[0], (list, tuple))
        ):
            annotations = _mehyar_triples_to_entities(raw_ann)
        elif isinstance(raw_ann, list):
            annotations = raw_ann
        else:
            annotations = []
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


def _dataturks_annotation_to_entities(annotation: Any) -> list[dict[str, Any]]:
    if not isinstance(annotation, list):
        return []
    out: list[dict[str, Any]] = []
    for ann in annotation:
        if not isinstance(ann, dict):
            continue
        pts = ann.get("points")
        if not isinstance(pts, list) or not pts:
            continue
        p0 = pts[0]
        if not isinstance(p0, dict):
            continue
        start, end = p0.get("start"), p0.get("end")
        if start is None or end is None:
            continue
        labels = ann.get("label")
        lab_list: list[Any]
        if isinstance(labels, list):
            lab_list = labels
        elif labels is not None:
            lab_list = [labels]
        else:
            continue
        for lab in lab_list:
            if lab is None:
                continue
            out.append({"start": int(start), "end": int(end) + 1, "label": str(lab)})
    return out


def iter_dataturks_resume_jsonl(
    path: Path, source_name: str
) -> Iterable[dict[str, Any]]:
    """DataTurks export: one JSON object per line with ``content`` and ``annotation``."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            text = _safe_text(item.get("content"))
            entities = _dataturks_annotation_to_entities(item.get("annotation"))
            yield _make_record(
                record_id=_stable_id(source_name, str(idx), text[:100]),
                source=source_name,
                source_file=path,
                source_row=idx,
                text=text,
                labels={"entities": entities},
                metadata={"annotation_count": len(entities)},
                status="ok" if text else "empty_text",
                error=None if text else "empty_text",
            )


def _mehyar_triples_to_entities(annotations: Any) -> list[dict[str, Any]]:
    if not isinstance(annotations, list):
        return []
    out: list[dict[str, Any]] = []
    for t in annotations:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            out.append({"start": int(t[0]), "end": int(t[1]), "label": str(t[2])})
    return out


def iter_mehyar_annotated_json_dir(
    root: Path, source_name: str
) -> Iterable[dict[str, Any]]:
    """Mehyarmlaweh/NER-Annotated-CVs: ``text`` + ``annotations`` as [start, end, label] triples."""
    if not root.is_dir():
        return
    for idx, path in enumerate(sorted(root.glob("*.json"))):
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        text = _safe_text(data.get("text"))
        entities = _mehyar_triples_to_entities(data.get("annotations"))
        yield _make_record(
            record_id=_stable_id(source_name, path.stem, text[:80]),
            source=source_name,
            source_file=path,
            source_row=idx,
            text=text,
            labels={"entities": entities},
            metadata={"annotation_count": len(entities)},
            status="ok" if text else "empty_text",
            error=None if text else "empty_text",
        )


def json_resume_schema_to_text(obj: dict[str, Any]) -> str:
    """Flatten [jsonresume.org](https://jsonresume.org/schema/)-style JSON to plain text."""
    parts: list[str] = []
    basics = obj.get("basics") if isinstance(obj.get("basics"), dict) else {}
    if basics:
        for key in ("name", "label", "email", "phone", "summary"):
            v = basics.get(key)
            if v and str(v).strip():
                parts.append(str(v).strip())
        loc = basics.get("location")
        if isinstance(loc, dict) and loc.get("address"):
            parts.append(str(loc["address"]).strip())
    for edu in obj.get("education") or []:
        if not isinstance(edu, dict):
            continue
        chunk = " ".join(
            _safe_text(edu.get(k))
            for k in ("studyType", "area", "institution", "score", "unlabeled")
            if edu.get(k)
        )
        if chunk:
            parts.append(chunk)
    for work in obj.get("work") or []:
        if not isinstance(work, dict):
            continue
        header = " ".join(
            _safe_text(work.get(k)) for k in ("position", "name") if work.get(k)
        )
        if header:
            parts.append(header)
        for hl in work.get("highlights") or []:
            if hl and str(hl).strip():
                parts.append(str(hl).strip())
        u = work.get("unlabeled")
        if u and str(u).strip():
            parts.append(str(u).strip())
    for sk in obj.get("skills") or []:
        if not isinstance(sk, dict):
            continue
        kws = sk.get("keywords")
        if isinstance(kws, list) and kws:
            parts.append(", ".join(str(x) for x in kws if x))
    return "\n\n".join(p for p in parts if p).strip()


def iter_json_resume_file(path: Path, source_name: str) -> Iterable[dict[str, Any]]:
    """Single demo resume in JSON Resume schema (e.g. NLP_NER_ON_RESUME/resume.json)."""
    if not path.is_file():
        return
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (JSONDecodeError, OSError):
        return
    if not isinstance(obj, dict) or "basics" not in obj:
        return
    text = json_resume_schema_to_text(obj)
    basics = obj.get("basics") if isinstance(obj.get("basics"), dict) else {}
    name = _safe_text(basics.get("name"))
    yield _make_record(
        record_id=_stable_id(source_name, path.name, text[:100]),
        source=source_name,
        source_file=path,
        source_row=0,
        text=text,
        category=_safe_text(basics.get("label")) or None,
        labels={"schema": "json_resume"},
        metadata={"name": name},
        status="ok" if text else "empty_text",
        error=None if text else "empty_text",
    )


def _mehyar_annotated_root(workspace: Path) -> Path | None:
    for rel in (
        Path("NER-Annotated-CVs") / "extracted" / "ResumesJsonAnnotated",
        Path("NER-Annotated-CVs") / "ResumesJsonAnnotated",
    ):
        p = workspace / rel
        if p.is_dir():
            return p
    return None


def iter_workspace_cloned_repos(workspace: Path) -> list[Iterable[dict[str, Any]]]:
    """Sister clones under the same folder as this project (e.g. ``cv_analysis``)."""
    out: list[Iterable[dict[str, Any]]] = []
    dt = workspace / "Entity-Recognition-In-Resumes-SpaCy"
    train_f = dt / "traindata.json"
    test_f = dt / "testdata.json"
    if train_f.is_file():
        out.append(iter_dataturks_resume_jsonl(train_f, "dataturks_resume_ner_train"))
    if test_f.is_file():
        out.append(iter_dataturks_resume_jsonl(test_f, "dataturks_resume_ner_test"))
    ner_root = _mehyar_annotated_root(workspace)
    if ner_root is not None:
        out.append(iter_mehyar_annotated_json_dir(ner_root, "mehyar_ner_annotated_cv"))
    nlp_ner = workspace / "NLP_NER_ON_RESUME" / "resume.json"
    if nlp_ner.is_file():
        out.append(iter_json_resume_file(nlp_ner, "nlp_ner_on_resume_json_demo"))
    return out


def iter_structured_resume_csv(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = _safe_text(row.get("Resume_Text"))
            skills = [
                x.strip() for x in _safe_text(row.get("Skills")).split(",") if x.strip()
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
                record_id=_stable_id(
                    "resume_dataset_2_csv", str(idx), metadata["email"], text[:100]
                ),
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
                record_id=_stable_id(
                    "resume_corpus_csv", external_id or str(idx), text[:100]
                ),
                source="resume_corpus_csv",
                source_file=path,
                source_row=idx,
                text=text,
                category=category,
                labels={"category": category},
                metadata={
                    "external_id": external_id,
                    "has_html": bool(_safe_text(row.get("Resume_html"))),
                },
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
    include_workspace_clones: bool = True,
) -> dict[str, Any]:
    sources: list[Iterable[dict[str, Any]]] = [
        iter_ner_json(source_root / "train.json", "train_json_ner"),
        iter_ner_json(source_root / "sample.json", "sample_json_ner"),
        iter_ner_json(
            source_root / "Entity Recognition in Resumes.json",
            "entity_recognition_json",
        ),
        iter_structured_resume_csv(source_root / "resume_dataset_2.csv"),
        iter_resume_corpus_csv(source_root / "Resume" / "Resume.csv"),
    ]
    if include_pdfs:
        sources.append(
            iter_category_pdfs(source_root / "data" / "data", limit=pdf_limit)
        )
    if include_workspace_clones:
        sources.extend(iter_workspace_cloned_repos(source_root.resolve()))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "output_path": str(output_path),
        "source_root": str(source_root),
        "include_pdfs": include_pdfs,
        "pdf_limit": pdf_limit,
        "include_workspace_clones": include_workspace_clones,
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
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stats["stats_path"] = str(stats_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unify all resume datasets into one canonical JSONL file."
    )
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
    parser.add_argument(
        "--include-pdfs",
        action="store_true",
        help="Also extract and include data/data/*.pdf.",
    )
    parser.add_argument(
        "--pdf-limit",
        type=int,
        default=0,
        help="Limit PDF extraction count (0 = no limit).",
    )
    parser.add_argument(
        "--no-workspace-clones",
        action="store_true",
        help="Skip GitHub sister folders (Entity-Recognition-*, NER-Annotated-CVs, NLP_NER_ON_RESUME) next to --source-root.",
    )
    args = parser.parse_args()

    stats = write_unified_dataset(
        source_root=args.source_root.resolve(),
        output_path=args.output,
        include_pdfs=args.include_pdfs,
        pdf_limit=args.pdf_limit,
        include_workspace_clones=not args.no_workspace_clones,
    )
    print(
        "Unified dataset written: "
        f"{stats['records']} records, "
        f"{stats['empty_or_error_records']} empty/error -> {stats['output_path']}"
    )
    print(f"Stats: {stats['stats_path']}")


if __name__ == "__main__":
    main()
