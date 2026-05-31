"""Bronze NER annotations → profile row merge."""

from __future__ import annotations

import json
from pathlib import Path

from src.ingest.ner_bronze_merge import (
    load_ner_entities_by_resume_id,
    merge_ner_labels_into_profile_rows,
    resume_id_from_annotation_id,
)


def test_resume_id_from_annotation_id_dataturks_style() -> None:
    assert resume_id_from_annotation_id("dataturks_train_ann_000042") == "dataturks_train_000042"
    assert resume_id_from_annotation_id("mehyar_ann_000001") == "mehyar_000001"


def test_resume_id_from_annotation_id_unknown() -> None:
    assert resume_id_from_annotation_id("ann_001") is None
    assert resume_id_from_annotation_id("") is None


def test_load_ner_index_uses_metadata_resume_id(tmp_path: Path) -> None:
    ner = tmp_path / "ner.jsonl"
    ner.write_text(
        json.dumps(
            {
                "annotation_id": "ann_001",
                "source": "x",
                "source_file": "f.json",
                "text": "hello",
                "entities": [{"start": 0, "end": 2, "label": "X", "text": "he"}],
                "metadata": {"resume_id": "cv_alpha_01"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    idx = load_ner_entities_by_resume_id(ner)
    assert "cv_alpha_01" in idx
    assert len(idx["cv_alpha_01"]) == 1


def test_merge_skips_when_entities_already_present() -> None:
    rows = [
        {
            "cv_id": "cv1",
            "labels": {"entities": [{"label": "OLD", "text": "x"}]},
        }
    ]
    ner_idx = {"cv1": [{"label": "NEW", "text": "y"}]}
    assert merge_ner_labels_into_profile_rows(rows, ner_idx) == 0
    assert rows[0]["labels"]["entities"][0]["label"] == "OLD"


def test_merge_fills_empty_entities() -> None:
    rows = [{"cv_id": "cv1", "labels": {"entities": []}}]
    ner_idx = {"cv1": [{"label": "SKILL", "text": "Python"}]}
    assert merge_ner_labels_into_profile_rows(rows, ner_idx) == 1
    assert rows[0]["labels"]["entities"][0]["label"] == "SKILL"
