"""Schema checks for Bronze JSONL / ground truth produced by the external import path."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.schemas.documents import validate_ground_truth_df

_FIX = Path(__file__).resolve().parent / "fixtures" / "bronze_mini"


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@pytest.mark.skipif(
    not (_FIX / "resumes_bronze.jsonl").is_file(), reason="fixture missing"
)
def test_resumes_bronze_schema() -> None:
    for row in _iter_jsonl(_FIX / "resumes_bronze.jsonl"):
        for key in (
            "resume_id",
            "source",
            "source_file",
            "raw_text",
            "language",
            "labels",
            "metadata",
        ):
            assert key in row
        assert isinstance(row["labels"], dict) and "entities" in row["labels"]
        assert isinstance(row["labels"]["entities"], list)
        assert row["raw_text"].strip()


@pytest.mark.skipif(
    not (_FIX / "jobs_bronze.jsonl").is_file(), reason="fixture missing"
)
def test_jobs_bronze_schema() -> None:
    for row in _iter_jsonl(_FIX / "jobs_bronze.jsonl"):
        for key in (
            "job_id",
            "source",
            "source_file",
            "raw_text",
            "title",
            "language",
            "metadata",
        ):
            assert key in row
        assert row["raw_text"].strip()


@pytest.mark.skipif(
    not (_FIX / "ner_annotations_bronze.jsonl").is_file(), reason="fixture missing"
)
def test_ner_annotations_bronze_schema() -> None:
    for row in _iter_jsonl(_FIX / "ner_annotations_bronze.jsonl"):
        for key in (
            "annotation_id",
            "source",
            "source_file",
            "text",
            "entities",
            "metadata",
        ):
            assert key in row
        for ent in row["entities"]:
            assert "start" in ent and "end" in ent and "label" in ent and "text" in ent


@pytest.mark.skipif(not (_FIX / "ground_truth.csv").is_file(), reason="fixture missing")
def test_ground_truth_columns() -> None:
    df = pd.read_csv(_FIX / "ground_truth.csv")
    for c in ("job_id", "resume_id", "relevance", "source"):
        assert c in df.columns
    out = validate_ground_truth_df(df)
    assert "cv_id" in out.columns and "relevant" in out.columns
