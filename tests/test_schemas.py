import pandas as pd
import pytest

from src.schemas.documents import validate_ground_truth_df, validate_processed_df


def test_validate_processed():
    df = pd.DataFrame({"cv_id": [1], "text": ["hello world"]})
    out = validate_processed_df(df, "cv_id", "text")
    assert len(out) == 1


def test_ground_truth_accepts_graded_labels():
    df = pd.DataFrame(
        {"cv_id": [1, 2, 3, 4], "job_id": [10, 10, 11, 11], "relevant": [0, 1, 2, 3]}
    )
    out = validate_ground_truth_df(df)
    assert sorted(out["relevant"].tolist()) == [0, 1, 2, 3]


def test_ground_truth_accepts_relevance_alias():
    df = pd.DataFrame({"cv_id": [1, 2], "job_id": [10, 10], "relevance": [3, 0]})
    out = validate_ground_truth_df(df)


def test_ground_truth_accepts_resume_id_alias():
    df = pd.DataFrame({"resume_id": [1], "job_id": [2], "relevance": [2]})
    out = validate_ground_truth_df(df)
    assert "cv_id" in out.columns


def test_ground_truth_invalid_out_of_range():
    df = pd.DataFrame({"cv_id": [1], "job_id": [2], "relevant": [5]})
    with pytest.raises(ValueError):
        validate_ground_truth_df(df)
