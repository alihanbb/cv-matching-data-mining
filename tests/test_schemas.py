import pandas as pd
import pytest

from src.schemas.documents import validate_ground_truth_df, validate_processed_df


def test_validate_processed():
    df = pd.DataFrame({"cv_id": [1], "text": ["hello world"]})
    out = validate_processed_df(df, "cv_id", "text")
    assert len(out) == 1


def test_ground_truth_invalid():
    df = pd.DataFrame({"cv_id": [1], "job_id": [2], "relevant": [2]})
    with pytest.raises(ValueError):
        validate_ground_truth_df(df)
