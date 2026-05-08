import numpy as np
import pytest

pytest.importorskip("rank_bm25")

from src.features.bm25_scorer import bm25_matrix


def test_bm25_matrix_normalized_columns() -> None:
    jobs = ["python machine learning engineer kubernetes", "data scientist sql"]
    cvs = [
        "python developer with ml experience",
        "kubernetes admin and docker",
        "sql analyst with bi tools",
    ]
    m = bm25_matrix(jobs, cvs)
    assert m.shape == (3, 2)
    assert np.all(m >= 0) and np.all(m <= 1 + 1e-9)
    for j in range(m.shape[1]):
        assert m[:, j].max() <= 1.0 + 1e-9
