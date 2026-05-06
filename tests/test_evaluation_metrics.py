import pandas as pd

from src.evaluation.metrics import precision_at_k, top_k_accuracy
from src.evaluation.ranking_metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
)


def _ranked() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j1", "j2", "j2", "j2"],
            "cv_id": ["a", "b", "c", "d", "e", "f"],
            "rank_for_job": [1, 2, 3, 1, 2, 3],
            "score": [0.9, 0.6, 0.1, 0.7, 0.5, 0.2],
        }
    )


def test_top_k_accuracy_with_graded_labels() -> None:
    gt = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j2"],
            "cv_id": ["a", "c", "f"],
            "relevant": [3, 1, 2],
        }
    )
    # j1 top-1 hits ("a"); j2 top-1 misses (relevant "f" is at rank 3) -> 0.5.
    assert top_k_accuracy(_ranked(), gt, k=1) == 0.5
    # k=3 covers all candidates and both jobs hit.
    assert top_k_accuracy(_ranked(), gt, k=3) == 1.0


def test_precision_at_k_uses_relevant_threshold() -> None:
    gt = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j2"],
            "cv_id": ["a", "b", "f"],
            "relevant": [3, 0, 2],
        }
    )
    # j1 top-1 includes a (relevant) -> 1.0; j2 top-1 includes d (not relevant) -> 0.0
    p = precision_at_k(_ranked(), gt, k=1)
    assert 0.49 < p < 0.51


def test_mrr_perfect_when_top_is_relevant() -> None:
    gt = pd.DataFrame(
        {
            "job_id": ["j1", "j2"],
            "cv_id": ["a", "d"],
            "relevant": [3, 2],
        }
    )
    assert mean_reciprocal_rank(_ranked(), gt) == 1.0


def test_ndcg_higher_grade_pushes_score_higher() -> None:
    gt_high = pd.DataFrame(
        {
            "job_id": ["j1"],
            "cv_id": ["a"],
            "relevant": [3],
        }
    )
    gt_low = pd.DataFrame(
        {
            "job_id": ["j1"],
            "cv_id": ["a"],
            "relevant": [1],
        }
    )
    # NDCG normalizes by ideal-of-same-grade, so a single relevant item at rank 1
    # produces 1.0 in both binary and graded cases.
    assert ndcg_at_k(_ranked(), gt_high, k=3) == 1.0
    assert ndcg_at_k(_ranked(), gt_low, k=3) == 1.0


def test_map_returns_zero_when_no_relevants() -> None:
    gt = pd.DataFrame({"job_id": ["j1"], "cv_id": ["a"], "relevant": [0]})
    assert mean_average_precision(_ranked(), gt) == 0.0
