import pandas as pd

from src.evaluation.metrics import recall_at_k


def test_recall_at_k() -> None:
    ranked = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j1"],
            "cv_id": ["a", "b", "c"],
            "rank_for_job": [1, 2, 3],
            "score": [0.9, 0.5, 0.1],
        }
    )
    gt = pd.DataFrame({"job_id": ["j1", "j1"], "cv_id": ["a", "b"], "relevant": [2, 1]})
    r = recall_at_k(ranked, gt, k=2)
    assert abs(r - 1.0) < 1e-6
