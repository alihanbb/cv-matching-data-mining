import pandas as pd

from src.evaluation.ranking_metrics import mean_reciprocal_rank, ndcg_at_k


def test_mrr_perfect():
    ranked = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j2", "j2"],
            "cv_id": ["a", "b", "c", "d"],
            "rank_for_job": [1, 2, 1, 2],
            "score": [0.9, 0.1, 0.8, 0.2],
        }
    )
    gt = pd.DataFrame(
        {
            "job_id": ["j1", "j2"],
            "cv_id": ["a", "c"],
            "relevant": [1, 1],
        }
    )
    assert mean_reciprocal_rank(ranked, gt) == 1.0


def test_ndcg_positive():
    ranked = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j1"],
            "cv_id": ["a", "b", "c"],
            "rank_for_job": [1, 2, 3],
            "score": [0.9, 0.5, 0.1],
        }
    )
    gt = pd.DataFrame({"job_id": ["j1"], "cv_id": ["a"], "relevant": [1]})
    assert ndcg_at_k(ranked, gt, k=3) > 0.9
