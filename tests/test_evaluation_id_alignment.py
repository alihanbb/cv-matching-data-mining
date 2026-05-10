import pandas as pd

from src.evaluation.compare_models import _normalize_ranked_ids
from src.schemas.documents import validate_ground_truth_df


def test_ground_truth_and_ranked_ids_align_after_normalization() -> None:
    gt_raw = pd.DataFrame(
        {
            "job_id": ["vanetik_vacancy_001", "vanetik_vacancy_001"],
            "resume_id": ["corpus_vanetik_cv_014", "vanetik_cv_999"],
            "relevance": [3, 1],
        }
    )
    gt = validate_ground_truth_df(gt_raw)

    ranked_raw = pd.DataFrame(
        {
            "job_id": ["vanetik_vacancy_001", "vanetik_vacancy_001", "vanetik_vacancy_001"],
            "cv_id": [
                "corpus_corpus_vanetik_cv_014",
                "corpus_vanetik_cv_014",
                "vanetik_cv_999",
            ],
            "score": [0.91, 0.87, 0.50],
            "rank_for_job": [2, 3, 1],
        }
    )
    ranked = _normalize_ranked_ids(ranked_raw)

    merged = gt.merge(ranked[["job_id", "cv_id"]], on=["job_id", "cv_id"], how="inner")
    assert len(merged) == 2
    assert set(merged["cv_id"].tolist()) == {"vanetik_cv_014", "vanetik_cv_999"}

