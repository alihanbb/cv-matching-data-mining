import pandas as pd

from src.utils.dashboard_ranking import prepare_candidate_ranking_view


def test_prepare_candidate_ranking_view_sorts_by_selected_score_and_recomputes_rank() -> None:
    df = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j1", "j1"],
            "cv_id": [
                "corpus_corpus_vanetik_cv_014",
                "vanetik_cv_999",
                "corpus_vanetik_cv_014",
                "vanetik_cv_111",
            ],
            "source": [
                "vacancy_resume_matching",
                "vacancy_resume_matching",
                "vacancy_resume_matching",
                "vacancy_resume_matching",
            ],
            # Intentionally stale/inconsistent with selected score.
            "rank_for_job": [1, 2, 3, 4],
            "final_score_v2_bm25": [0.70, 0.95, 0.90, 0.20],
        }
    )

    out = prepare_candidate_ranking_view(
        df,
        job_id="j1",
        score_column="final_score_v2_bm25",
        top_n=10,
        include_ner_sources=False,
    )

    assert out["cv_id"].tolist() == ["vanetik_cv_999", "vanetik_cv_014", "vanetik_cv_111"]
    assert out["rank_for_job"].tolist() == [1, 2, 3]
    assert out["final_score_v2_bm25"].tolist() == sorted(
        out["final_score_v2_bm25"].tolist(), reverse=True
    )

