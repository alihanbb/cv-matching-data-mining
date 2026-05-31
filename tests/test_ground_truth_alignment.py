from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.alignment import build_alignment_report
from src.utils.id_normalization import normalize_cv_id, normalize_job_id


def test_normalize_cv_id_docx_variant() -> None:
    assert normalize_cv_id("CV_014.docx") == "vanetik_cv_014"


def test_normalize_job_id_vacancy_variant() -> None:
    assert normalize_job_id("vacancy_1") == "vanetik_vacancy_001"


def test_alignment_report_reaches_reasonable_match_ratio(tmp_path: Path) -> None:
    gt = pd.DataFrame(
        {
            "job_id": ["vacancy_1", "vacancy_01", "vanetik_vacancy_1"],
            "resume_id": ["CV_014.docx", "014", "cv014"],
            "relevance": [3, 2, 1],
        }
    )
    scores = pd.DataFrame(
        {
            "job_id": ["vanetik_vacancy_001", "vacancy_1"],
            "cv_id": ["vanetik_cv_014", "cv_014"],
            "ranking_score": [0.9, 0.8],
        }
    )

    gt_path = tmp_path / "ground_truth.csv"
    score_path = tmp_path / "candidate_scores_explained.csv"
    unmatched_path = tmp_path / "unmatched_ground_truth_examples.csv"
    gt.to_csv(gt_path, index=False)
    scores.to_csv(score_path, index=False)

    report = build_alignment_report(gt_path, score_path, unmatched_path)

    assert report["total_ground_truth_rows"] == 3
    assert report["matched_ground_truth_rows"] == 3
    assert report["match_ratio"] >= 0.5
    assert unmatched_path.is_file()
    unmatched = pd.read_csv(unmatched_path)
    assert unmatched.empty
