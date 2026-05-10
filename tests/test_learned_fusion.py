from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.learned_fusion import (
    DEFAULT_FEATURE_COLS,
    predict_learned_fusion,
    train_learned_fusion,
)
from src.pipeline.orchestrator import run_full_pipeline


def _base_cfg(root: Path) -> dict:
    silver = root / "data" / "silver"
    gold = root / "data" / "gold"
    return {
        "paths": {
            "processed_cvs": str(silver / "cleaned_cvs.csv"),
            "processed_jobs": str(silver / "cleaned_jobs.csv"),
            "tfidf_model": str(gold / "models" / "tfidf_model.pkl"),
            "output_rankings": str(gold / "rankings" / "candidate_scores.csv"),
            "output_explanations": str(gold / "rankings" / "candidate_scores_explained.csv"),
            "evaluation_results_csv": str(gold / "evaluation" / "evaluation_results.csv"),
            "model_comparison_csv": str(gold / "evaluation" / "model_comparison.csv"),
            "ground_truth": str(root / "data" / "evaluation" / "ground_truth.csv"),
        },
        "skills": {"path": str(Path(__file__).resolve().parents[1] / "config" / "skills.yaml")},
        "privacy": {"anonymize": True},
        "preprocessing": {"language": "en", "remove_stopwords": True, "lemmatize": True},
        "tfidf": {"max_features": 500, "ngram_range": [1, 2], "min_df": 1, "max_df": 0.99},
        "embeddings": {"enabled": False},
        "bm25": {"enabled": False},
        "fusion": {"weights": {"tfidf": 0.5, "dense": 0.0, "skills": 0.3, "experience": 0.2}},
        "matching": {"top_k": 3},
        "evaluation": {"top_k_values": [1, 3]},
        "pipeline": {"write_explanations": True},
        "experiment": {"write_manifest": False},
        "logging": {"level": "WARNING"},
        "silver": {"write_unified_resumes": False},
    }


def _write_minimal_processed_tables(root: Path) -> None:
    silver = root / "data" / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    cvs = pd.DataFrame(
        {
            "cv_id": ["vanetik_cv_001", "vanetik_cv_002", "vanetik_cv_003"],
            "text": [
                "Python backend engineer with APIs and SQL experience." * 2,
                "Data analyst with SQL, pandas and dashboard work." * 2,
                "Junior software engineer with python and docker basics." * 2,
            ],
        }
    )
    jobs = pd.DataFrame(
        {
            "job_id": ["vanetik_vacancy_001", "vanetik_vacancy_002"],
            "text": [
                "Backend role requiring Python, APIs, SQL, and production work." * 2,
                "Analyst role requiring SQL, pandas, and reporting." * 2,
            ],
        }
    )
    cvs.to_csv(silver / "cleaned_cvs.csv", index=False)
    jobs.to_csv(silver / "cleaned_jobs.csv", index=False)


def _write_ground_truth(root: Path) -> None:
    ev = root / "data" / "evaluation"
    ev.mkdir(parents=True, exist_ok=True)
    gt = pd.DataFrame(
        {
            "job_id": ["vanetik_vacancy_001", "vanetik_vacancy_001", "vanetik_vacancy_002"],
            "resume_id": ["vanetik_cv_001", "corpus_vanetik_cv_003", "vanetik_cv_002"],
            "relevance": [3, 1, 2],
        }
    )
    gt.to_csv(ev / "ground_truth.csv", index=False)


def test_train_and_predict_learned_fusion() -> None:
    scores_df = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j2", "j2"],
            "cv_id": ["c1", "c2", "c1", "c2"],
            "tfidf_score": [0.9, 0.2, 0.1, 0.8],
            "semantic_score": [0.8, 0.1, 0.2, 0.7],
            "bm25_score": [0.7, 0.2, 0.2, 0.6],
            "skill_score": [0.9, 0.1, 0.2, 0.8],
            "experience_score": [0.8, 0.2, 0.1, 0.7],
            "must_have_coverage": [1.0, 0.2, 0.3, 0.9],
        }
    )
    gt_df = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j2", "j2"],
            "cv_id": ["c1", "c2", "c1", "c2"],
            "relevance": [3, 0, 1, 2],
        }
    )
    model = train_learned_fusion(scores_df, gt_df, feature_cols=list(DEFAULT_FEATURE_COLS))
    preds = predict_learned_fusion(scores_df, model, feature_cols=list(DEFAULT_FEATURE_COLS))
    assert len(preds) == len(scores_df)
    assert ((preds >= 0.0) & (preds <= 1.0)).all()


def test_pipeline_skips_learned_fusion_when_ground_truth_missing(tmp_path: Path) -> None:
    _write_minimal_processed_tables(tmp_path)
    cfg = _base_cfg(tmp_path)
    # Do not create ground_truth.csv on purpose.
    run_full_pipeline(tmp_path, cfg, ingest=False, semantic=False, evaluate=False, bm25=False)
    out = pd.read_csv(tmp_path / "data" / "gold" / "rankings" / "candidate_scores_explained.csv")
    assert "learned_fusion_score" in out.columns
    assert out["learned_fusion_score"].isna().all()


def test_model_comparison_contains_learned_fusion_row(tmp_path: Path) -> None:
    _write_minimal_processed_tables(tmp_path)
    _write_ground_truth(tmp_path)
    cfg = _base_cfg(tmp_path)
    run_full_pipeline(tmp_path, cfg, ingest=False, semantic=False, evaluate=True, bm25=False)
    comp = pd.read_csv(tmp_path / "data" / "gold" / "evaluation" / "model_comparison.csv")
    assert "Learned Fusion" in set(comp["model"].tolist())


def test_pipeline_writes_learned_fusion_score_column_when_ground_truth_exists(tmp_path: Path) -> None:
    _write_minimal_processed_tables(tmp_path)
    _write_ground_truth(tmp_path)
    cfg = _base_cfg(tmp_path)
    run_full_pipeline(tmp_path, cfg, ingest=False, semantic=False, evaluate=False, bm25=False)
    explained = pd.read_csv(tmp_path / "data" / "gold" / "rankings" / "candidate_scores_explained.csv")
    assert "learned_fusion_score" in explained.columns
    vals = explained["learned_fusion_score"].dropna()
    assert not vals.empty
    assert ((vals >= 0.0) & (vals <= 1.0)).all()

