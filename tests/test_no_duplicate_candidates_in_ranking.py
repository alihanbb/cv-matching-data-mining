from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.orchestrator import run_full_pipeline
from src.utils.id_normalization import normalize_cv_id


def _cfg(root: Path) -> dict:
    silver = root / "data" / "silver"
    gold = root / "data" / "gold"
    return {
        "paths": {
            "processed_cvs": str(silver / "cleaned_cvs.csv"),
            "processed_jobs": str(silver / "cleaned_jobs.csv"),
            "tfidf_model": str(gold / "models" / "tfidf_model.pkl"),
            "output_rankings": str(gold / "rankings" / "candidate_scores.csv"),
            "output_explanations": str(gold / "rankings" / "candidate_scores_explained.csv"),
        },
        "skills": {"path": str(Path(__file__).resolve().parents[1] / "config" / "skills.yaml")},
        "privacy": {"anonymize": True},
        "preprocessing": {"language": "en", "remove_stopwords": True, "lemmatize": True},
        "tfidf": {"max_features": 300, "ngram_range": [1, 2], "min_df": 1, "max_df": 0.99},
        "embeddings": {"enabled": False},
        "bm25": {"enabled": False},
        "fusion": {"weights": {"tfidf": 0.5, "dense": 0.0, "skills": 0.3, "experience": 0.2}},
        "matching": {"top_k": 10},
        "pipeline": {"write_explanations": True},
        "experiment": {"write_manifest": False},
        "logging": {"level": "WARNING"},
        "silver": {"write_unified_resumes": False},
    }


def test_ranking_has_no_duplicate_canonical_cv_ids(tmp_path: Path) -> None:
    silver = tmp_path / "data" / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    cvs = pd.DataFrame(
        {
            "cv_id": [
                "vanetik_cv_014",
                "corpus_vanetik_cv_014",
                "corpus_corpus_vanetik_cv_014",
                "vanetik_cv_999",
            ],
            "text": [
                "Python backend engineer with FastAPI and PostgreSQL experience." * 2,
                "Python backend engineer with FastAPI and PostgreSQL experience." * 2,
                "Python backend engineer with FastAPI and PostgreSQL experience." * 2,
                "Data analyst with SQL and dashboard experience." * 2,
            ],
        }
    )
    jobs = pd.DataFrame(
        {
            "job_id": ["vanetik_vacancy_001"],
            "text": ["Backend role requiring Python, APIs, SQL, and production experience." * 2],
        }
    )
    cvs.to_csv(silver / "cleaned_cvs.csv", index=False)
    jobs.to_csv(silver / "cleaned_jobs.csv", index=False)

    cfg = _cfg(tmp_path)
    run_full_pipeline(tmp_path, cfg, ingest=False, semantic=False, evaluate=False, bm25=False)

    out = pd.read_csv(tmp_path / "data" / "gold" / "rankings" / "candidate_scores_explained.csv")
    out["cv_id_canonical"] = out["cv_id"].map(normalize_cv_id)
    dup_counts = out.groupby(["job_id", "cv_id_canonical"]).size()
    assert not (dup_counts > 1).any(), "Canonical CV IDs must be unique per job ranking."

