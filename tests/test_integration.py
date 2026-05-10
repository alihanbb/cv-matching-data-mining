"""End-to-end integration test for the CV-matching pipeline.

Creates a minimal temporary bronze corpus (3 CV files + 2 job files),
runs the full pipeline (TF-IDF only — no SBERT, no BM25 to keep it fast
and dependency-free), and asserts output shape and value invariants.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingest.build_processed import build_processed_from_raw
from src.pipeline.orchestrator import run_full_pipeline

# ---------------------------------------------------------------------------
# Toy corpus fixtures
# ---------------------------------------------------------------------------

_CV_TEXTS = [
    """\
Alice Smith  |  alice@example.com
Experience
2019-2023: Software Engineer at Acme Corp
  - Developed REST APIs using Python and FastAPI
  - Managed PostgreSQL databases, wrote SQL queries
  - Led a team of 4 engineers for 2 years
Skills: Python, FastAPI, PostgreSQL, SQL, Docker, Git
Education: BSc Computer Science, 2018
""",
    """\
Bob Jones
Work History
2020-2024: Data Analyst, Beta Inc
  - Built ETL pipelines in Python and pandas
  - Created dashboards with Tableau
  - Maintained MySQL databases
Skills: Python, pandas, SQL, MySQL, Tableau, Excel
Education: BSc Statistics, 2019
""",
    """\
Carol Yılmaz
2021-2024: Junior Developer, Gamma Ltd
  - Wrote Python scripts for data processing
  - Basic knowledge of PostgreSQL and MongoDB
Skills: Python, JavaScript, HTML, CSS, Git
Education: BSc Information Systems, 2021
""",
]

_JOB_TEXTS = [
    """\
Senior Backend Developer
Requirements:
- 3+ years of Python experience (must-have)
- FastAPI or Django framework (must-have)
- PostgreSQL or MySQL database management (must-have)
- Docker containerisation knowledge (nice-to-have)
- Excellent communication skills
""",
    """\
Junior Data Analyst
Requirements:
- Python and pandas (must-have)
- SQL knowledge (must-have)
- Tableau or Power BI (nice-to-have)
- Excel proficiency (nice-to-have)
- 1+ year experience
""",
]


# ---------------------------------------------------------------------------
# Minimal config dict (no SBERT, no BM25 — pure TF-IDF run)
# ---------------------------------------------------------------------------


def _make_cfg(root: Path) -> dict:
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
        "preprocessing": {
            "language": "en",
            "remove_stopwords": True,
            "lemmatize": True,
        },
        "tfidf": {
            "max_features": 500,
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 0.99,
            "sublinear_tf": True,
        },
        "embeddings": {"enabled": False},
        "bm25": {"enabled": False},
        "fusion": {"weights": {"tfidf": 0.5, "dense": 0.0, "skills": 0.3, "experience": 0.2}},
        "fusion_v2": {
            "weights": {
                "tfidf": 0.4,
                "dense": 0.0,
                "bm25": 0.2,
                "skills": 0.2,
                "experience": 0.2,
            }
        },
        "matching": {"top_k": 3},
        "evaluation": {"top_k_values": [1, 3]},
        "pipeline": {"write_explanations": True},
        "experiment": {"write_manifest": False},
        "logging": {"level": "WARNING"},
        "silver": {"write_unified_resumes": False},
        "ingest": {
            "raw_cvs_dir": str(root / "bronze_cvs"),
            "raw_jobs_dir": str(root / "bronze_jobs"),
            "cv_corpus_jsonl": {"enabled": False, "path": ""},
        },
    }


# ---------------------------------------------------------------------------
# Fixture: temporary project root with toy bronze files
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    bronze_cvs = tmp_path / "bronze_cvs"
    bronze_jobs = tmp_path / "bronze_jobs"
    bronze_cvs.mkdir(parents=True)
    bronze_jobs.mkdir(parents=True)

    for i, text in enumerate(_CV_TEXTS, start=1):
        (bronze_cvs / f"cv_{i:02d}.txt").write_text(text, encoding="utf-8")

    for i, text in enumerate(_JOB_TEXTS, start=1):
        (bronze_jobs / f"job_{i:02d}.txt").write_text(text, encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestStep:
    """Unit-level: test that ingest produces valid Silver CSVs."""

    def test_build_processed_creates_csvs(self, project_root: Path) -> None:
        cfg = _make_cfg(project_root)
        cv_out = Path(cfg["paths"]["processed_cvs"])
        job_out = Path(cfg["paths"]["processed_jobs"])
        cv_out.parent.mkdir(parents=True, exist_ok=True)
        job_out.parent.mkdir(parents=True, exist_ok=True)

        n_bronze, n_extra, n_jobs = build_processed_from_raw(
            Path(cfg["ingest"]["raw_cvs_dir"]),
            Path(cfg["ingest"]["raw_jobs_dir"]),
            cv_out,
            job_out,
        )

        assert cv_out.is_file(), "cleaned_cvs.csv not created"
        assert job_out.is_file(), "cleaned_jobs.csv not created"
        assert n_bronze == len(_CV_TEXTS), f"Expected {len(_CV_TEXTS)} CV rows, got {n_bronze}"
        assert n_jobs == len(_JOB_TEXTS), f"Expected {len(_JOB_TEXTS)} job rows, got {n_jobs}"

    def test_silver_csv_has_required_columns(self, project_root: Path) -> None:
        cfg = _make_cfg(project_root)
        cv_out = Path(cfg["paths"]["processed_cvs"])
        job_out = Path(cfg["paths"]["processed_jobs"])
        cv_out.parent.mkdir(parents=True, exist_ok=True)
        job_out.parent.mkdir(parents=True, exist_ok=True)

        build_processed_from_raw(
            Path(cfg["ingest"]["raw_cvs_dir"]),
            Path(cfg["ingest"]["raw_jobs_dir"]),
            cv_out,
            job_out,
        )

        cvs = pd.read_csv(cv_out)
        jobs = pd.read_csv(job_out)
        for col in ("cv_id", "raw_text", "text"):
            assert col in cvs.columns
        for col in ("job_id", "raw_text", "text"):
            assert col in jobs.columns
        assert len(cvs) == len(_CV_TEXTS)
        assert len(jobs) == len(_JOB_TEXTS)
        # No empty text cells
        assert cvs["text"].notna().all()
        assert jobs["text"].notna().all()


class TestFullPipeline:
    """Integration: ingest → matching → rankings output."""

    @pytest.fixture()
    def ingested_root(self, project_root: Path) -> Path:
        """Run ingest step once; return root ready for pipeline."""
        cfg = _make_cfg(project_root)
        cv_out = Path(cfg["paths"]["processed_cvs"])
        job_out = Path(cfg["paths"]["processed_jobs"])
        cv_out.parent.mkdir(parents=True, exist_ok=True)
        job_out.parent.mkdir(parents=True, exist_ok=True)
        build_processed_from_raw(
            Path(cfg["ingest"]["raw_cvs_dir"]),
            Path(cfg["ingest"]["raw_jobs_dir"]),
            cv_out,
            job_out,
        )
        return project_root

    def test_pipeline_creates_rankings_csv(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False)
        out = Path(cfg["paths"]["output_rankings"])
        assert out.is_file(), f"Rankings CSV not created at {out}"

    def test_rankings_has_correct_columns(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False)
        df = pd.read_csv(cfg["paths"]["output_rankings"])
        for col in ("job_id", "cv_id", "rank_for_job"):
            assert col in df.columns, f"Missing column '{col}'"

    def test_scores_in_valid_range(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False)
        df = pd.read_csv(cfg["paths"]["output_rankings"])
        score_cols = [
            c for c in df.columns if "score" in c.lower() and df[c].dtype in (float, np.float64)
        ]
        for col in score_cols:
            vals = df[col].dropna()
            assert (vals >= -1e-9).all() and (
                vals <= 1.0 + 1e-9
            ).all(), f"Column '{col}' has values outside [0, 1]: min={vals.min():.4f} max={vals.max():.4f}"

    def test_at_least_one_ranking_per_job(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False)
        df = pd.read_csv(cfg["paths"]["output_rankings"])
        for job_id in df["job_id"].unique():
            rows = df[df["job_id"] == job_id]
            assert len(rows) >= 1, f"No rankings found for job_id={job_id}"

    def test_top_k_respected(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        top_k = int(cfg["matching"]["top_k"])
        run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False)
        df = pd.read_csv(cfg["paths"]["output_rankings"])
        for job_id, grp in df.groupby("job_id"):
            assert len(grp) <= top_k, f"job_id={job_id} has {len(grp)} rows, exceeds top_k={top_k}"

    def test_ranks_are_contiguous_from_one(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False)
        df = pd.read_csv(cfg["paths"]["output_rankings"])
        for job_id, grp in df.groupby("job_id"):
            ranks = sorted(grp["rank_for_job"].tolist())
            assert ranks[0] == 1, f"job_id={job_id}: first rank is {ranks[0]}, expected 1"
            assert ranks == list(
                range(1, len(ranks) + 1)
            ), f"job_id={job_id}: ranks not contiguous: {ranks}"

    def test_explained_csv_written(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False)
        explained = Path(cfg["paths"]["output_explanations"])
        assert explained.is_file(), "Explained rankings CSV not written"
        df = pd.read_csv(explained)
        assert not df.empty
        for col in ("score_check", "score_diff", "score_warning"):
            assert col in df.columns, f"Missing audit column '{col}' in explained CSV"

    def test_no_duplicate_cv_per_job(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False)
        df = pd.read_csv(cfg["paths"]["output_rankings"])
        for job_id, grp in df.groupby("job_id"):
            assert grp["cv_id"].nunique() == len(
                grp
            ), f"job_id={job_id}: duplicate CV IDs in rankings"

    def test_pipeline_returns_empty_metrics_without_ground_truth(self, ingested_root: Path) -> None:
        cfg = _make_cfg(ingested_root)
        metrics = run_full_pipeline(ingested_root, cfg, semantic=False, bm25=False, evaluate=False)
        assert isinstance(metrics, dict)


class TestConfigSchema:
    """Unit: PipelineConfig Pydantic validation."""

    def _base_cfg(self) -> dict:
        return {
            "paths": {
                "processed_cvs": "data/silver/cleaned_cvs.csv",
                "processed_jobs": "data/silver/cleaned_jobs.csv",
                "tfidf_model": "data/gold/models/tfidf_model.pkl",
                "output_rankings": "data/gold/rankings/candidate_scores.csv",
            }
        }

    def test_valid_minimal_config(self) -> None:
        from src.config.schema import PipelineConfig

        cfg = PipelineConfig.model_validate(self._base_cfg())
        assert cfg.matching.top_k == 10
        assert cfg.privacy.anonymize is True

    def test_invalid_top_k_raises(self) -> None:
        from pydantic import ValidationError
        from src.config.schema import PipelineConfig

        bad = {**self._base_cfg(), "matching": {"top_k": 0}}
        with pytest.raises(ValidationError, match="top_k"):
            PipelineConfig.model_validate(bad)

    def test_invalid_logging_level_raises(self) -> None:
        from pydantic import ValidationError
        from src.config.schema import PipelineConfig

        bad = {**self._base_cfg(), "logging": {"level": "VERBOSE"}}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(bad)

    def test_invalid_ngram_range_raises(self) -> None:
        from pydantic import ValidationError
        from src.config.schema import PipelineConfig

        bad = {**self._base_cfg(), "tfidf": {"ngram_range": [2, 1]}}
        with pytest.raises(ValidationError, match="ngram_range"):
            PipelineConfig.model_validate(bad)

    def test_unknown_key_raises(self) -> None:
        from pydantic import ValidationError
        from src.config.schema import PipelineConfig

        bad = {**self._base_cfg(), "nonexistent_section": {"foo": "bar"}}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(bad)

    def test_missing_paths_raises(self) -> None:
        from pydantic import ValidationError
        from src.config.schema import PipelineConfig

        with pytest.raises(ValidationError, match="paths"):
            PipelineConfig.model_validate({})
