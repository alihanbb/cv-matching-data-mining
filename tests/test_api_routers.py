"""API router tests — route ordering, caching, and basic response shapes.

Test edilen kritik davranışlar:
- /api/cvs/sources route'u /api/cvs/{cv_id}'den önce eşleşmeli (P0 Fix 3)
- Disk-cache: ikinci çağrıda CSV yeniden okunmamalı (P2 Fix 7)
- Rankings, Jobs, CVs temel response şekillerini doğrula
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App import — slowapi/etc yoksa skip et
# ---------------------------------------------------------------------------
try:
    from api.main import app

    _app_available = True
except Exception:
    _app_available = False

pytestmark = pytest.mark.skipif(not _app_available, reason="api/main.py import failed (missing deps?)")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Fixture DataFrames
# ---------------------------------------------------------------------------

SAMPLE_CVS = pd.DataFrame(
    {
        "cv_id": ["cv_001", "cv_002"],
        "source": ["sample", "huggingface_cv_matcher"],
        "text": ["Python developer with 5 years exp.", "Java engineer cloud AWS"],
    }
)

SAMPLE_JOBS = pd.DataFrame(
    {
        "job_id": ["job_001"],
        "title": ["Data Scientist"],
        "source": ["sample"],
        "text": ["Looking for a Python ML engineer."],
    }
)

SAMPLE_RANKINGS = pd.DataFrame(
    {
        "job_id": ["job_001", "job_001"],
        "cv_id": ["cv_001", "cv_002"],
        "rank_for_job": [1, 2],
        "final_score": [0.85, 0.72],
        "tfidf_score": [0.70, 0.60],
        "semantic_score": [0.88, 0.75],
        "bm25_score": [0.65, 0.55],
        "skill_score": [0.90, 0.70],
        "must_have_coverage": [0.80, 0.60],
        "nice_to_have_coverage": [0.50, 0.40],
        "matched_required_skills": ["python,sql", "java,aws"],
        "missing_critical_skills": ["docker", "kubernetes"],
        "experience_note": ["5 yrs", "3 yrs"],
        "cv_years_experience": [5.0, 3.0],
        "job_min_years_experience": [3.0, 3.0],
        "explanation": ["Strong match", "Partial match"],
    }
)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body


# ---------------------------------------------------------------------------
# /api/cvs/sources — route ordering fix (P0 Fix 3)
# ---------------------------------------------------------------------------

def test_cvs_sources_not_treated_as_cv_id(client):
    """'sources' literal path must NOT be captured by /cvs/{cv_id}."""
    with patch("api.routers.cv.load_cvs_df", return_value=SAMPLE_CVS):
        resp = client.get("/api/cvs/sources")
    assert resp.status_code == 200
    body = resp.json()
    # Response must have "sources" key — not a cv_id 404 error
    assert "sources" in body, f"Expected 'sources' key, got: {body}"
    assert isinstance(body["sources"], list)


def test_cvs_sources_returns_unique_sources(client):
    with patch("api.routers.cv.load_cvs_df", return_value=SAMPLE_CVS):
        resp = client.get("/api/cvs/sources")
    sources = resp.json()["sources"]
    assert len(sources) == len(set(sources)), "Sources list must not contain duplicates"


# ---------------------------------------------------------------------------
# /api/cvs
# ---------------------------------------------------------------------------

def test_list_cvs(client):
    with patch("api.routers.cv.load_cvs_df", return_value=SAMPLE_CVS):
        resp = client.get("/api/cvs")
    assert resp.status_code == 200
    body = resp.json()
    assert "cvs" in body
    assert "total" in body


def test_list_cvs_source_filter(client):
    with patch("api.routers.cv.load_cvs_df", return_value=SAMPLE_CVS):
        resp = client.get("/api/cvs?source=sample")
    body = resp.json()
    assert body["total"] == 1


def test_get_cv_not_found(client):
    with patch("api.routers.cv.load_cvs_df", return_value=SAMPLE_CVS):
        resp = client.get("/api/cvs/nonexistent_cv")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/jobs
# ---------------------------------------------------------------------------

def test_list_jobs(client):
    with patch("api.routers.job.load_jobs_df", return_value=SAMPLE_JOBS):
        resp = client.get("/api/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert "jobs" in body


def test_get_job_not_found(client):
    with patch("api.routers.job.load_jobs_df", return_value=SAMPLE_JOBS):
        resp = client.get("/api/jobs/nonexistent_job")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/ranking
# ---------------------------------------------------------------------------

def test_get_rankings_all(client):
    with patch("api.routers.ranking.load_rankings_df", return_value=SAMPLE_RANKINGS):
        resp = client.get("/api/ranking")
    assert resp.status_code == 200
    body = resp.json()
    assert "rankings" in body
    assert isinstance(body["rankings"], list)


def test_get_rankings_filter_by_job(client):
    with patch("api.routers.ranking.load_rankings_df", return_value=SAMPLE_RANKINGS):
        resp = client.get("/api/ranking?job_id=job_001")
    body = resp.json()
    assert body["total"] == 2


def test_get_job_ranking(client):
    with patch("api.routers.ranking.load_rankings_df", return_value=SAMPLE_RANKINGS):
        resp = client.get("/api/ranking/job_001")
    assert resp.status_code == 200
    body = resp.json()
    assert "candidates" in body
    assert len(body["candidates"]) <= 10


def test_get_job_ranking_not_found(client):
    with patch("api.routers.ranking.load_rankings_df", return_value=SAMPLE_RANKINGS):
        resp = client.get("/api/ranking/nonexistent_job")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Disk-cache davranışı — mtime değişmezse CSV yeniden okunmamalı (P2 Fix 7)
# ---------------------------------------------------------------------------

def test_cvs_cache_avoids_redundant_read(tmp_path):
    """load_cvs_df: aynı mtime ile iki çağrıda CSV yalnızca bir kez okunmalı."""
    import api.routers.cv as cv_router

    csv_path = tmp_path / "cleaned_cvs.csv"
    SAMPLE_CVS.to_csv(csv_path, index=False)

    # Patch the module-level path and reset cache
    original_path = cv_router.CVS_SILVER_PATH
    cv_router.CVS_SILVER_PATH = csv_path
    cv_router._cvs_cache["df"] = None
    cv_router._cvs_cache["mtime"] = None

    try:
        with patch("pandas.read_csv", wraps=pd.read_csv) as mock_read:
            cv_router.load_cvs_df()  # first call → reads from disk
            cv_router.load_cvs_df()  # second call → should use cache
        assert mock_read.call_count == 1, (
            f"Expected 1 disk read, got {mock_read.call_count}. Cache is not working."
        )
    finally:
        cv_router.CVS_SILVER_PATH = original_path
        cv_router._cvs_cache["df"] = None
        cv_router._cvs_cache["mtime"] = None
