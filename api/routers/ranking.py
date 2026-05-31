"""Ranking endpoints for CV-Job matching."""

from pathlib import Path
from typing import Optional, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RANKINGS_PATH = PROJECT_ROOT / "data" / "gold" / "rankings" / "candidate_scores_explained.csv"

# ---------------------------------------------------------------------------
# Disk-cache: CSV dosyası değişmediği sürece yeniden okunmaz.
# ---------------------------------------------------------------------------
_rankings_cache: dict = {"df": None, "mtime": None}


def load_rankings_df() -> pd.DataFrame:
    """Load rankings from explained CSV with mtime-based disk cache."""
    if not RANKINGS_PATH.is_file():
        return pd.DataFrame()

    current_mtime = RANKINGS_PATH.stat().st_mtime
    if _rankings_cache["mtime"] == current_mtime and _rankings_cache["df"] is not None:
        return _rankings_cache["df"]

    df = pd.read_csv(RANKINGS_PATH)
    from src.utils.id_normalization import normalize_cv_id, normalize_job_id
    if "cv_id" in df.columns:
        df["cv_id"] = df["cv_id"].apply(normalize_cv_id)
    if "job_id" in df.columns:
        df["job_id"] = df["job_id"].apply(normalize_job_id)

    _rankings_cache["df"] = df
    _rankings_cache["mtime"] = current_mtime
    return df


@router.get("/ranking")
async def get_rankings(
    job_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    score_column: str = Query("final_score"),
):
    """Get rankings, optionally filtered by job_id."""
    df = load_rankings_df()
    if df.empty:
        return {"rankings": [], "total": 0}

    if job_id:
        from src.utils.id_normalization import normalize_job_id
        df = df[df["job_id"] == normalize_job_id(job_id)]

    if score_column not in df.columns:
        score_column = "ranking_score"

    df = df.sort_values(score_column, ascending=False).head(limit)

    result = []
    for _, row in df.iterrows():
        result.append({
            "job_id": row["job_id"],
            "cv_id": row["cv_id"],
            "rank": int(row.get("rank_for_job", 0)),
            "score": float(row.get(score_column, 0)),
            "tfidf_score": float(row.get("tfidf_score", 0)),
            "semantic_score": float(row.get("semantic_score", 0)),
            "bm25_score": float(row.get("bm25_score", 0)),
            "skill_score": float(row.get("skill_score", 0)),
            "matched_skills": row.get("matched_required_skills", ""),
            "missing_skills": row.get("missing_critical_skills", ""),
            "experience_note": row.get("experience_note", ""),
        })

    return {
        "rankings": result,
        "total": len(result),
        "job_id": job_id,
        "score_column": score_column,
    }


@router.get("/ranking/jobs")
async def get_ranking_jobs():
    """Get list of jobs that have rankings."""
    df = load_rankings_df()
    if df.empty:
        return {"jobs": []}
    jobs = df["job_id"].unique().tolist()
    return {"jobs": jobs}


@router.get("/ranking/{job_id}")
async def get_job_ranking(
    job_id: str,
    top_n: int = Query(10, ge=1, le=50),
    score_column: str = Query("final_score"),
):
    """Get top N candidates for a specific job."""
    from src.utils.id_normalization import normalize_job_id

    df = load_rankings_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No rankings found")

    normalized_job = normalize_job_id(job_id)
    job_df = df[df["job_id"] == normalized_job]

    if job_df.empty:
        raise HTTPException(status_code=404, detail=f"No rankings for job {job_id}")

    if score_column not in job_df.columns:
        score_column = "ranking_score"

    job_df = job_df.sort_values(score_column, ascending=False).head(top_n)

    candidates = []
    for _, row in job_df.iterrows():
        candidates.append({
            "rank": int(row.get("rank_for_job", 0)),
            "cv_id": row["cv_id"],
            "score": float(row.get(score_column, 0)),
            "tfidf_score": float(row.get("tfidf_score", 0)),
            "semantic_score": float(row.get("semantic_score", 0)),
            "bm25_score": float(row.get("bm25_score", 0)),
            "skill_score": float(row.get("skill_score", 0)),
            "must_have_coverage": float(row.get("must_have_coverage", 0)),
            "nice_to_have_coverage": float(row.get("nice_to_have_coverage", 0)),
            "matched_skills": row.get("matched_required_skills", ""),
            "missing_skills": row.get("missing_critical_skills", ""),
            "cv_years_experience": float(row.get("cv_years_experience", 0)),
            "job_min_years_experience": float(row.get("job_min_years_experience", 0)),
            "explanation": row.get("explanation", ""),
        })

    return {
        "job_id": job_id,
        "total_candidates": len(job_df),
        "top_n": top_n,
        "score_column": score_column,
        "candidates": candidates,
    }


@router.get("/ranking/{job_id}/coverage")
async def get_job_coverage(job_id: str):
    """Get requirement coverage analysis for a job."""
    from src.utils.id_normalization import normalize_job_id

    df = load_rankings_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No rankings found")

    normalized_job = normalize_job_id(job_id)
    job_df = df[df["job_id"] == normalized_job]

    if job_df.empty:
        raise HTTPException(status_code=404, detail=f"No rankings for job {job_id}")

    coverage_data = []
    for _, row in job_df.iterrows():
        coverage_data.append({
            "cv_id": row["cv_id"],
            "rank": int(row.get("rank_for_job", 0)),
            "must_have_coverage": float(row.get("must_have_coverage", 0)),
            "nice_to_have_coverage": float(row.get("nice_to_have_coverage", 0)),
            "matched_required_skills": row.get("matched_required_skills", ""),
            "missing_critical_skills": row.get("missing_critical_skills", ""),
        })

    return {
        "job_id": job_id,
        "coverage": coverage_data,
    }