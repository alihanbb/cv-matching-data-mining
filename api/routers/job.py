"""Job posting endpoints."""

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JOBS_SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_jobs.csv"

# ---------------------------------------------------------------------------
# Disk-cache: CSV dosyası değişmediği sürece yeniden okunmaz.
# ---------------------------------------------------------------------------
_jobs_cache: dict = {"df": None, "mtime": None}


def load_jobs_df() -> pd.DataFrame:
    """Load jobs from silver CSV with mtime-based disk cache."""
    if not JOBS_SILVER_PATH.is_file():
        return pd.DataFrame()

    current_mtime = JOBS_SILVER_PATH.stat().st_mtime
    if _jobs_cache["mtime"] == current_mtime and _jobs_cache["df"] is not None:
        return _jobs_cache["df"]

    df = pd.read_csv(JOBS_SILVER_PATH)
    _jobs_cache["df"] = df
    _jobs_cache["mtime"] = current_mtime
    return df


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List all job postings."""
    df = load_jobs_df()
    if df.empty:
        return {"jobs": [], "total": 0, "limit": limit, "offset": offset}

    jobs = df[["job_id", "title", "source"]].drop_duplicates().iloc[offset:offset + limit]
    return {
        "jobs": jobs.to_dict(orient="records"),
        "total": df["job_id"].nunique(),
        "limit": limit,
        "offset": offset,
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job details by ID."""
    from src.utils.id_normalization import normalize_job_id

    df = load_jobs_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No jobs found")

    normalized_id = normalize_job_id(job_id)
    job = df[df["job_id"].map(normalize_job_id) == normalized_id]

    if job.empty:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    row = job.iloc[0]
    return {
        "job_id": row["job_id"],
        "title": row.get("title", ""),
        "source": row.get("source", ""),
        "text": row.get("text", "")[:5000],
    }


@router.get("/jobs/{job_id}/requirements")
async def get_job_requirements(job_id: str):
    """Get extracted requirements for a job."""
    from src.utils.id_normalization import normalize_job_id

    unified_path = PROJECT_ROOT / "data" / "silver" / "unified_resumes.jsonl"
    if not unified_path.is_file():
        return {"job_id": job_id, "requirements": None, "message": "No unified data"}

    import json
    with open(unified_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("job_id") and normalize_job_id(data["job_id"]) == normalize_job_id(job_id):
                return {
                    "job_id": job_id,
                    "required_skills": data.get("required_skills", []),
                    "required_years": data.get("required_years_experience"),
                    "education": data.get("education_requirements", []),
                }

    return {"job_id": job_id, "requirements": None}