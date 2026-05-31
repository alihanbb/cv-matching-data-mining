"""CV (Candidate) endpoints."""

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CVS_SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_cvs.csv"
UNIFIED_PATH = PROJECT_ROOT / "data" / "silver" / "unified_resumes.jsonl"

# ---------------------------------------------------------------------------
# Disk-cache: CSV dosyası değişmediği sürece yeniden okunmaz.
# ---------------------------------------------------------------------------
_cvs_cache: dict = {"df": None, "mtime": None}


def load_cvs_df() -> pd.DataFrame:
    """Load CVs from silver CSV with mtime-based disk cache."""
    if not CVS_SILVER_PATH.is_file():
        return pd.DataFrame()

    current_mtime = CVS_SILVER_PATH.stat().st_mtime
    if _cvs_cache["mtime"] == current_mtime and _cvs_cache["df"] is not None:
        return _cvs_cache["df"]

    df = pd.read_csv(CVS_SILVER_PATH)
    _cvs_cache["df"] = df
    _cvs_cache["mtime"] = current_mtime
    return df


@router.get("/cvs")
async def list_cvs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    source: Optional[str] = Query(None),
):
    """List all CVs."""
    df = load_cvs_df()
    if df.empty:
        return {"cvs": [], "total": 0, "limit": limit, "offset": offset}

    if source:
        df = df[df["source"] == source]

    cvs = df[["cv_id", "source"]].drop_duplicates().iloc[offset:offset + limit]
    return {
        "cvs": cvs.to_dict(orient="records"),
        "total": df["cv_id"].nunique(),
        "limit": limit,
        "offset": offset,
    }


# IMPORTANT: /cvs/sources MUST be declared before /cvs/{cv_id} so FastAPI
# does not treat the literal string "sources" as a cv_id path parameter.
@router.get("/cvs/sources")
async def get_cv_sources():
    """Get available CV sources."""
    df = load_cvs_df()
    if df.empty:
        return {"sources": []}
    sources = df["source"].dropna().unique().tolist()
    return {"sources": sources}


@router.get("/cvs/{cv_id}")
async def get_cv(cv_id: str):
    """Get CV details by ID."""
    from src.utils.id_normalization import normalize_cv_id

    df = load_cvs_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No CVs found")

    normalized_id = normalize_cv_id(cv_id)
    cv = df[df["cv_id"].map(normalize_cv_id) == normalized_id]

    if cv.empty:
        raise HTTPException(status_code=404, detail=f"CV {cv_id} not found")

    row = cv.iloc[0]
    return {
        "cv_id": row["cv_id"],
        "source": row.get("source", ""),
        "text": row.get("text", "")[:5000],
    }


@router.get("/cvs/{cv_id}/profile")
async def get_cv_profile(cv_id: str):
    """Get extracted profile for a CV."""
    from src.utils.id_normalization import normalize_cv_id

    if not UNIFIED_PATH.is_file():
        return {"cv_id": cv_id, "profile": None, "message": "No unified data"}

    import json
    with open(UNIFIED_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("cv_id") and normalize_cv_id(data["cv_id"]) == normalize_cv_id(cv_id):
                return {
                    "cv_id": cv_id,
                    "extracted_skills": data.get("extracted_skills", []),
                    "skill_categories": data.get("skill_categories", []),
                    "total_years_experience": data.get("total_years_experience", 0),
                    "cv_quality_score": data.get("cv_quality_score", 0),
                }

    return {"cv_id": cv_id, "profile": None}
