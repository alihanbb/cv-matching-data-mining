"""Models package for API."""

from pydantic import BaseModel
from typing import Optional, List


class JobBase(BaseModel):
    job_id: str
    title: Optional[str] = None
    source: Optional[str] = None


class Job(JobBase):
    text: Optional[str] = None


class CVBase(BaseModel):
    cv_id: str
    source: Optional[str] = None


class CV(CVBase):
    text: Optional[str] = None


class RankingCandidate(BaseModel):
    rank: int
    cv_id: str
    score: float
    tfidf_score: Optional[float] = 0
    semantic_score: Optional[float] = 0
    bm25_score: Optional[float] = 0
    skill_score: Optional[float] = 0
    matched_skills: Optional[str] = ""
    missing_skills: Optional[str] = ""


class EvaluationMetric(BaseModel):
    model: str
    ndcg_at_5: float
    mrr: float
    precision_at_5: float = 0