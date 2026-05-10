from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.id_normalization import normalize_cv_id, normalize_job_id


class CleanDocument(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class GroundTruthRow(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    cv_id: str
    job_id: str
    relevant: int

    @field_validator("cv_id", "job_id", mode="before")
    @classmethod
    def _norm_ids(cls, v):
        s = str(v).strip()
        if not s:
            raise ValueError("cv_id/job_id must be non-empty")
        return s

    @field_validator("relevant", mode="before")
    @classmethod
    def _coerce_rel(cls, v):
        return int(v)

    @field_validator("relevant")
    @classmethod
    def _rel_in_range(cls, v: int) -> int:
        # Graded relevance: 0 = not relevant, 1 = weak, 2 = relevant, 3 = highly relevant.
        # Binary labels (0/1) remain valid as a special case.
        if v not in (0, 1, 2, 3):
            raise ValueError("relevant must be one of {0, 1, 2, 3}")
        return v


def validate_processed_df(df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    missing = {id_col, text_col} - set(df.columns)
    if missing:
        raise ValueError(f"Processed table missing columns: {missing}")
    out = df[[id_col, text_col]].dropna()
    for _, row in out.iterrows():
        CleanDocument(doc_id=str(row[id_col]), text=str(row[text_col]))
    return out


def validate_ground_truth_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a ground-truth DataFrame.

    Accepts either ``relevant`` or ``relevance``; ``resume_id`` is accepted as alias
    of ``cv_id``. The returned frame always exposes ``cv_id`` and ``relevant``.
    """
    df = df.copy()
    if "cv_id" not in df.columns and "resume_id" in df.columns:
        df = df.rename(columns={"resume_id": "cv_id"})
    if "relevant" not in df.columns and "relevance" in df.columns:
        df = df.rename(columns={"relevance": "relevant"})
    for c in ("cv_id", "job_id", "relevant"):
        if c not in df.columns:
            raise ValueError(f"Ground truth must have column {c} (or resume_id alias for cv_id)")
    rows: list[dict] = []
    for _, r in df.iterrows():
        gr = GroundTruthRow(
            cv_id=normalize_cv_id(r["cv_id"]),
            job_id=normalize_job_id(r["job_id"]),
            relevant=int(r["relevant"]),
        )
        rows.append(gr.model_dump())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["relevant"] = out["relevant"].astype(int)
    # If aliases/prefixes collapse to the same canonical pair, keep strongest label.
    out = out.groupby(["job_id", "cv_id"], as_index=False, sort=False)["relevant"].max()
    return out
