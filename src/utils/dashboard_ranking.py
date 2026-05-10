from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils.id_normalization import normalize_cv_id, normalize_job_id

RANKING_SOURCE_TAGS: tuple[str, ...] = (
    "vacancy_resume_matching",
    "sample",
    "huggingface_cv_matcher",
)

NER_SOURCE_TAGS: tuple[str, ...] = (
    "dataturks_resume_ner_train",
    "dataturks_resume_ner_test",
    "mehyar_ner_annotated_cv",
    "nlp_ner_on_resume_json_demo",
)


def _as_source_set(values: Iterable[str] | None) -> set[str]:
    return {str(v).strip() for v in (values or []) if str(v).strip()}


def _best_score_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    for col in ("final_score", "final_score_v2_bm25", "learned_fusion_score", "ranking_score", "score"):
        if col in df.columns:
            return col
    return requested


def prepare_candidate_ranking_view(
    rankings: pd.DataFrame,
    *,
    job_id: str,
    score_column: str,
    top_n: int,
    include_ner_sources: bool = False,
    ranking_sources: Iterable[str] | None = RANKING_SOURCE_TAGS,
    ner_sources: Iterable[str] | None = NER_SOURCE_TAGS,
) -> pd.DataFrame:
    """Return a sorted, canonicalized, de-duplicated ranking block for one job."""
    if rankings.empty:
        return rankings.copy()

    out = rankings.copy()
    out["job_id"] = out["job_id"].map(normalize_job_id)
    out["cv_id"] = out["cv_id"].map(normalize_cv_id)
    out = out[(out["job_id"] != "") & (out["cv_id"] != "")]

    target_job = normalize_job_id(job_id)
    out = out[out["job_id"] == target_job]
    if out.empty:
        return out

    if "source" in out.columns:
        out["source"] = out["source"].fillna("").astype(str).str.strip()
        allow = _as_source_set(ranking_sources)
        if allow:
            out = out[out["source"].isin(allow)]
        if not include_ner_sources:
            ner = _as_source_set(ner_sources)
            if ner:
                out = out[~out["source"].isin(ner)]
    if out.empty:
        return out

    score_col = _best_score_column(out, score_column)
    out[score_col] = pd.to_numeric(out.get(score_col), errors="coerce")
    out = out.dropna(subset=[score_col])
    if out.empty:
        return out

    if "rank_for_job" in out.columns:
        out["_rank_for_job_old"] = pd.to_numeric(out["rank_for_job"], errors="coerce")
    else:
        out["_rank_for_job_old"] = float("inf")

    out = out.sort_values(
        by=[score_col, "_rank_for_job_old", "cv_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=["cv_id"], keep="first")
    out["rank_for_job"] = range(1, len(out) + 1)
    out = out.drop(columns=["_rank_for_job_old"], errors="ignore")

    return out.head(int(top_n)).reset_index(drop=True)

