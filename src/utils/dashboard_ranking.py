from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils.candidate_dedup import dedupe_candidates_by_canonical_cv_id, resolve_score_column
from src.utils.id_normalization import normalize_cv_id, normalize_job_id

# Keep in sync with ``ingest.ranking_sources`` in config/config.yaml (dashboard + pipeline).
RANKING_SOURCE_TAGS: tuple[str, ...] = (
    "vacancy_resume_matching",
    "sample",
    "huggingface_cv_matcher",
    "dataturks_resume_ner_train",
    "dataturks_resume_ner_test",
    "mehyar_ner_annotated_cv",
    "nlp_ner_on_resume_json_demo",
    "cv_analysis_pdf_corpus",
)

NER_SOURCE_TAGS: tuple[str, ...] = (
    "dataturks_resume_ner_train",
    "dataturks_resume_ner_test",
    "mehyar_ner_annotated_cv",
    "nlp_ner_on_resume_json_demo",
)


def _as_source_set(values: Iterable[str] | None) -> set[str]:
    return {str(v).strip() for v in (values or []) if str(v).strip()}


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

    score_col = resolve_score_column(out, score_column)
    out = dedupe_candidates_by_canonical_cv_id(
        out,
        score_column=score_col,
        keep_canonical_column=False,
    )
    return out.head(int(top_n)).reset_index(drop=True)
