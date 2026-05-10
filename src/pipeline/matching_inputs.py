from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.extraction.experience_extractor import (
    cv_max_years,
    extract_experience_signals,
    extract_job_required_years,
)
from src.extraction.requirements_extractor import (
    JobRequirements,
    extract_job_requirements,
    requirement_coverage_matrix,
)
from src.extraction.skill_extractor import extract_skill_ids_sets_for_corpus
from src.extraction.skills_lexicon import SkillsLexicon, load_skills_lexicon
from src.features.bm25_scorer import bm25_matrix
from src.features.semantic_encoder import (
    dense_cosine_similarity,
    encode_normalized,
    try_load_semantic_encoder,
)
from src.features.tfidf_vectorizer import TfidfFeatureBuilder
from src.pipeline.io import read_processed_csv
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.pii import anonymize_text
from src.schemas.documents import validate_processed_df
from src.scoring.fusion import experience_match_matrix, skill_jaccard_matrix
from src.utils.helpers import resolve_path
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MatchingMatrices:
    # ------------------------------------------------------------------
    # Similarity / score matrices — shape (n_cv, n_job)
    # ------------------------------------------------------------------
    sim_lex: np.ndarray
    dense_sim: np.ndarray | None
    bm25: np.ndarray | None
    skill_score: np.ndarray
    exp_mat: np.ndarray

    # Extended coverage matrices (added to avoid re-computation in orchestrator)
    must_cov: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    nice_cov: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    jaccard_mat: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # ------------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------------
    cv_ids: list[str] = field(default_factory=list)
    job_ids: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Skill / experience data (added to avoid re-computation in orchestrator)
    # ------------------------------------------------------------------
    cv_skill_sets: list[set[str]] = field(default_factory=list)
    job_reqs: list[JobRequirements] = field(default_factory=list)
    job_skill_sets: list[set[str]] = field(default_factory=list)
    cv_years_list: list[float] = field(default_factory=list)
    job_req_years: list[float | None] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Processed DataFrames (for text access in enrichment steps)
    # ------------------------------------------------------------------
    cvs_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    jobs_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------
    dense_enabled: bool = False
    bm25_enabled: bool = False


def build_matching_matrices(
    root: Path,
    cfg: dict[str, Any],
    *,
    semantic: bool = True,
    bm25: bool = False,
) -> MatchingMatrices:
    """Load data, build all feature/similarity matrices and return as one object.

    This is the single source of truth for the feature-building step.
    ``orchestrator.py``, ``weight_optimizer.py``, ``learned_fusion.py`` and
    ``compare_models.py`` all call this function rather than rebuilding
    matrices independently.
    """
    from src.config.defaults import (
        FUSION_V1_WEIGHTS,
    )  # noqa: F401 — triggers defaults load

    paths = cfg["paths"]
    proc_cvs = resolve_path(root, paths["processed_cvs"])
    proc_jobs = resolve_path(root, paths["processed_jobs"])
    skill_cfg = cfg.get("skills", {})
    lex_path = resolve_path(root, skill_cfg.get("path", "config/skills.yaml"))
    lex: SkillsLexicon = load_skills_lexicon(lex_path)
    privacy = cfg.get("privacy", {})
    anonymize = bool(privacy.get("anonymize", True))

    if not anonymize:
        logger.warning(
            "PII anonymization is DISABLED (privacy.anonymize=false). "
            "Ensure this is intentional and compliant with data-protection policies."
        )

    pre = cfg.get("preprocessing", {})
    cleaner = TextCleaner(
        remove_stopwords=pre.get("remove_stopwords", True),
        lemmatize=pre.get("lemmatize", True),
        language=pre.get("language", "en"),
    )

    raw_cvs = read_processed_csv(proc_cvs, "cv_id")
    raw_jobs = read_processed_csv(proc_jobs, "job_id")
    cv_text_col = "raw_text" if "raw_text" in raw_cvs.columns else "text"
    job_text_col = "raw_text" if "raw_text" in raw_jobs.columns else "text"
    cvs = validate_processed_df(raw_cvs, "cv_id", cv_text_col)
    jobs = validate_processed_df(raw_jobs, "job_id", job_text_col)
    if "source" in raw_cvs.columns:
        src_df = raw_cvs[["cv_id", "source"]].drop_duplicates(
            subset=["cv_id"], keep="first"
        )
        cvs = cvs.merge(src_df, on="cv_id", how="left")
        cvs["source"] = cvs["source"].fillna("").astype(str)
    else:
        cvs["source"] = ""
    if "source" in raw_jobs.columns:
        js_df = raw_jobs[["job_id", "source"]].drop_duplicates(
            subset=["job_id"], keep="first"
        )
        jobs = jobs.merge(js_df, on="job_id", how="left")
        jobs["source"] = jobs["source"].fillna("").astype(str)
    else:
        jobs["source"] = ""
    if cv_text_col != "text":
        cvs = cvs.rename(columns={cv_text_col: "text"})
    if job_text_col != "text":
        jobs = jobs.rename(columns={job_text_col: "text"})
    if cvs.empty or jobs.empty:
        raise ValueError("Processed CV or job tables are empty — run ingest first.")

    src_cv_text = cvs["text"].map(
        lambda t: anonymize_text(str(t)) if anonymize else str(t)
    )
    src_job_text = jobs["text"].map(
        lambda t: anonymize_text(str(t)) if anonymize else str(t)
    )
    cvs = cvs.assign(_work_text=src_cv_text)
    jobs = jobs.assign(_work_text=src_job_text)
    cvs["clean_text"] = cvs["_work_text"].map(cleaner.clean)
    jobs["clean_text"] = jobs["_work_text"].map(cleaner.clean)

    cv_ids = [str(x) for x in cvs["cv_id"].tolist()]
    job_ids = [str(x) for x in jobs["job_id"].tolist()]

    # Skill extraction
    cv_skill_sets = extract_skill_ids_sets_for_corpus(cvs["_work_text"].tolist(), lex)
    job_reqs = [
        extract_job_requirements(str(t), lex) for t in jobs["_work_text"].tolist()
    ]
    job_skill_sets = [r.must_have | r.nice_to_have for r in job_reqs]

    # Coverage matrices
    must_cov_m, nice_cov_m, skill_score_m = requirement_coverage_matrix(
        cv_skill_sets, job_reqs
    )
    jaccard_mat = skill_jaccard_matrix(cv_skill_sets, job_skill_sets)

    # Experience
    cv_years_list = [
        float(cv_max_years(extract_experience_signals(str(t))))
        for t in cvs["_work_text"].tolist()
    ]
    job_req_years = [
        extract_job_required_years(str(t)) for t in jobs["_work_text"].tolist()
    ]
    exp_mat = experience_match_matrix(cv_years_list, job_req_years)

    # TF-IDF
    tfidf_cfg = cfg.get("tfidf", {})
    builder = TfidfFeatureBuilder(tfidf_cfg)
    corpus = cvs["clean_text"].tolist() + jobs["clean_text"].tolist()
    builder.fit(corpus)
    X_cv = builder.transform(cvs["clean_text"].tolist())
    X_job = builder.transform(jobs["clean_text"].tolist())
    sim_lex = np.clip(cosine_similarity(X_cv, X_job, dense_output=True), 0.0, 1.0)

    # Dense (SBERT)
    emb_cfg = cfg.get("embeddings", {})
    dense_enabled = bool(semantic and emb_cfg.get("enabled", True))
    dense_sim: np.ndarray | None = None
    if dense_enabled:
        from src.config.defaults import (
            DEFAULT_EMBEDDING_BATCH_SIZE,
            DEFAULT_EMBEDDING_MODEL,
        )

        model = try_load_semantic_encoder(
            emb_cfg.get("model_name", DEFAULT_EMBEDDING_MODEL),
            device=emb_cfg.get("device"),
        )
        if model is None:
            dense_enabled = False
            logger.warning("Semantic encoder unavailable — dense channel disabled.")
        else:
            bs = int(emb_cfg.get("batch_size", DEFAULT_EMBEDDING_BATCH_SIZE))
            logger.info(
                "Encoding %d CVs and %d jobs with dense model", len(cvs), len(jobs)
            )
            e_cv = encode_normalized(model, cvs["clean_text"].tolist(), bs)
            e_job = encode_normalized(model, jobs["clean_text"].tolist(), bs)
            dense_sim = np.clip(dense_cosine_similarity(e_cv, e_job), 0.0, 1.0)

    # BM25
    bm25_mat: np.ndarray | None = None
    bm25_cfg = cfg.get("bm25", {})
    bm25_enabled = bool(bm25 or bm25_cfg.get("enabled", False))
    if bm25_enabled:
        try:
            bm25_mat = bm25_matrix(
                jobs["clean_text"].tolist(), cvs["clean_text"].tolist()
            )
        except ImportError as exc:
            logger.warning("BM25 unavailable — channel disabled: %s", exc)
            bm25_enabled = False

    return MatchingMatrices(
        sim_lex=sim_lex,
        dense_sim=dense_sim,
        bm25=bm25_mat,
        skill_score=skill_score_m,
        exp_mat=exp_mat,
        must_cov=must_cov_m,
        nice_cov=nice_cov_m,
        jaccard_mat=jaccard_mat,
        cv_ids=cv_ids,
        job_ids=job_ids,
        cv_skill_sets=cv_skill_sets,
        job_reqs=job_reqs,
        job_skill_sets=job_skill_sets,
        cv_years_list=cv_years_list,
        job_req_years=job_req_years,
        cvs_df=cvs,
        jobs_df=jobs,
        dense_enabled=dense_enabled,
        bm25_enabled=bm25_enabled,
    )


def rankings_from_fused(
    fused: np.ndarray,
    cv_ids: list[str],
    job_ids: list[str],
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for j, jid in enumerate(job_ids):
        col = fused[:, j]
        order = np.argsort(-col)
        for rank, idx in enumerate(order[:top_k], start=1):
            rows.append(
                {
                    "job_id": jid,
                    "cv_id": cv_ids[idx],
                    "score": float(col[idx]),
                    "rank_for_job": rank,
                }
            )
    return pd.DataFrame(rows)
