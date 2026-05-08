from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.defaults import (
    FUSION_V1_WEIGHTS,
    FUSION_V2_WEIGHTS,
    SCORE_AUDIT_WARN_THRESHOLD,
)
from src.evaluation.metrics import precision_at_k, recall_at_k, top_k_accuracy
from src.evaluation.ranking_metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k
from src.extraction.skill_extractor import extract_skill_ids_sets_for_corpus
from src.extraction.skills_lexicon import load_skills_lexicon
from src.features.tfidf_vectorizer import TfidfFeatureBuilder
from src.ingest.build_processed import build_processed_from_raw
from src.ingest.cv_corpus import extra_cvs_from_ingest_config
from src.models.matcher import enrich_detailed, rank_candidates_for_jobs
from src.pipeline.io import read_processed_csv
from src.pipeline.matching_inputs import build_matching_matrices
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.pii import anonymize_text
from src.processing.cv_sections import cv_quality_score, segment_cv, write_unified_resumes_jsonl
from src.schemas.documents import validate_ground_truth_df, validate_processed_df
from src.scoring.fusion import fuse_scores, fuse_weighted_raw, skill_jaccard_matrix
from src.scoring.score_audit import add_score_audit_columns
from src.utils.experiment import write_run_manifest
from src.utils.helpers import ensure_parent, resolve_path
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maybe_unified_jsonl(
    root: Path,
    cvs: pd.DataFrame,
    lex,
    cleaner: TextCleaner,
    out_path: Path,
    *,
    anonymize: bool,
) -> None:
    rows: list[dict[str, Any]] = []
    for _, r in cvs.iterrows():
        raw_text = str(r["text"])
        work_text = anonymize_text(raw_text) if anonymize else raw_text
        cleaned = cleaner.clean(work_text)
        sections = segment_cv(work_text)
        skill_ids = set()
        for part in sections.values():
            skill_ids |= set(s for s in extract_skill_ids_sets_for_corpus([part], lex)[0])
        skill_ids |= set(extract_skill_ids_sets_for_corpus([work_text], lex)[0])
        cats = lex.categories_for(skill_ids)
        from src.extraction.experience_extractor import cv_total_years_estimate, extract_experience_signals
        sig = extract_experience_signals(work_text)
        years = float(cv_total_years_estimate(sig))
        q = cv_quality_score(sections, work_text)
        rows.append(
            {
                "cv_id": str(r["cv_id"]),
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "sections": sections,
                "extracted_skills": sorted(skill_ids),
                "skill_categories": cats,
                "total_years_experience": years,
                "cv_quality_score": q,
            }
        )
    write_unified_resumes_jsonl(rows, out_path)
    logger.info("Wrote unified resumes JSONL (%d rows): %s", len(rows), out_path)


def _pair_value(mat: np.ndarray, cv_pos: dict, job_pos: dict, row: pd.Series) -> float:
    i = cv_pos[str(row["cv_id"])]
    j = job_pos[str(row["job_id"])]
    return float(mat[i, j])


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def run_full_pipeline(
    root: Path,
    cfg: dict[str, Any],
    *,
    ingest: bool = False,
    semantic: bool = True,
    evaluate: bool | None = None,
    bm25: bool = False,
    write_unified: bool | None = None,
) -> dict[str, float]:
    """Run the full CV–job matching pipeline.

    When ``bm25`` is True or enabled in config, ranking uses Hybrid V2 weights
    including a BM25 channel (requires ``rank_bm25``).
    """
    paths = cfg["paths"]
    proc_cvs = resolve_path(root, paths["processed_cvs"])
    proc_jobs = resolve_path(root, paths["processed_jobs"])
    feat_model = resolve_path(root, paths["tfidf_model"])
    out_rank = resolve_path(root, paths["output_rankings"])
    explain_path = resolve_path(
        root,
        paths.get("output_explanations", "data/gold/rankings/candidate_scores_explained.csv"),
    )

    if write_unified is None:
        write_unified = bool(cfg.get("silver", {}).get("write_unified_resumes", False))
    unified_path = resolve_path(root, cfg.get("silver", {}).get("unified_resumes", "data/silver/unified_resumes.jsonl"))

    # ------------------------------------------------------------------
    # Optional ingest step
    # ------------------------------------------------------------------
    if ingest:
        ing = cfg.get("ingest", {})
        raw_cvs = resolve_path(root, ing.get("raw_cvs_dir", "data/bronze/cvs"))
        raw_jobs = resolve_path(root, ing.get("raw_jobs_dir", "data/bronze/job_descriptions"))
        extra_cv = extra_cvs_from_ingest_config(root, ing)
        n_bronze, n_corpus, n_job = build_processed_from_raw(
            raw_cvs, raw_jobs, proc_cvs, proc_jobs, extra_cv_rows=extra_cv or None
        )
        logger.info(
            "Ingest: %d bronze CV files + %d JSONL corpus rows → %d total CV rows; %d job files → Silver",
            n_bronze, n_corpus, n_bronze + n_corpus, n_job,
        )

    # ------------------------------------------------------------------
    # Build all feature matrices via the single shared builder
    # ------------------------------------------------------------------
    bm25_cfg = cfg.get("bm25", {})
    bm25_enabled_cfg = bool(bm25 or bm25_cfg.get("enabled", False))
    m = build_matching_matrices(root, cfg, semantic=semantic, bm25=bm25_enabled_cfg)

    # Re-save TF-IDF model (build_matching_matrices does not persist it)
    pre = cfg.get("preprocessing", {})
    cleaner = TextCleaner(
        remove_stopwords=pre.get("remove_stopwords", True),
        lemmatize=pre.get("lemmatize", True),
        language=pre.get("language", "en"),
    )
    tfidf_cfg = cfg.get("tfidf", {})
    tfidf_builder = TfidfFeatureBuilder(tfidf_cfg)
    corpus = m.cvs_df["clean_text"].tolist() + m.jobs_df["clean_text"].tolist()
    tfidf_builder.fit(corpus)
    ensure_parent(feat_model)
    tfidf_builder.save(feat_model)

    # ------------------------------------------------------------------
    # Optional unified JSONL output
    # ------------------------------------------------------------------
    if write_unified:
        skill_cfg = cfg.get("skills", {})
        lex_path = resolve_path(root, skill_cfg.get("path", "config/skills.yaml"))
        lex = load_skills_lexicon(lex_path)
        privacy = cfg.get("privacy", {})
        anonymize = bool(privacy.get("anonymize", True))
        # Load raw CVs from silver CSV (before clean_text) for JSONL output
        raw_cvs_df = validate_processed_df(read_processed_csv(proc_cvs, "cv_id"), "cv_id", "text")
        _maybe_unified_jsonl(root, raw_cvs_df, lex, cleaner, unified_path, anonymize=anonymize)

    # ------------------------------------------------------------------
    # Fusion configuration
    # ------------------------------------------------------------------
    fusion_cfg = cfg.get("fusion", {})
    weights = dict(fusion_cfg.get("weights", {})) or dict(FUSION_V1_WEIGHTS)

    fusion_v2_cfg = cfg.get("fusion_v2", {})
    weights_v2 = dict(fusion_v2_cfg.get("weights", {})) or dict(FUSION_V2_WEIGHTS)

    fused_rank_v1, w_used = fuse_scores(
        m.sim_lex, m.dense_sim, m.skill_score, m.exp_mat, weights, m.dense_enabled, bm25=None
    )

    if m.bm25_enabled and m.bm25 is not None:
        fused_rank_used, w_rank = fuse_scores(
            m.sim_lex, m.dense_sim, m.skill_score, m.exp_mat, weights_v2, m.dense_enabled, bm25=m.bm25
        )
    else:
        fused_rank_used, w_rank = fused_rank_v1, w_used

    logger.info("Fusion weights for ranking (normalized channels): %s", w_rank)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    top_k = int(cfg.get("matching", {}).get("top_k", 10))
    dense_for_components = m.dense_sim if m.dense_sim is not None else np.zeros_like(m.sim_lex)
    bm25_for_components = m.bm25 if m.bm25 is not None else np.zeros_like(m.sim_lex)
    components = {
        "tfidf": m.sim_lex,
        "dense": dense_for_components,
        "skills": m.skill_score,
        "experience": m.exp_mat,
        "bm25": bm25_for_components,
        "skill_jaccard": m.jaccard_mat,
    }

    ranked = rank_candidates_for_jobs(
        fused_rank_used, m.cv_ids, m.job_ids, top_k, components, score_column="ranking_score",
    )
    ranked_det = enrich_detailed(
        ranked, m.cv_ids, m.job_ids, m.cv_skill_sets, m.job_reqs,
        cv_years=m.cv_years_list, job_required_years=m.job_req_years,
        must_cov=m.must_cov, nice_cov=m.nice_cov,
        semantic_mat=dense_for_components,
        lex=load_skills_lexicon(resolve_path(root, cfg.get("skills", {}).get("path", "config/skills.yaml"))),
    )

    # Index lookups for matrix access
    cv_pos = {str(cid): idx for idx, cid in enumerate(m.cv_ids)}
    job_pos = {str(jid): idx for idx, jid in enumerate(m.job_ids)}

    renamed = ranked_det.rename(columns={
        "score_tfidf": "tfidf_score",
        "score_dense": "semantic_score",
        "score_skills": "skill_score",
        "score_experience": "experience_score",
        "score_bm25": "bm25_score",
        "score_skill_jaccard": "skill_jaccard",
    })
    renamed["final_score_normalized"] = renamed.apply(
        lambda r: _pair_value(fused_rank_v1, cv_pos, job_pos, r), axis=1
    )

    if m.bm25_enabled and m.bm25 is not None:
        fused_raw_v2, _ = fuse_weighted_raw(
            m.sim_lex, m.dense_sim, m.skill_score, m.exp_mat, weights_v2, m.dense_enabled, bm25=m.bm25
        )
        renamed["final_score_v2_bm25"] = renamed.apply(
            lambda r: _pair_value(fused_raw_v2, cv_pos, job_pos, r), axis=1
        )
    else:
        renamed["final_score_v2_bm25"] = np.nan

    renamed = add_score_audit_columns(
        renamed, weights,
        has_semantic=m.dense_enabled, has_bm25=m.bm25_enabled,
        weights_v2=weights_v2 if m.bm25_enabled else None,
    )
    renamed["final_score"] = renamed["final_score_raw"]
    renamed["score_diff"] = (renamed["final_score"].astype(float) - renamed["score_check"].astype(float)).abs()
    renamed["score_warning"] = renamed["score_diff"].apply(
        lambda x: "CHECK>0.01" if float(x) > SCORE_AUDIT_WARN_THRESHOLD else ""
    )

    # ------------------------------------------------------------------
    # Optional learned fusion score
    # ------------------------------------------------------------------
    lf_path = resolve_path(root, "artifacts/learned_fusion_weights.json")
    if lf_path.is_file():
        with open(lf_path, encoding="utf-8") as f:
            lw = json.load(f)
        wv = np.array(
            [float(lw.get(k, 0)) for k in ("tfidf", "semantic", "bm25", "skills", "experience")],
            dtype=np.float64,
        )

        def _lf_row(r: pd.Series) -> float:
            v = np.array(
                [float(r.get(k, 0)) for k in ("tfidf_score", "semantic_score", "bm25_score", "skill_score", "experience_score")],
                dtype=np.float64,
            )
            return float((v * wv).sum())

        renamed["learned_fusion_score"] = renamed.apply(_lf_row, axis=1)
    else:
        renamed["learned_fusion_score"] = np.nan

    # Matched / missing skills summary
    ms: list[str] = []
    mis: list[str] = []
    for _, r in renamed.iterrows():
        i = cv_pos[str(r["cv_id"])]
        j = job_pos[str(r["job_id"])]
        cv_s = m.cv_skill_sets[i]
        job_s = m.job_skill_sets[j]
        ms.append(";".join(sorted(cv_s & job_s)))
        mis.append(";".join(sorted(job_s - cv_s)))
    renamed["matched_skills"] = ms
    renamed["missing_skills"] = mis

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    ensure_parent(out_rank)
    ranked_simple = renamed.copy()
    ranked_simple["score"] = ranked_simple["ranking_score"]
    ranked_simple = ranked_simple.drop(
        columns=["matched_skills", "missing_skills", "experience_note", "explanation", "suggested_improvements", "ranking_score"],
        errors="ignore",
    )
    if "final_score" not in ranked_simple.columns:
        ranked_simple["final_score"] = renamed["final_score"]
    ranked_simple.to_csv(out_rank, index=False)

    if cfg.get("pipeline", {}).get("write_explanations", True):
        ensure_parent(explain_path)
        preferred = [
            "job_id", "cv_id", "rank_for_job", "ranking_score",
            "tfidf_score", "semantic_score", "bm25_score", "skill_jaccard", "skill_score", "experience_score",
            "must_have_coverage", "nice_to_have_coverage",
            "matched_required_skills", "missing_critical_skills", "matched_optional_skills", "missing_optional_skills",
            "cv_years_experience", "job_min_years_experience",
            "final_score_raw", "final_score_normalized", "final_score", "final_score_v2_bm25", "learned_fusion_score",
            "score_check", "score_diff", "score_warning",
            "matched_skills", "missing_skills", "explanation", "suggested_improvements",
        ]
        ordered = [c for c in preferred if c in renamed.columns]
        explained = renamed[ordered + [c for c in renamed.columns if c not in ordered]]
        explained.to_csv(explain_path, index=False)
        logger.info("Wrote explainable rankings: %s", explain_path)

    # ------------------------------------------------------------------
    # Offline evaluation
    # ------------------------------------------------------------------
    metrics: dict[str, float] = {}
    gt_path = paths.get("ground_truth")
    run_eval = bool(evaluate) if evaluate is not None else True
    if gt_path and run_eval:
        gt_file = resolve_path(root, gt_path)
        if gt_file.is_file():
            gt = validate_ground_truth_df(pd.read_csv(gt_file))
            eval_cfg = cfg.get("evaluation", {})
            ks = eval_cfg.get("top_k_values", [1, 3, 5])
            ranked_eval = renamed.copy()
            ranked_eval["score"] = ranked_eval["ranking_score"]
            for k in ks:
                metrics[f"topk_hit_rate_{k}"] = float(top_k_accuracy(ranked_eval, gt, int(k)))
                metrics[f"precision_at_{k}"] = float(precision_at_k(ranked_eval, gt, int(k)))
                metrics[f"recall_at_{k}"] = float(recall_at_k(ranked_eval, gt, int(k)))
                metrics[f"ndcg_at_{k}"] = float(ndcg_at_k(ranked_eval, gt, int(k)))
            metrics["mrr"] = float(mean_reciprocal_rank(ranked_eval, gt))
            metrics["map"] = float(mean_average_precision(ranked_eval, gt))
            for k, v in metrics.items():
                logger.info("%s: %.4f", k, v)
        elif evaluate:
            raise FileNotFoundError(
                f"Ground truth file not found: {gt_file}\n"
                "Provide a ground_truth.csv (job_id,cv_id,relevance) before --evaluate."
            )
        else:
            logger.info("Ground truth missing at %s — skipping metrics.", gt_file)

    # ------------------------------------------------------------------
    # Run manifest
    # ------------------------------------------------------------------
    artifact_paths = {
        "input_cvs": str(proc_cvs),
        "input_jobs": str(proc_jobs),
        "output_rankings": str(out_rank),
        "tfidf_model": str(feat_model),
    }
    if cfg.get("pipeline", {}).get("write_explanations", True):
        artifact_paths["output_explained"] = str(explain_path)

    if cfg.get("experiment", {}).get("write_manifest", True):
        write_run_manifest(
            root, cfg, artifact_paths, metrics,
            notes=json.dumps({"dense_enabled": m.dense_enabled, "bm25_enabled": m.bm25_enabled}),
        )

    logger.info("Wrote rankings: %s", out_rank)
    logger.info("Saved TF-IDF model: %s", feat_model)
    return metrics
