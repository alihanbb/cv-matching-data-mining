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
from src.evaluation.compare_models import evaluate_models
from src.extraction.skill_extractor import extract_skill_ids_sets_for_corpus
from src.extraction.skills_lexicon import load_skills_lexicon
from src.features.tfidf_vectorizer import TfidfFeatureBuilder
from src.ingest.build_processed import build_processed_from_raw
from src.ingest.cv_corpus import extra_cvs_from_ingest_config
from src.models.learned_fusion import (
    DEFAULT_FEATURE_COLS,
    export_learned_fusion_weights_json,
    predict_learned_fusion,
    save_learned_fusion_model,
    train_learned_fusion,
)
from src.models.matcher import enrich_detailed, rank_candidates_for_jobs
from src.pipeline.io import read_processed_csv
from src.pipeline.matching_inputs import build_matching_matrices
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.pii import anonymize_text
from src.processing.cv_sections import (
    cv_quality_score,
    segment_cv,
    write_unified_resumes_jsonl,
)
from src.schemas.documents import validate_ground_truth_df, validate_processed_df
from src.scoring.fusion import fuse_scores, fuse_weighted_raw
from src.scoring.score_audit import weighted_fusion_v1_row
from src.silver.build import read_cv_quality_scores
from src.utils.candidate_dedup import (
    dedupe_candidates_by_canonical_cv_id,
    dedupe_candidates_by_job_cv_id,
)
from src.utils.experiment import write_run_manifest
from src.utils.helpers import ensure_parent, resolve_path
from src.utils.id_normalization import normalize_cv_id, normalize_job_id
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
        from src.extraction.experience_extractor import (
            cv_total_years_estimate,
            extract_experience_signals,
        )

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
    top_candidates_path = resolve_path(
        root,
        paths.get("top_candidates_csv", "data/gold/rankings/top_candidates_by_job.csv"),
    )
    eval_results_csv = resolve_path(
        root,
        paths.get("evaluation_results_csv", "data/gold/evaluation/evaluation_results.csv"),
    )
    model_comparison_csv = resolve_path(
        root,
        paths.get("model_comparison_csv", "data/gold/evaluation/model_comparison.csv"),
    )
    score_audit_report_csv = resolve_path(
        root,
        paths.get("score_audit_report_csv", "data/gold/evaluation/score_audit_report.csv"),
    )

    resume_prof_path = resolve_path(
        root,
        cfg.get("silver", {}).get("resume_profiles", "data/silver/resume_profiles.jsonl"),
    )

    if write_unified is None:
        write_unified = bool(cfg.get("silver", {}).get("write_unified_resumes", False))
    unified_path = resolve_path(
        root,
        cfg.get("silver", {}).get("unified_resumes", "data/silver/unified_resumes.jsonl"),
    )

    # ------------------------------------------------------------------
    # Optional ingest step
    # ------------------------------------------------------------------
    if ingest:
        ing = cfg.get("ingest", {})
        raw_cvs = resolve_path(root, ing.get("raw_cvs_dir", "data/bronze/cvs"))
        raw_jobs = resolve_path(root, ing.get("raw_jobs_dir", "data/bronze/job_descriptions"))
        extra_cv = extra_cvs_from_ingest_config(root, ing)
        pre = cfg.get("preprocessing", {})
        n_bronze, n_corpus, n_job = build_processed_from_raw(
            raw_cvs,
            raw_jobs,
            proc_cvs,
            proc_jobs,
            root=root,
            ingest_cfg=ing,
            preprocessor_cfg=pre,
            extra_cv_rows=extra_cv or None,
            pipeline_cfg=cfg,
        )
        logger.info(
            "Ingest: %d bronze CV files + %d JSONL corpus rows -> %d total CV rows; "
            "%d job files -> Silver",
            n_bronze,
            n_corpus,
            n_bronze + n_corpus,
            n_job,
        )

    # ------------------------------------------------------------------
    # Build all feature matrices via the single shared builder
    # ------------------------------------------------------------------
    bm25_cfg = cfg.get("bm25", {})
    bm25_enabled_cfg = bool(bm25 or bm25_cfg.get("enabled", False))
    m = build_matching_matrices(root, cfg, semantic=semantic, bm25=bm25_enabled_cfg)

    # Persist the already-fitted TF-IDF model from build_matching_matrices.
    # No second fit is needed — re-fitting on the same corpus is redundant and
    # would waste ~50 % of pipeline time for large corpora.
    if m.tfidf_builder is not None:
        ensure_parent(feat_model)
        m.tfidf_builder.save(feat_model)
    else:
        # Fallback: should not happen in normal flow, but guard defensively.
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
        raw_full = read_processed_csv(proc_cvs, "cv_id")
        uc = "raw_text" if "raw_text" in raw_full.columns else "text"
        raw_cvs_df = validate_processed_df(raw_full, "cv_id", uc)
        _maybe_unified_jsonl(root, raw_cvs_df, lex, cleaner, unified_path, anonymize=anonymize)

    # ------------------------------------------------------------------
    # Fusion configuration
    # ------------------------------------------------------------------
    fusion_cfg = cfg.get("fusion", {})
    weights = dict(fusion_cfg.get("weights", {})) or dict(FUSION_V1_WEIGHTS)

    fusion_v2_cfg = cfg.get("fusion_v2", {})
    weights_v2 = dict(fusion_v2_cfg.get("weights", {})) or dict(FUSION_V2_WEIGHTS)

    fused_rank_v1, w_used = fuse_scores(
        m.sim_lex,
        m.dense_sim,
        m.skill_score,
        m.exp_mat,
        weights,
        m.dense_enabled,
        bm25=None,
    )

    if m.bm25_enabled and m.bm25 is not None:
        fused_rank_used, w_rank = fuse_scores(
            m.sim_lex,
            m.dense_sim,
            m.skill_score,
            m.exp_mat,
            weights_v2,
            m.dense_enabled,
            bm25=m.bm25,
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
        fused_rank_used,
        m.cv_ids,
        m.job_ids,
        top_k,
        components,
        score_column="ranking_score",
    )
    ranked_det = enrich_detailed(
        ranked,
        m.cv_ids,
        m.job_ids,
        m.cv_skill_sets,
        m.job_reqs,
        cv_years=m.cv_years_list,
        job_required_years=m.job_req_years,
        must_cov=m.must_cov,
        nice_cov=m.nice_cov,
        semantic_mat=dense_for_components,
        lex=load_skills_lexicon(
            resolve_path(root, cfg.get("skills", {}).get("path", "config/skills.yaml"))
        ),
    )

    # Index lookups for matrix access
    cv_pos = {str(cid): idx for idx, cid in enumerate(m.cv_ids)}
    job_pos = {str(jid): idx for idx, jid in enumerate(m.job_ids)}

    renamed = ranked_det.rename(
        columns={
            "score_tfidf": "tfidf_score",
            "score_dense": "semantic_score",
            "score_skills": "skill_score",
            "score_experience": "experience_score",
            "score_bm25": "bm25_score",
            "score_skill_jaccard": "skill_jaccard_score",
        }
    )
    renamed["cv_id"] = renamed["cv_id"].map(normalize_cv_id)
    renamed["job_id"] = renamed["job_id"].map(normalize_job_id)
    renamed = renamed[(renamed["cv_id"] != "") & (renamed["job_id"] != "")]
    renamed["fusion_minmax_normalized_v1"] = renamed.apply(
        lambda r: _pair_value(fused_rank_v1, cv_pos, job_pos, r), axis=1
    )

    fused_raw_v1, _ = fuse_weighted_raw(
        m.sim_lex,
        m.dense_sim,
        m.skill_score,
        m.exp_mat,
        weights,
        m.dense_enabled,
        bm25=None,
    )
    renamed["final_score_v1"] = renamed.apply(
        lambda r: _pair_value(fused_raw_v1, cv_pos, job_pos, r), axis=1
    )
    renamed["final_score_raw"] = renamed["final_score_v1"]
    row_chk = renamed.apply(
        lambda r: weighted_fusion_v1_row(r, weights, has_semantic=m.dense_enabled),
        axis=1,
    ).astype(float)
    renamed["score_check"] = row_chk
    renamed["final_score"] = renamed["final_score_v1"]
    raw_score_diff = (
        renamed["final_score_v1"].astype(float) - renamed["score_check"].astype(float)
    ).abs()
    renamed["score_diff"] = raw_score_diff.where(raw_score_diff > 1e-9, 0.0)
    renamed["score_warning"] = renamed["score_diff"].apply(
        lambda x: "CHECK>0.01" if float(x) > SCORE_AUDIT_WARN_THRESHOLD else ""
    )

    if m.bm25_enabled and m.bm25 is not None:
        fused_raw_v2, _ = fuse_weighted_raw(
            m.sim_lex,
            m.dense_sim,
            m.skill_score,
            m.exp_mat,
            weights_v2,
            m.dense_enabled,
            bm25=m.bm25,
        )
        renamed["final_score_v2_bm25"] = renamed.apply(
            lambda r: _pair_value(fused_raw_v2, cv_pos, job_pos, r), axis=1
        )
    else:
        renamed["final_score_v2_bm25"] = np.nan

    qmap = read_cv_quality_scores(resume_prof_path)
    renamed["cv_quality_score"] = renamed["cv_id"].map(lambda x: float(qmap.get(str(x), 0.0)))

    src_series = m.cvs_df.set_index("cv_id")["source"] if "source" in m.cvs_df.columns else None
    if src_series is not None:
        renamed["source"] = renamed["cv_id"].map(lambda x: str(src_series.get(x, "") or ""))
    else:
        renamed["source"] = ""

    # Gold-level dedup: keep best row per (job_id, canonical_cv_id) by main ranking score.
    renamed = dedupe_candidates_by_canonical_cv_id(
        renamed,
        score_column="ranking_score",
        keep_canonical_column=False,
    )
    # Final guard for Gold outputs: enforce one row per (job_id, cv_id).
    renamed = dedupe_candidates_by_job_cv_id(
        renamed,
        score_column="ranking_score",
    )

    # ------------------------------------------------------------------
    # Learned fusion score (additional model; V1/V2 unchanged)
    # ------------------------------------------------------------------
    gt_rel = paths.get("ground_truth", "data/evaluation/ground_truth.csv")
    gt_file = resolve_path(root, gt_rel)
    lf_model_path = resolve_path(root, "artifacts/models/learned_fusion.pt")
    lf_weights_path = resolve_path(root, "artifacts/models/learned_fusion_weights.json")
    if gt_file.is_file():
        try:
            gt_df = validate_ground_truth_df(pd.read_csv(gt_file))
            if gt_df.empty:
                raise ValueError("Ground truth is empty after validation.")
            lf_model = train_learned_fusion(
                renamed,
                gt_df,
                feature_cols=list(DEFAULT_FEATURE_COLS),
                target_col="relevant",
                epochs=100,
                lr=0.01,
            )
            save_learned_fusion_model(lf_model, lf_model_path)
            export_learned_fusion_weights_json(lf_model, list(DEFAULT_FEATURE_COLS), lf_weights_path)
            renamed["learned_fusion_score"] = predict_learned_fusion(
                renamed, lf_model, feature_cols=list(DEFAULT_FEATURE_COLS)
            )
            logger.info("Learned fusion model artifacts saved: %s, %s", lf_model_path, lf_weights_path)
        except (ValueError, ImportError) as exc:
            logger.warning("Learned fusion training skipped: %s", exc)
            renamed["learned_fusion_score"] = np.nan
    else:
        logger.warning("Learned fusion training skipped: %s not found.", gt_rel)
        renamed["learned_fusion_score"] = np.nan

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    ensure_parent(out_rank)
    ranked_simple = renamed.copy()
    ranked_simple["score"] = ranked_simple["ranking_score"]
    ranked_simple = ranked_simple.drop(
        columns=[
            "experience_note",
            "explanation",
            "suggested_improvements",
            "ranking_score",
        ],
        errors="ignore",
    )
    if "final_score" not in ranked_simple.columns:
        ranked_simple["final_score"] = renamed["final_score"]
    ranked_simple.to_csv(out_rank, index=False)

    if cfg.get("pipeline", {}).get("write_explanations", True):
        ensure_parent(explain_path)
        ensure_parent(top_candidates_path)
        ensure_parent(score_audit_report_csv)
        preferred = [
            "job_id",
            "cv_id",
            "source",
            "rank_for_job",
            "ranking_score",
            "tfidf_score",
            "semantic_score",
            "bm25_score",
            "skill_jaccard_score",
            "must_have_coverage",
            "nice_to_have_coverage",
            "skill_score",
            "experience_score",
            "cv_quality_score",
            "final_score_v1",
            "final_score_v2_bm25",
            "fusion_minmax_normalized_v1",
            "final_score_raw",
            "final_score",
            "learned_fusion_score",
            "score_check",
            "score_diff",
            "score_warning",
            "matched_required_skills",
            "missing_critical_skills",
            "matched_optional_skills",
            "missing_optional_skills",
            "cv_years_experience",
            "job_min_years_experience",
            "explanation",
            "suggested_improvements",
        ]
        ordered = [c for c in preferred if c in renamed.columns]
        explained = renamed[ordered + [c for c in renamed.columns if c not in ordered]]
        explained.to_csv(explain_path, index=False)
        logger.info("Wrote explainable rankings: %s", explain_path)

        top_df = (
            explained.sort_values(["job_id", "rank_for_job"])
            .groupby("job_id", sort=False)
            .head(top_k)
        )
        top_df.to_csv(top_candidates_path, index=False)
        logger.info("Wrote top candidates by job: %s", top_candidates_path)

        ensure_parent(score_audit_report_csv.parent)
        warn_mask = explained["score_warning"].astype(str).str.len() > 0
        audit_report = pd.DataFrame(
            [
                {"metric": "pair_rows", "value": len(explained)},
                {"metric": "score_warning_rows", "value": int(warn_mask.sum())},
                {
                    "metric": "max_score_diff",
                    "value": float(explained["score_diff"].astype(float).max()),
                },
            ]
        )
        audit_report.to_csv(score_audit_report_csv, index=False)

    # ------------------------------------------------------------------
    # Offline evaluation
    # ------------------------------------------------------------------
    metrics: dict[str, float] = {}
    gt_rel = paths.get("ground_truth")
    if evaluate is False:
        logger.info("Offline evaluation skipped (--no-evaluate).")
    elif not gt_rel:
        logger.warning(
            "Evaluation skipped: paths.ground_truth not set "
            "(expected e.g. data/evaluation/ground_truth.csv)."
        )
    else:
        gt_file = resolve_path(root, gt_rel)
        if gt_file.is_file():
            eval_df, comp_df = evaluate_models(root, cfg, semantic=semantic)
            if not eval_df.empty:
                row = eval_df.loc[eval_df["model"] == "Hybrid V1"]
                if not row.empty:
                    for col in eval_df.columns:
                        if col == "model":
                            continue
                        try:
                            metrics[str(col)] = float(row.iloc[0][col])
                        except (TypeError, ValueError):
                            pass
                for k, v in metrics.items():
                    logger.info("%s: %.4f", k, v)
        else:
            logger.warning(
                "Evaluation skipped: %s not found.",
                gt_rel,
            )

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
            root,
            cfg,
            artifact_paths,
            metrics,
            notes=json.dumps({"dense_enabled": m.dense_enabled, "bm25_enabled": m.bm25_enabled}),
        )

    logger.info("Wrote rankings: %s", out_rank)
    logger.info("Saved TF-IDF model: %s", feat_model)
    return metrics
