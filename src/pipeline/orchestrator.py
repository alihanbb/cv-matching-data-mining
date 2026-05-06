from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import precision_at_k, top_k_accuracy
from src.evaluation.ranking_metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k
from src.extraction.experience_extractor import (
    cv_max_years,
    extract_experience_signals,
    extract_job_required_years,
)
from src.extraction.skill_extractor import extract_skills
from src.features.semantic_encoder import dense_cosine_similarity, encode_normalized, try_load_semantic_encoder
from src.features.tfidf_vectorizer import TfidfFeatureBuilder
from src.ingest.build_processed import build_processed_from_raw
from src.models.matcher import enrich_with_explanations, rank_candidates_for_jobs
from src.models.similarity import cosine_pairs
from src.preprocessing.cleaner import TextCleaner
from src.schemas.documents import validate_ground_truth_df, validate_processed_df
from src.scoring.fusion import experience_match_matrix, fuse_scores, skill_jaccard_matrix
from src.utils.experiment import write_run_manifest
from src.utils.helpers import ensure_parent, resolve_path

logger = logging.getLogger(__name__)


def _read_processed_or_raise(path: Path, id_col: str) -> pd.DataFrame:
    """Read a Silver CSV; raise a clear error when missing or empty."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed table not found: {path}\n"
            "Run the ingest step first: python main.py --ingest"
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Processed table is empty (no header): {path}\n"
            "Place input files under data/bronze/, then re-run with --ingest."
        ) from None
    if id_col not in df.columns:
        raise ValueError(
            f"Processed table missing column '{id_col}': {path}\n"
            "Re-run ingest with valid bronze data."
        )
    return df


def run_full_pipeline(
    root: Path,
    cfg: dict[str, Any],
    *,
    ingest: bool = False,
    semantic: bool = True,
    evaluate: bool | None = None,
) -> dict[str, float]:
    """Run the full CV–job matching pipeline.

    Parameters
    ----------
    ingest:
        Rebuild Silver CSVs from Bronze before scoring.
    semantic:
        Enable dense embedding channel when True (and the model is available).
    evaluate:
        Force-on / force-off evaluation. When ``None`` (default), evaluation
        runs automatically when ``data/evaluation/ground_truth.csv`` exists.
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

    if ingest:
        ing = cfg.get("ingest", {})
        raw_cvs = resolve_path(root, ing.get("raw_cvs_dir", "data/bronze/cvs"))
        raw_jobs = resolve_path(root, ing.get("raw_jobs_dir", "data/bronze/job_descriptions"))
        n_cv, n_job = build_processed_from_raw(raw_cvs, raw_jobs, proc_cvs, proc_jobs)
        logger.info("Ingest: %d CV files, %d job files -> processed CSVs", n_cv, n_job)

    pre = cfg.get("preprocessing", {})
    cleaner = TextCleaner(
        remove_stopwords=pre.get("remove_stopwords", True),
        lemmatize=pre.get("lemmatize", True),
        language=pre.get("language", "en"),
    )

    cvs = _read_processed_or_raise(proc_cvs, "cv_id")
    jobs = _read_processed_or_raise(proc_jobs, "job_id")
    cvs = validate_processed_df(cvs, "cv_id", "text")
    jobs = validate_processed_df(jobs, "job_id", "text")
    if cvs.empty or jobs.empty:
        raise ValueError(
            "Processed CV or job table is empty.\n"
            f" - CVs:  {proc_cvs}\n"
            f" - Jobs: {proc_jobs}\n"
            "Add files under data/bronze/cvs and data/bronze/job_descriptions, "
            "then re-run with --ingest."
        )

    cvs["clean_text"] = cvs["text"].map(cleaner.clean)
    jobs["clean_text"] = jobs["text"].map(cleaner.clean)

    cv_ids = [str(x) for x in cvs["cv_id"].tolist()]
    job_ids = [str(x) for x in jobs["job_id"].tolist()]

    cv_skill_sets = [set(s.lower() for s in extract_skills(t)) for t in cvs["text"]]
    job_skill_sets = [set(s.lower() for s in extract_skills(t)) for t in jobs["text"]]
    cv_years_list = [cv_max_years(extract_experience_signals(t)) for t in cvs["text"]]
    job_req_list = [extract_job_required_years(t) for t in jobs["text"]]

    skills_mat = skill_jaccard_matrix(cv_skill_sets, job_skill_sets)
    exp_mat = experience_match_matrix(cv_years_list, job_req_list)

    tfidf_cfg = cfg.get("tfidf", {})
    builder = TfidfFeatureBuilder(tfidf_cfg)
    corpus = cvs["clean_text"].tolist() + jobs["clean_text"].tolist()
    builder.fit(corpus)
    X_cv = builder.transform(cvs["clean_text"].tolist())
    X_job = builder.transform(jobs["clean_text"].tolist())
    ensure_parent(feat_model)
    builder.save(feat_model)

    sim_lex = cosine_pairs(X_cv, X_job)

    emb_cfg = cfg.get("embeddings", {})
    dense_enabled = bool(semantic and emb_cfg.get("enabled", True))
    dense_sim = None
    model = None
    if dense_enabled:
        model = try_load_semantic_encoder(emb_cfg.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
        if model is None:
            dense_enabled = False
        else:
            bs = int(emb_cfg.get("batch_size", 32))
            logger.info("Encoding %d CVs and %d jobs with dense model", len(cvs), len(jobs))
            e_cv = encode_normalized(model, cvs["clean_text"].tolist(), bs)
            e_job = encode_normalized(model, jobs["clean_text"].tolist(), bs)
            dense_sim = dense_cosine_similarity(e_cv, e_job)

    fusion_cfg = cfg.get("fusion", {})
    weights = dict(fusion_cfg.get("weights", {}))
    if not weights:
        # Default split documented in README: balances lexical and semantic
        # signals with skill / experience as supporting structured channels.
        weights = {"tfidf": 0.35, "dense": 0.35, "skills": 0.20, "experience": 0.10}

    fused, w_used = fuse_scores(sim_lex, dense_sim, skills_mat, exp_mat, weights, dense_enabled)
    logger.info("Fusion weights (normalized): %s", w_used)

    top_k = int(cfg.get("matching", {}).get("top_k", 10))
    components = {"tfidf": sim_lex, "skills": skills_mat, "experience": exp_mat}
    if dense_sim is not None:
        components["dense"] = dense_sim

    ranked = rank_candidates_for_jobs(fused, cv_ids, job_ids, top_k, components)
    ranked = enrich_with_explanations(
        ranked,
        cv_ids,
        job_ids,
        cv_skill_sets,
        job_skill_sets,
        cv_years_list,
        job_req_list,
    )

    ensure_parent(out_rank)
    ranked_out = ranked.drop(
        columns=["matched_skills", "missing_skills", "experience_note"],
        errors="ignore",
    )
    ranked_out.to_csv(out_rank, index=False)

    if cfg.get("pipeline", {}).get("write_explanations", True):
        ensure_parent(explain_path)
        explained = ranked.rename(
            columns={
                "score": "final_score",
                "score_tfidf": "tfidf_score",
                "score_dense": "semantic_score",
                "score_skills": "skill_score",
                "score_experience": "experience_score",
                "experience_note": "explanation",
            }
        )
        if "semantic_score" not in explained.columns:
            explained["semantic_score"] = 0.0
        preferred = [
            "job_id",
            "cv_id",
            "rank_for_job",
            "tfidf_score",
            "semantic_score",
            "skill_score",
            "experience_score",
            "final_score",
            "matched_skills",
            "missing_skills",
            "explanation",
        ]
        ordered = [c for c in preferred if c in explained.columns]
        explained = explained[ordered + [c for c in explained.columns if c not in ordered]]
        explained.to_csv(explain_path, index=False)
        logger.info("Wrote explainable rankings: %s", explain_path)

    metrics: dict[str, float] = {}
    gt_path = paths.get("ground_truth")
    run_eval = bool(evaluate) if evaluate is not None else True
    if gt_path and run_eval:
        gt_file = resolve_path(root, gt_path)
        if gt_file.is_file():
            gt = validate_ground_truth_df(pd.read_csv(gt_file))
            eval_cfg = cfg.get("evaluation", {})
            ks = eval_cfg.get("top_k_values", [1, 3, 5])
            for k in ks:
                metrics[f"topk_hit_rate_{k}"] = float(top_k_accuracy(ranked_out, gt, int(k)))
                metrics[f"precision_at_{k}"] = float(precision_at_k(ranked_out, gt, int(k)))
                metrics[f"ndcg_at_{k}"] = float(ndcg_at_k(ranked_out, gt, int(k)))
            metrics["mrr"] = float(mean_reciprocal_rank(ranked_out, gt))
            metrics["map"] = float(mean_average_precision(ranked_out, gt))
            for k, v in metrics.items():
                logger.info("%s: %.4f", k, v)
        elif evaluate:
            raise FileNotFoundError(
                f"Ground truth file not found: {gt_file}\n"
                "Provide a ground_truth.csv (job_id,cv_id,relevance) before --evaluate."
            )
        else:
            logger.info("Ground truth missing at %s — skipping metrics.", gt_file)

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
            notes="dense_enabled=%s" % dense_enabled,
        )

    logger.info("Wrote rankings: %s", out_rank)
    logger.info("Saved TF-IDF model: %s", feat_model)
    return metrics
