from __future__ import annotations

import numpy as np
import pandas as pd

from src.extraction.requirements_extractor import JobRequirements, pair_requirement_summary
from src.extraction.skills_lexicon import SkillsLexicon
from src.scoring.explain import full_explanation_text, suggested_improvements_text


def rank_candidates_for_jobs(
    sim_matrix: np.ndarray,
    cv_ids: list,
    job_ids: list,
    top_k: int,
    component_matrices: dict[str, np.ndarray] | None = None,
    *,
    score_column: str = "score",
) -> pd.DataFrame:
    """
    sim_matrix shape: (n_cvs, n_jobs) — similarity[cv_idx, job_idx].
    Returns long table: job_id, cv_id, score, rank_for_job, optional channel scores.
    """
    rows: list[dict] = []
    n_cvs, n_jobs = sim_matrix.shape
    comps = component_matrices or {}
    for j in range(n_jobs):
        col = sim_matrix[:, j]
        order = np.argsort(-col)[:top_k]
        for rank, i in enumerate(order, start=1):
            row: dict = {
                "job_id": job_ids[j],
                "cv_id": cv_ids[i],
                score_column: float(col[i]),
                "rank_for_job": rank,
            }
            for name, mat in comps.items():
                row[f"score_{name}"] = float(mat[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def enrich_with_explanations(
    ranked: pd.DataFrame,
    cv_ids: list,
    job_ids: list,
    cv_skill_sets: list[set[str]],
    job_skill_sets: list[set[str]],
    cv_years: list[float],
    job_required: list[float | None],
) -> pd.DataFrame:
    from src.scoring.explain import explain_pair

    cv_pos = {str(cid): idx for idx, cid in enumerate(cv_ids)}
    job_pos = {str(jid): idx for idx, jid in enumerate(job_ids)}
    matched: list[str] = []
    missing: list[str] = []
    notes: list[str] = []
    for _, r in ranked.iterrows():
        i = cv_pos[str(r["cv_id"])]
        j = job_pos[str(r["job_id"])]
        ex = explain_pair(cv_skill_sets[i], job_skill_sets[j], cv_years[i], job_required[j])
        matched.append(ex["matched_skills"])
        missing.append(ex["missing_skills"])
        notes.append(ex["experience_note"])
    out = ranked.copy()
    out["matched_skills"] = matched
    out["missing_skills"] = missing
    out["experience_note"] = notes
    return out


def enrich_detailed(
    ranked: pd.DataFrame,
    cv_ids: list,
    job_ids: list,
    cv_skill_sets: list[set[str]],
    job_reqs: list[JobRequirements],
    *,
    cv_years: list[float],
    job_required_years: list[float | None],
    must_cov: np.ndarray,
    nice_cov: np.ndarray,
    semantic_mat: np.ndarray,
    lex: SkillsLexicon,
) -> pd.DataFrame:
    cv_pos = {str(cid): idx for idx, cid in enumerate(cv_ids)}
    job_pos = {str(jid): idx for idx, jid in enumerate(job_ids)}
    rows = []
    for _, r in ranked.iterrows():
        i = cv_pos[str(r["cv_id"])]
        j = job_pos[str(r["job_id"])]
        req = job_reqs[j]
        det = pair_requirement_summary(cv_skill_sets[i], req)
        sem = float(semantic_mat[i, j])
        note = ""
        jr = job_required_years[j]
        if jr is None or jr <= 0:
            note = "ilan_yılı_belirsiz"
        elif cv_years[i] >= jr:
            note = f"deneyim_tamam:{cv_years[i]:.1f}_>=_{jr:.1f}"
        else:
            note = f"deneyim_eksik:{cv_years[i]:.1f}_<_gerekli_{jr:.1f}"
        matched_line = str(det["matched_required_skills"])
        if matched_line.strip():
            disp_m = ";".join(
                lex.skills[x].display for x in matched_line.split(";") if x and x in lex.skills
            )
        else:
            disp_m = ""
        expl = full_explanation_text(
            matched_display=disp_m,
            missing_critical=str(det["missing_critical_skills"]),
            semantic_sim=sem,
            exp_note=note,
            must_cov=float(must_cov[i, j]),
            nice_cov=float(nice_cov[i, j]),
        )
        sug = suggested_improvements_text(
            missing_critical=str(det["missing_critical_skills"]),
            missing_optional=str(det["missing_optional_skills"]),
            exp_note=note,
            semantic_sim=sem,
        )
        row = r.to_dict()
        row.update(
            {
                "must_have_coverage": float(must_cov[i, j]),
                "nice_to_have_coverage": float(nice_cov[i, j]),
                "matched_required_skills": det["matched_required_skills"],
                "missing_critical_skills": det["missing_critical_skills"],
                "matched_optional_skills": det["matched_optional_skills"],
                "missing_optional_skills": det["missing_optional_skills"],
                "cv_years_experience": float(cv_years[i]),
                "job_min_years_experience": float(job_required_years[j]) if job_required_years[j] else 0.0,
                "experience_note": note,
                "explanation": expl,
                "suggested_improvements": sug,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
