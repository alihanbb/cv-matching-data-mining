from __future__ import annotations


def explain_pair(
    cv_skills: set[str],
    job_skills: set[str],
    cv_years_max: float,
    job_required_years: float | None,
) -> dict[str, str]:
    matched = sorted(cv_skills & job_skills, key=str.lower)
    missing = sorted(job_skills - cv_skills, key=str.lower)
    if job_required_years is None or job_required_years <= 0:
        note = "ilan_yılı_belirsiz"
    elif cv_years_max >= job_required_years:
        note = f"deneyim_tamam:{cv_years_max:.1f}_>=_{job_required_years:.1f}"
    else:
        note = f"deneyim_eksik:{cv_years_max:.1f}_<_gerekli_{job_required_years:.1f}"
    return {
        "matched_skills": ";".join(matched),
        "missing_skills": ";".join(missing),
        "experience_note": note,
    }
