from __future__ import annotations

from src.config.defaults import SEMANTIC_LOW_THRESHOLD


def explain_pair(
    cv_skills: set[str],
    job_skills: set[str],
    cv_years_max: float,
    job_required_years: float | None,
) -> dict[str, str]:
    """Backward-compatible explain without structured requirements."""
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


def full_explanation_text(
    *,
    matched_display: str,
    missing_critical: str,
    semantic_sim: float,
    exp_note: str,
    must_cov: float,
    nice_cov: float,
) -> str:
    parts = [
        f"Must-have coverage={must_cov:.2f}; nice-to-have coverage={nice_cov:.2f}.",
        f"Semantic similarity≈{semantic_sim:.3f}.",
        f"Experience: {exp_note}.",
        f"Matched skills: {matched_display or '—'}; Missing critical: {missing_critical or '—'}.",
    ]
    return " ".join(parts)


def suggested_improvements_text(
    *,
    missing_critical: str,
    missing_optional: str,
    exp_note: str,
    semantic_sim: float,
) -> str:
    sug: list[str] = []
    if missing_critical:
        sug.append(f"Öne çıkarın veya edinin: {missing_critical.replace(';', ', ')}.")
    if missing_optional:
        sug.append(f"İsteğe bağlı güçlü sinyaller: {missing_optional.replace(';', ', ')}.")
    if semantic_sim < SEMANTIC_LOW_THRESHOLD:
        sug.append("Özet ve deneyim bölümlerini ilan diline ve anahtar kelimelere yaklaştırın.")
    if "eksik" in exp_note or "gerekli" in exp_note:
        sug.append("Deneyim yılını net tarih aralıklarıyla gösterin.")
    return " ".join(sug) if sug else "Güçlü görünüyor; ölçülebilir sonuçlar eklemeye devam edin."
