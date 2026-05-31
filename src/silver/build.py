"""Silver-layer artifacts produced from Bronze-aligned rows during ingest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.extraction.experience_extractor import (
    cv_total_years_estimate,
    extract_experience_signals,
    extract_job_required_years,
)
from src.extraction.requirements_extractor import extract_job_requirements
from src.extraction.skill_extractor import extract_skill_ids_sets_for_corpus
from src.extraction.skills_lexicon import SkillsLexicon, load_skills_lexicon
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.pii import anonymize_text
from src.processing.cv_sections import (
    cv_quality_score,
    segment_cv,
    write_unified_resumes_jsonl,
)
from src.utils.helpers import ensure_parent, resolve_path
from src.utils.id_normalization import normalize_cv_id, normalize_job_id

logger = logging.getLogger(__name__)

SHORT_TEXT_CHARS = 40


def _dedupe_cv_first(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    dropped = 0
    for r in rows:
        cid = normalize_cv_id(r.get("cv_id", "") or "")
        if not cid:
            continue
        if cid in seen:
            dropped += 1
            continue
        seen.add(cid)
        out.append(r)
    return out, dropped


def _dedupe_jobs_first(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    dropped = 0
    for r in rows:
        jid = normalize_job_id(r.get("job_id", "") or "")
        if not jid:
            continue
        if jid in seen:
            dropped += 1
            continue
        seen.add(jid)
        out.append(r)
    return out, dropped


def _cv_bundles(
    row: dict[str, Any],
    *,
    lex: SkillsLexicon,
    cleaner: TextCleaner,
    anonymize_docs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cid = normalize_cv_id(row["cv_id"])
    raw_base = str(row.get("raw_text", "") or "")
    src = str(row.get("source", "") or "").strip()
    sfile = str(row.get("source_file", "") or "").strip()

    anon_text = anonymize_text(raw_base) if anonymize_docs else raw_base
    cleaned = cleaner.clean(anon_text)
    sections = segment_cv(anon_text)
    skill_ids: set[str] = set()
    for part in sections.values():
        skill_ids |= set(extract_skill_ids_sets_for_corpus([part], lex)[0])
    skill_ids |= set(extract_skill_ids_sets_for_corpus([anon_text], lex)[0])
    cats = lex.categories_for(skill_ids)
    sig = extract_experience_signals(anon_text)
    years = float(cv_total_years_estimate(sig))
    q = cv_quality_score(sections, anon_text)

    labs = row.get("labels")
    gold_ents: list[Any] = []
    if isinstance(labs, dict):
        ge = labs.get("entities")
        if isinstance(ge, list):
            gold_ents = [e for e in ge if isinstance(e, dict)]

    unified = {
        "record_id": cid,
        "cv_id": cid,
        "source": src,
        "source_file": sfile,
        "raw_text": raw_base,
        "cleaned_text": cleaned,
        "sections": sections,
        "extracted_skills": sorted(skill_ids),
        "skill_categories": cats,
        "total_years_experience": years,
        "cv_quality_score": q,
    }
    if gold_ents:
        unified["gold_ner_entities"] = gold_ents
    profile = {
        "cv_id": cid,
        "source": src,
        "source_file": sfile,
        "total_years_experience": years,
        "cv_quality_score": q,
        "skills_count": len(skill_ids),
        "skill_categories": cats,
        "extracted_skill_ids_sample": sorted(skill_ids)[:50],
        "section_lengths": {k: len(v or "") for k, v in sections.items()},
        "silver_layer": "resume_profile_v1",
    }
    if gold_ents:
        profile["gold_ner_entity_count"] = len(gold_ents)
    return unified, profile


def _job_bundles(
    row: dict[str, Any],
    *,
    lex: SkillsLexicon,
    cleaner: TextCleaner,
    anonymize_docs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    jid = normalize_job_id(row["job_id"])
    raw_base = str(row.get("raw_text", "") or "")
    anon_text = anonymize_text(raw_base) if anonymize_docs else raw_base
    cleaned = cleaner.clean(anon_text)
    reqs = extract_job_requirements(anon_text, lex)
    jr = extract_job_required_years(anon_text)
    must_s = reqs.must_have
    nice_s = reqs.nice_to_have

    unified = {
        "job_id": jid,
        "source": str(row.get("source", "") or ""),
        "source_file": str(row.get("source_file", "") or ""),
        "title": str(row.get("title", "") or ""),
        "raw_text": raw_base,
        "cleaned_text": cleaned,
        "must_have_skills_lex": sorted(must_s)[:120],
        "nice_to_have_skills_lex": sorted(nice_s)[:120],
        "job_min_years_experience": float(jr) if jr else None,
    }
    profile = {
        "job_id": jid,
        "source": unified["source"],
        "source_file": unified["source_file"],
        "title": unified["title"],
        "must_have_skill_count": len(must_s),
        "nice_to_have_skill_count": len(nice_s),
        "job_min_years_experience": jr,
        "silver_layer": "job_profile_v1",
    }
    return unified, profile


def write_silver_artifacts(
    root: Path,
    cfg: dict[str, Any],
    *,
    profile_cv_rows: list[dict[str, Any]],
    job_rows_in: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write ``unified_resumes.jsonl``, profile JSONL files, and ``silver_stats.json``."""
    silver_cfg = cfg.get("silver", {})
    unify_path = resolve_path(
        root, silver_cfg.get("unified_resumes", "data/silver/unified_resumes.jsonl")
    )
    resume_prof = resolve_path(
        root, silver_cfg.get("resume_profiles", "data/silver/resume_profiles.jsonl")
    )
    job_prof = resolve_path(root, silver_cfg.get("job_profiles", "data/silver/job_profiles.jsonl"))
    stats_path = resolve_path(root, silver_cfg.get("stats_path", "data/silver/silver_stats.json"))

    lex_path = resolve_path(root, cfg.get("skills", {}).get("path", "config/skills.yaml"))
    lex = load_skills_lexicon(lex_path)
    pre = cfg.get("preprocessing", {})
    cleaner = TextCleaner(
        remove_stopwords=pre.get("remove_stopwords", True),
        lemmatize=pre.get("lemmatize", True),
        language=pre.get("language", "en"),
    )
    anon = bool(cfg.get("privacy", {}).get("anonymize", True))

    prof_rows_dedup, dup_c = _dedupe_cv_first(profile_cv_rows)
    job_rows_dedup, dup_j = _dedupe_jobs_first(job_rows_in)

    uni_cv: list[dict[str, Any]] = []
    prof_cv_lines: list[dict[str, Any]] = []
    dropped_short = 0
    for r in prof_rows_dedup:
        if len(str(r.get("raw_text", "") or "").strip()) < SHORT_TEXT_CHARS:
            dropped_short += 1
            continue
        u, p = _cv_bundles(r, lex=lex, cleaner=cleaner, anonymize_docs=anon)
        uni_cv.append(u)
        prof_cv_lines.append(p)

    prof_job_lines: list[dict[str, Any]] = []
    for r in job_rows_dedup:
        if len(str(r.get("raw_text", "") or "").strip()) < SHORT_TEXT_CHARS:
            continue
        _, p = _job_bundles(r, lex=lex, cleaner=cleaner, anonymize_docs=anon)
        prof_job_lines.append(p)

    write_unified_resumes_jsonl(uni_cv, unify_path)

    ensure_parent(resume_prof)
    with resume_prof.open("w", encoding="utf-8", newline="\n") as fr:
        for row in prof_cv_lines:
            fr.write(json.dumps(row, ensure_ascii=False) + "\n")

    ensure_parent(job_prof)
    with job_prof.open("w", encoding="utf-8", newline="\n") as fj:
        for row in prof_job_lines:
            fj.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats: dict[str, Any] = {
        "silver_version": "1",
        "unified_resumes_lines": len(uni_cv),
        "resume_profiles_lines": len(prof_cv_lines),
        "job_profiles_lines": len(prof_job_lines),
        "duplicate_cv_dropped": dup_c,
        "duplicate_job_dropped": dup_j,
        "profiles_dropped_short_text": dropped_short,
        "privacy_anonymized": anon,
        "paths": {
            "unified_resumes": str(unify_path),
            "resume_profiles": str(resume_prof),
            "job_profiles": str(job_prof),
            "silver_stats": str(stats_path),
        },
    }
    ensure_parent(stats_path)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Silver artifact stats: %s", stats_path)
    return stats


def read_cv_quality_scores(path: Path) -> dict[str, float]:
    """Map ``cv_id`` → ``cv_quality_score`` from ``resume_profiles.jsonl``."""
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = str(obj.get("cv_id", "") or "").strip()
            cid = normalize_cv_id(cid)
            if not cid:
                continue
            try:
                out[cid] = float(obj.get("cv_quality_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                out[cid] = 0.0
    return out
