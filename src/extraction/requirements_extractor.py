from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.config.defaults import SKILL_SCORE_MUST_WEIGHT, SKILL_SCORE_NICE_WEIGHT
from src.extraction.skill_extractor import extract_skill_ids
from src.extraction.skills_lexicon import SkillsLexicon

_MUST_HINTS_EN = (
    r"\bmust[- ]have\b",
    r"\brequired\b",
    r"\bmandatory\b",
    r"\bminimum\b",
    r"\bqualifications\b",
    r"\brequirements\b",
)
_NICE_HINTS_EN = (
    r"\bnice[- ]to[- ]have\b",
    r"\bpreferred\b",
    r"\bplus\b",
    r"\bbonus\b",
    r"\boptional\b",
)
_MUST_HINTS_TR = (
    r"\bzorunlu\b",
    r"\bgerekli\b",
    r"\bşart\b",
    r"\bminimum\b",
    r"\baranan\b",
)
_NICE_HINTS_TR = (
    r"\btercih\b",
    r"\bartı\b",
    r"\biyi olur\b",
    r"\bistenir\b",
    r"\bek olursa\b",
)


def _split_sections(text: str) -> tuple[str, str, str]:
    if not text:
        return "", "", ""
    lines = text.splitlines()
    must_idx: list[int] = []
    nice_idx: list[int] = []
    for i, line in enumerate(lines):
        l = line.strip().lower()
        if any(re.search(p, l) for p in _MUST_HINTS_EN + _MUST_HINTS_TR):
            must_idx.append(i)
        if any(re.search(p, l) for p in _NICE_HINTS_EN + _NICE_HINTS_TR):
            nice_idx.append(i)
    if not must_idx and not nice_idx:
        return text, text, ""

    def block_from(start_positions: list[int], other_positions: list[int]) -> str:
        if not start_positions:
            return ""
        start = min(start_positions)
        end = len(lines)
        for other in other_positions:
            if other > start:
                end = min(end, other)
        return "\n".join(lines[start:end])

    must_block = block_from(must_idx, nice_idx) if must_idx else ""
    nice_block = block_from(nice_idx, must_idx) if nice_idx else ""
    head_end = min(must_idx + nice_idx) if (must_idx or nice_idx) else 0
    head = "\n".join(lines[:head_end])
    if not must_block:
        must_block = "\n".join(lines) if not head else head + "\n" + "\n".join(lines[head_end:])
        if not must_block.strip():
            must_block = text
    return head, must_block, nice_block


@dataclass
class JobRequirements:
    must_have: set[str]
    nice_to_have: set[str]


def extract_job_requirements(text: str, lex: SkillsLexicon) -> JobRequirements:
    _, must_text, nice_text = _split_sections(text)
    must_ids = set(extract_skill_ids(must_text, lex))
    nice_ids = set(extract_skill_ids(nice_text, lex))
    nice_ids -= must_ids
    if not must_ids and not nice_ids:
        all_ids = set(extract_skill_ids(text, lex))
        return JobRequirements(must_have=all_ids, nice_to_have=set())
    return JobRequirements(must_have=must_ids, nice_to_have=nice_ids)


def coverage_sets(cv_skills: set[str], required: set[str]) -> tuple[float, set[str], set[str]]:
    if not required:
        return 1.0, set(), set()
    matched = cv_skills & required
    cov = len(matched) / len(required)
    return cov, matched, required - matched


def skill_score_from_coverage(must_cov: float, nice_cov: float, has_nice: bool) -> float:
    if not has_nice:
        return float(must_cov)
    return SKILL_SCORE_MUST_WEIGHT * float(must_cov) + SKILL_SCORE_NICE_WEIGHT * float(nice_cov)


def requirement_coverage_matrix(
    cv_skill_sets: list[set[str]],
    job_reqs: list[JobRequirements],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_c, n_j = len(cv_skill_sets), len(job_reqs)
    must_cov_m = np.zeros((n_c, n_j), dtype=np.float64)
    nice_cov_m = np.zeros((n_c, n_j), dtype=np.float64)
    skill_score_m = np.zeros((n_c, n_j), dtype=np.float64)
    for j in range(n_j):
        req = job_reqs[j]
        has_nice = bool(req.nice_to_have)
        for i in range(n_c):
            cv = cv_skill_sets[i]
            must_cov, _, _ = coverage_sets(cv, req.must_have)
            if has_nice:
                nice_cov, _, _ = coverage_sets(cv, req.nice_to_have)
            else:
                nice_cov = 1.0
            must_cov_m[i, j] = must_cov
            nice_cov_m[i, j] = nice_cov if has_nice else 0.0
            skill_score_m[i, j] = skill_score_from_coverage(must_cov, nice_cov, has_nice)
    return must_cov_m, nice_cov_m, skill_score_m


def pair_requirement_summary(cv_skills: set[str], req: JobRequirements) -> dict[str, object]:
    must_cov, m_match, m_miss = coverage_sets(cv_skills, req.must_have)
    has_nice = bool(req.nice_to_have)
    if has_nice:
        nice_cov, o_match, o_miss = coverage_sets(cv_skills, req.nice_to_have)
    else:
        nice_cov, o_match, o_miss = 1.0, set(), set()
    return {
        "must_have_coverage": must_cov,
        "nice_to_have_coverage": nice_cov,
        "matched_required_skills": ";".join(sorted(m_match)),
        "missing_critical_skills": ";".join(sorted(m_miss)),
        "matched_optional_skills": ";".join(sorted(o_match)),
        "missing_optional_skills": ";".join(sorted(o_miss)),
    }
