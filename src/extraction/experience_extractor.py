from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExperienceSignals:
    years_mentioned: list[float]
    role_hints: list[str]


_ROLE_PATTERNS = (
    r"\b(intern|internship)\b",
    r"\b(junior|jr\.?)\b",
    r"\b(senior|sr\.?|lead|principal|staff)\b",
    r"\b(engineer|developer|scientist|analyst|consultant)\b",
)


def extract_experience_signals(text: str) -> ExperienceSignals:
    if not text:
        return ExperienceSignals(years_mentioned=[], role_hints=[])
    t = text.lower()
    years: list[float] = []
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?\.?)\s+(?:of\s+)?(?:experience|exp\.?)",
        t,
    ):
        try:
            years.append(float(m.group(1)))
        except ValueError:
            continue
    for m in re.finditer(r"(?:experience|exp\.?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?\.?)", t):
        try:
            years.append(float(m.group(1)))
        except ValueError:
            continue
    roles: list[str] = []
    for pat in _ROLE_PATTERNS:
        for m in re.finditer(pat, t, re.IGNORECASE):
            roles.append(m.group(1))
    return ExperienceSignals(years_mentioned=years, role_hints=sorted(set(roles)))


def extract_job_required_years(text: str) -> float | None:
    """Minimum required years of experience mentioned in a job posting (heuristic)."""
    if not text:
        return None
    t = text.lower()
    candidates: list[float] = []
    for m in re.finditer(
        r"(?:at least|minimum|min\.?)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?\.?)",
        t,
    ):
        try:
            candidates.append(float(m.group(1)))
        except ValueError:
            continue
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?\.?)\s+(?:of\s+)?(?:experience|exp\.?)",
        t,
    ):
        try:
            candidates.append(float(m.group(1)))
        except ValueError:
            continue
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?\.?)", t):
        try:
            candidates.append(float(m.group(1)))
        except ValueError:
            continue
    return max(candidates) if candidates else None


def cv_max_years(signals: ExperienceSignals) -> float:
    return max(signals.years_mentioned) if signals.years_mentioned else 0.0
