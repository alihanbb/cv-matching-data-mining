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

_EN_YEAR_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?\.?)\b",
    r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?\.?)\s+(?:of\s+)?(?:experience|exp\.?)\b",
    r"(?:at least|minimum|min\.?)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?\.?)\b",
    r"(?:experience|exp\.?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?\.?)\b",
]
_TR_YEAR_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*\+\s*(?:yıl|yil|senelik)\b",
    r"(\d+(?:\.\d+)?)\s*(?:yıl|yil|senelik)\s*(?:deneyim|tecrübe|tecrube)?",
    r"(?:en az|minimum|min\.?)\s*(\d+(?:\.\d+)?)\s*(?:yıl|yil|senelik)\b",
    r"(\d+(?:\.\d+)?)\s*(?:senelik|senelik\s+tecrübe|senelik\s+tecrube|yıllık\s+tecrübe|yillik\s+tecrube)\b",
]


def _collect_years(text: str) -> list[float]:
    years: list[float] = []
    for pat in _EN_YEAR_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                years.append(float(m.group(1)))
            except (ValueError, IndexError):
                continue
    lowered = text.lower()
    for pat in _TR_YEAR_PATTERNS:
        for m in re.finditer(pat, lowered):
            try:
                years.append(float(m.group(1)))
            except (ValueError, IndexError):
                continue
    return years


def extract_experience_signals(text: str) -> ExperienceSignals:
    if not text:
        return ExperienceSignals(years_mentioned=[], role_hints=[])
    t = text.lower()
    years = _collect_years(text)
    roles: list[str] = []
    for pat in _ROLE_PATTERNS:
        for m in re.finditer(pat, t, re.IGNORECASE):
            roles.append(m.group(1))
    return ExperienceSignals(years_mentioned=years, role_hints=sorted(set(roles)))


def extract_job_required_years(text: str) -> float | None:
    if not text:
        return None
    lowered = text.lower()
    candidates: list[float] = []
    for pat in _EN_YEAR_PATTERNS:
        for m in re.finditer(pat, lowered):
            try:
                candidates.append(float(m.group(1)))
            except (ValueError, IndexError):
                continue
    for pat in _TR_YEAR_PATTERNS:
        for m in re.finditer(pat, lowered):
            try:
                candidates.append(float(m.group(1)))
            except (ValueError, IndexError):
                continue
    return max(candidates) if candidates else None


def cv_max_years(signals: ExperienceSignals) -> float:
    return max(signals.years_mentioned) if signals.years_mentioned else 0.0


def cv_total_years_estimate(signals: ExperienceSignals) -> float:
    """Conservative total-years signal: max of explicit mentions (extendable)."""
    return cv_max_years(signals)
