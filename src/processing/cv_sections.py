from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config.defaults import CV_QUALITY_WEIGHTS, CV_QUALITY_MEASURABLE_DIVISOR

_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "summary",
        re.compile(r"\b(profile|summary|objective|about me|öz|özet|profil)\b", re.I),
    ),
    (
        "skills",
        re.compile(
            r"\b(skills|technologies|tech stack|yetenek|teknik beceriler)\b", re.I
        ),
    ),
    (
        "experience",
        re.compile(
            r"\b(experience|employment|work history|iş deneyimi|deneyim|tecrübe|tecrube)\b",
            re.I,
        ),
    ),
    ("projects", re.compile(r"\b(projects|portfolio|proje|projeler)\b", re.I)),
    (
        "education",
        re.compile(r"\b(education|academic|üniversite|eğitim|eğitim)\b", re.I),
    ),
    (
        "certificates",
        re.compile(
            r"\b(certificates|certifications|licenses|language|diller|sertifika)\b",
            re.I,
        ),
    ),
]


def segment_cv(text: str) -> dict[str, str]:
    if not text:
        return {name: "" for name, _ in _SECTION_PATTERNS}
    lines = text.splitlines()
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 3 or len(stripped) > 160:
            continue
        lower = stripped.lower()
        for name, pat in _SECTION_PATTERNS:
            if pat.search(lower):
                hits.append((i, name))
                break
    if not hits:
        out = {name: "" for name, _ in _SECTION_PATTERNS}
        out["summary"] = text
        return out

    hits = sorted(hits, key=lambda x: x[0])
    sections: dict[str, list[str]] = {name: [] for name, _ in _SECTION_PATTERNS}
    preamble = lines[: hits[0][0]]
    if preamble:
        sections["summary"].extend(preamble)
    for idx, (start_line, sec_name) in enumerate(hits):
        next_start = hits[idx + 1][0] if idx + 1 < len(hits) else len(lines)
        sections[sec_name].extend(lines[start_line:next_start])
    return {k: "\n".join(v).strip() for k, v in sections.items()}


_MEAS_ACHIEVEMENT = re.compile(
    r"(\d+(\.\d+)?%|\d+\s*x\b|\$\s?\d|saved|increased|decreased|reduced|raised|başarı|arttırdım|azalttım|%\s?artış)",
    re.I,
)


def measurable_hits(text: str) -> int:
    return len(_MEAS_ACHIEVEMENT.findall(text or ""))


def cv_quality_score(sections: dict[str, str], full_text: str) -> float:
    score = 0.0
    for key, w in CV_QUALITY_WEIGHTS.items():
        if key == "measurable":
            hits = measurable_hits(full_text)
            score += w * min(1.0, hits / CV_QUALITY_MEASURABLE_DIVISOR)
            continue
        body = sections.get(key, "")
        if body and len(body.strip()) > 20:
            score += w
    return float(min(1.0, max(0.0, score)))


def write_unified_resumes_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
