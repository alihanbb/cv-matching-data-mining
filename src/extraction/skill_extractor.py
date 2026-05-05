from __future__ import annotations

import re
from typing import Iterable

DEFAULT_SKILLS: tuple[str, ...] = (
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "tensorflow",
    "pytorch",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "spark",
    "kafka",
    "nlp",
    "machine learning",
    "deep learning",
    "data mining",
    "etl",
    "tableau",
    "power bi",
    "excel",
    "git",
    "linux",
    "agile",
    "scrum",
    "react",
    "node",
    "mongodb",
)

# Surface forms → canonical display (light normalization).
_SKILL_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "k8s": "kubernetes",
}


def extract_skills(text: str, skill_lexicon: Iterable[str] | None = None) -> list[str]:
    if not text:
        return []
    lex = tuple(skill_lexicon) if skill_lexicon else DEFAULT_SKILLS
    hay = text.lower()
    for alias, canonical in _SKILL_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", hay):
            hay = f"{hay} {canonical}"
    found: list[str] = []
    for skill in sorted(lex, key=len, reverse=True):
        s = skill.lower().strip()
        if not s:
            continue
        if " " in s:
            if s in hay:
                found.append(skill)
        elif re.search(rf"\b{re.escape(s)}\b", hay):
            found.append(skill)
    seen: set[str] = set()
    out: list[str] = []
    for s in found:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out
