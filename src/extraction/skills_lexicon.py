from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillEntry:
    skill_id: str
    display: str
    category: str
    patterns: list[re.Pattern[str]] = field(default_factory=list)


@dataclass
class SkillsLexicon:
    skills: dict[str, SkillEntry]
    category_by_skill: dict[str, str]

    def categories_for(self, skill_ids: set[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for sid in sorted(skill_ids):
            cat = self.category_by_skill.get(sid, "uncategorized")
            out.setdefault(cat, []).append(self.skills[sid].display if sid in self.skills else sid)
        return out


def _compile_patterns(skill_id: str, display: str, aliases: list[str]) -> list[re.Pattern[str]]:
    parts: list[str] = []
    for raw in [skill_id.replace("_", " "), display] + list(aliases):
        s = str(raw).strip().lower()
        if not s or s in parts:
            continue
        parts.append(s)
    pats: list[re.Pattern[str]] = []
    for token in parts:
        if not token:
            continue
        escaped = re.escape(token)
        if " " in token or "." in token or "#" in token or "/" in token:
            pats.append(re.compile(rf"(?:^|[^a-z0-9]){escaped}(?:$|[^a-z0-9])", re.IGNORECASE))
        else:
            pats.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
    return pats


def load_skills_lexicon(path: str | Path) -> SkillsLexicon:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"skills lexicon not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    skills_raw = raw.get("skills") or {}
    entries: dict[str, SkillEntry] = {}
    cat_by: dict[str, str] = {}
    for skill_id, meta in skills_raw.items():
        if not isinstance(meta, dict):
            continue
        display = str(meta.get("display", skill_id))
        category = str(meta.get("category", "uncategorized"))
        aliases = [str(a) for a in (meta.get("aliases") or []) if str(a).strip()]
        entry = SkillEntry(
            skill_id=skill_id,
            display=display,
            category=category,
            patterns=_compile_patterns(skill_id, display, aliases),
        )
        entries[skill_id] = entry
        cat_by[skill_id] = category
    return SkillsLexicon(skills=entries, category_by_skill=cat_by)


_DEFAULT_LEXICON: SkillsLexicon | None = None


def default_lexicon(root: Path) -> SkillsLexicon:
    global _DEFAULT_LEXICON
    if _DEFAULT_LEXICON is None:
        _DEFAULT_LEXICON = load_skills_lexicon(root / "config" / "skills.yaml")
    return _DEFAULT_LEXICON
