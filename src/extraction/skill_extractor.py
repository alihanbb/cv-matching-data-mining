from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from src.extraction.skills_lexicon import SkillsLexicon, default_lexicon


def extract_skill_ids(text: str, lex: SkillsLexicon) -> list[str]:
    if not text:
        return []
    hay = text
    found: list[str] = []
    for sid, entry in lex.skills.items():
        for pat in entry.patterns:
            if pat.search(hay):
                found.append(sid)
                break
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def extract_skills(
    text: str, skill_lexicon: SkillsLexicon | None = None, *, root: Path | None = None
) -> list[str]:
    """Return display names for backward compatibility."""
    if skill_lexicon is None:
        if root is None:
            from src.utils.helpers import project_root

            root = project_root()
        skill_lexicon = default_lexicon(root)
    ids = extract_skill_ids(text, skill_lexicon)
    return [skill_lexicon.skills[i].display for i in ids if i in skill_lexicon.skills]


def skill_ids_to_display(ids: Iterable[str], lex: SkillsLexicon) -> list[str]:
    out: list[str] = []
    for i in ids:
        if i in lex.skills:
            out.append(lex.skills[i].display)
        else:
            out.append(str(i))
    return out


def extract_skill_ids_sets_for_corpus(
    texts: list[str], lex: SkillsLexicon
) -> list[set[str]]:
    return [set(extract_skill_ids(t, lex)) for t in texts]
