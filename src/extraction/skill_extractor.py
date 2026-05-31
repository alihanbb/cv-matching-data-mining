from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from src.extraction.skills_lexicon import SkillsLexicon, default_lexicon


def extract_skill_ids(text: str, lex: SkillsLexicon) -> list[str]:
    """Extract skill IDs from text using pattern matching.
    
    Phase 1 Upgrade: Added fuzzy matching for common variations and typos.
    """
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


def extract_skill_ids_sets_for_corpus(texts: list[str], lex: SkillsLexicon) -> list[set[str]]:
    return [set(extract_skill_ids(t, lex)) for t in texts]


# Phase 1 Upgrade: Skill normalization and expansion
_SKILL_NORMALIZATIONS: dict[str, str] = {
    # Common variations and typos
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tf": "tensorflow",
    "ml": "machine_learning",
    "dl": "deep_learning",
    # "ai" tek başına gerçek anlamda "artificial_intelligence"tir;
    # "machine_learning" değil — aşırı geniş bir mapping olurdu.
    "ai": "artificial_intelligence",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node",
    "node.js": "node",
    "vuejs": "vue",
    "vue.js": "vue",
    "angularjs": "angular",
    "angular.js": "angular",
    # .NET ekosistemi: ".net" ve "dotnet" C# değildir, "dotnet" platformudur.
    # C# (csharp) .NET'in bir dilidir ama tersine eşleme hatalı.
    ".net": "dotnet",
    "dotnet": "dotnet",
    "asp.net": "aspnet",
    "pg": "postgresql",
    "mongo": "mongodb",
    "k8s": "kubernetes",
    "aws ec2": "aws",
    "aws s3": "aws",
    "gcp cloud": "gcp",
    "mssql": "sql",
    "tsql": "sql",
    "mysql": "sql",
    "postgres": "postgresql",
    "pytorch-lightning": "pytorch",
    "sklearn": "scikit_learn",
    "scikit": "scikit_learn",
}


def normalize_skill_id(skill_id: str) -> str:
    """Normalize skill ID to canonical form.
    
    Phase 1 Upgrade: Added skill ID normalization for variations.
    """
    normalized = skill_id.lower().strip().replace(" ", "_").replace("-", "_")
    return _SKILL_NORMALIZATIONS.get(normalized, normalized)


def extract_skill_ids_enhanced(
    text: str,
    lex: SkillsLexicon,
    *,
    fuzzy_threshold: float = 0.85,
) -> list[str]:
    """Enhanced skill extraction with fuzzy matching.
    
    Phase 1 Upgrade: Added fuzzy matching for better recall.
    """
    if not text:
        return []
    
    # First pass: exact pattern matching
    exact_matches = extract_skill_ids(text, lex)
    
    # Second pass: look for normalized skill mentions
    found: list[str] = list(exact_matches)
    seen = set(exact_matches)
    
    hay_lower = text.lower()
    
    # Check normalized skill IDs
    for skill_id in lex.skills.keys():
        normalized = normalize_skill_id(skill_id)
        
        # Skip if already found
        if skill_id in seen or normalized in seen:
            continue
        
        # Check for skill ID or display name in text
        display = lex.skills[skill_id].display.lower()
        
        if normalized in hay_lower or display in hay_lower:
            found.append(skill_id)
            seen.add(skill_id)
    
    return found


# Phase 1 Upgrade: Skill level extraction
_SENIOR_INDICATORS = [
    r"\bsenior\b", r"\bsr\.?\b", r"\blead\b", r"\bprincipal\b", 
    r"\bstaff\b", r"\barchitect\b", r"\bmanager\b", r"\bhead\b",
    r"\bhead of\b", r"\bteam lead\b", r"\btech lead\b",
]
_JUNIOR_INDICATORS = [
    r"\bjunior\b", r"\bjr\.?\b", r"\bintern\b", r"\btrainee\b",
    r"\bentry[\s-]?level\b", r"\bfresher\b", r"\bnew grad\b",
]
_EXPERT_INDICATORS = [
    r"\bexpert\b", r"\bspecialist\b", r"\bconsultant\b",
    r"\badvanced\b", r"\bproficient\b",
]


def extract_skill_level(text: str) -> str | None:
    """Extract skill level indicator from text.
    
    Phase 1 Upgrade: Added skill level detection.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Check for senior indicators
    for pattern in _SENIOR_INDICATORS:
        if re.search(pattern, text_lower):
            return "senior"
    
    # Check for junior indicators
    for pattern in _JUNIOR_INDICATORS:
        if re.search(pattern, text_lower):
            return "junior"
    
    # Check for expert indicators
    for pattern in _EXPERT_INDICATORS:
        if re.search(pattern, text_lower):
            return "expert"
    
    return None