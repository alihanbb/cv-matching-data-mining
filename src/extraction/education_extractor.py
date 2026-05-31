"""Education extraction and matching for CV-Job matching.

Phase 2 Upgrade: Added education level matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Education level hierarchy (higher = more qualified)
EDUCATION_LEVELS = {
    "phd": 7,
    "doctorate": 7,
    "doctoral": 7,
    "postdoc": 6,
    "postdoctoral": 6,
    "master": 5,
    "msc": 5,
    "mba": 5,
    "ma": 5,
    "meng": 5,
    "bachelor": 4,
    "bsc": 4,
    "ba": 4,
    "btech": 4,
    "be": 4,
    "associate": 3,
    "diploma": 2,
    "certificate": 1,
    "high_school": 0,
    "secondary": 0,
}

# Degree keywords by language
_DEGREE_PATTERNS_EN = [
    r"\b(phd|doctorate|doctoral)\b",
    r"\b(master|msc|mba|ma|meng)\b",
    r"\b(bachelor|bsc|ba|btech|be)\b",
    r"\b(associate|diploma)\b",
    r"\b(certificate|certification)\b",
    r"\b(high school|secondary|ged)\b",
]

_DEGREE_PATTERNS_TR = [
    r"\b(doktora|phd)\b",
    r"\b(yüksek lisans|master|msc|mba)\b",
    r"\b(lisans|bsc|ba|b Lisans)\b",
    r"\b(ön lisans|associate)\b",
    r"\b(lise|diploma)\b",
    r"\b(sertifika)\b",
]

# Field keywords
_FIELD_PATTERNS = {
    "computer_science": [
        r"\b(computer science|computer engineering|software|informatics)\b",
        r"\b(bilgisayar)\b",
    ],
    "data_science": [
        r"\b(data science|data analytics|big data)\b",
        r"\b(veri bilimi|veri analizi)\b",
    ],
    "engineering": [
        r"\b(electrical|mechanical|civil|chemical)\s*engineering\b",
        r"\b(mühendislik)\b",
    ],
    "business": [
        r"\b(business|management|economics|finance|marketing)\b",
        r"\b(işletme|yönetim|ekonomi|finans|pazarlama)\b",
    ],
}


@dataclass
class EducationInfo:
    """Extracted education information from text."""
    level: Optional[int]  # 0-7 scale
    level_name: str
    field: Optional[str]
    gpa: Optional[float]
    institution: Optional[str]


def extract_education_info(text: str) -> EducationInfo:
    """Extract education information from CV/job text."""
    if not text:
        return EducationInfo(level=None, level_name="unknown", field=None, gpa=None, institution=None)
    
    text_lower = text.lower()
    
    # Find education level
    level = None
    level_name = "unknown"
    
    for pattern in _DEGREE_PATTERNS_EN + _DEGREE_PATTERNS_TR:
        match = re.search(pattern, text_lower)
        if match:
            degree = match.group(1).lower()
            for key, value in EDUCATION_LEVELS.items():
                if key in degree or degree in key:
                    level = value
                    level_name = key
                    break
            if level is not None:
                break
    
    # Find field
    field = None
    for field_name, patterns in _FIELD_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                field = field_name
                break
        if field:
            break
    
    # Extract GPA if present
    gpa = None
    gpa_patterns = [
        r"gpa[:\s]+(\d+\.?\d*)\s*/?\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)",
        r"ortalama[:\s]+(\d+\.?\d*)",
    ]
    for pattern in gpa_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                first = float(match.group(1))
                if match.lastindex and match.lastindex >= 2:
                    second = float(match.group(2))
                    if second > 0:
                        gpa = first / second
                else:
                    gpa = first
                break
            except (ValueError, IndexError):
                continue
    
    return EducationInfo(
        level=level,
        level_name=level_name,
        field=field,
        gpa=gpa,
        institution=None,  # Would need NER for institution extraction
    )


def compute_education_match_score(
    cv_education: EducationInfo,
    job_education: EducationInfo,
) -> float:
    """Compute education match score between CV and job requirement.
    
    Phase 2 Upgrade: Added education matching for better ranking.
    """
    if job_education.level is None:
        return 1.0  # No specific requirement
    
    if cv_education.level is None:
        return 0.3  # Unknown CV education
    
    # Level match (partial credit for lower levels)
    level_diff = cv_education.level - job_education.level
    
    if level_diff >= 0:
        # CV meets or exceeds requirement
        level_score = 1.0
    else:
        # CV below requirement
        level_score = max(0.0, 1.0 + level_diff * 0.3)
    
    # Field match bonus
    field_score = 1.0
    if job_education.field and cv_education.field:
        if cv_education.field == job_education.field:
            field_score = 1.3  # Bonus for exact field match
        elif _are_related_fields(cv_education.field, job_education.field):
            field_score = 1.1  # Partial bonus for related fields
    
    return min(1.0, level_score * field_score)


def _are_related_fields(field1: str, field2: str) -> bool:
    """Check if two fields are related."""
    related_groups = [
        {"computer_science", "data_science"},
        {"computer_science", "engineering"},
    ]
    
    for group in related_groups:
        if field1 in group and field2 in group:
            return True
    return False


def education_match_matrix(
    cv_educations: list[EducationInfo],
    job_educations: list[EducationInfo],
) -> list[list[float]]:
    """Compute education match matrix for all CV-job pairs."""
    import numpy as np
    
    n_cvs = len(cv_educations)
    n_jobs = len(job_educations)
    
    matrix = np.zeros((n_cvs, n_jobs), dtype=np.float64)
    
    for i, cv_edu in enumerate(cv_educations):
        for j, job_edu in enumerate(job_educations):
            matrix[i, j] = compute_education_match_score(cv_edu, job_edu)
    
    return matrix.tolist()