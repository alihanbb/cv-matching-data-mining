"""Certification extraction and matching for CV-Job matching.

Phase 2 Upgrade: Added certification matching.
"""

from __future__ import annotations

import re
from typing import Optional


# Common tech certifications
CERTIFICATION_PATTERNS = {
    # Cloud certifications
    "aws_certified": [
        r"\baws\s+(certified|solutions architect|developer|sysops)\b",
        r"\bAWS.*Certified\b",
    ],
    "azure_certified": [
        r"\bazure\s+(certified|administrator|developer|architect)\b",
        r"\bMicrosoft.*Certified\b",
    ],
    "gcp_certified": [
        r"\bgcp\s+(certified|professional|data engineer)\b",
        r"\bGoogle Cloud.*Certified\b",
    ],
    # DevOps certifications
    "kubernetes_certified": [
        r"\b(cka|ckad|cks|kubernetes)\s+certified\b",
    ],
    "docker_certified": [
        r"\bdocker\s+certified\b",
    ],
    # Data/ML certifications
    "tensorflow_certified": [
        r"\btensorflow\s+developer\s+certified\b",
    ],
    "aws_ml_certified": [
        r"\baws\s+(machine learning|sagemaker)\s+certified\b",
    ],
    # Project management
    "pmp": [
        r"\bpmp\s+certified\b",
        r"\bproject management professional\b",
    ],
    "scrum_master": [
        r"\b(csm|psm|scrum master)\b",
        r"\bcertified\s+scrum\s+master\b",
    ],
    # Database certifications
    "oracle_certified": [
        r"\boracle\s+certified\b",
    ],
    "mongodb_certified": [
        r"\bmongodb\s+(certified|developer|dba)\b",
    ],
    # Security certifications
    "cissp": [
        r"\bcissp\b",
    ],
    "security_plus": [
        r"\bsecurity\+\b",
    ],
    # Other common certs
    "cisco_certified": [
        r"\b(ccna|ccnp|ccie|cisco)\s+certified\b",
    ],
    "linux_certified": [
        r"\b(lpic|rhce|linux\+)\b",
    ],
    "itil_certified": [
        r"\bitil\s+certified\b",
    ],
}


def extract_certifications(text: str) -> set[str]:
    """Extract certifications from text.
    
    Phase 2 Upgrade: Added certification detection for better matching.
    """
    if not text:
        return set()
    
    text_lower = text.lower()
    found_certs = set()
    
    for cert_name, patterns in CERTIFICATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_certs.add(cert_name)
                break
    
    return found_certs


def compute_certification_match(
    cv_certs: set[str],
    job_certs: set[str],
) -> float:
    """Compute certification match score.
    
    Returns 1.0 if CV has all required certs, partial credit otherwise.
    """
    if not job_certs:
        return 1.0
    
    if not cv_certs:
        return 0.5  # Neutral - no certs but no requirement
    
    matched = cv_certs & job_certs
    coverage = len(matched) / len(job_certs)
    
    return coverage


def certification_match_matrix(
    cv_certifications: list[set[str]],
    job_certifications: list[set[str]],
) -> list[list[float]]:
    """Compute certification match matrix for all CV-job pairs."""
    import numpy as np
    
    n_cvs = len(cv_certifications)
    n_jobs = len(job_certifications)
    
    matrix = np.zeros((n_cvs, n_jobs), dtype=np.float64)
    
    for i, cv_certs in enumerate(cv_certifications):
        for j, job_certs in enumerate(job_certifications):
            matrix[i, j] = compute_certification_match(cv_certs, job_certs)
    
    return matrix.tolist()


# Phase 2 Upgrade: Certification priority mapping
CERTIFICATION_WEIGHTS = {
    # High priority (directly job-related)
    "aws_certified": 1.5,
    "azure_certified": 1.5,
    "gcp_certified": 1.5,
    "kubernetes_certified": 1.4,
    "tensorflow_certified": 1.3,
    # Medium priority
    "docker_certified": 1.2,
    "pmp": 1.2,
    "scrum_master": 1.1,
    "cisco_certified": 1.1,
    # Lower priority but still valuable
    "oracle_certified": 1.0,
    "mongodb_certified": 1.0,
    "linux_certified": 1.0,
    "cissp": 1.3,
    "security_plus": 1.2,
    "aws_ml_certified": 1.3,
    "itil_certified": 1.0,
}


def compute_weighted_cert_score(cv_certs: set[str]) -> float:
    """Compute weighted certification score based on cert value."""
    total_weight = 0.0
    
    for cert in cv_certs:
        weight = CERTIFICATION_WEIGHTS.get(cert, 1.0)
        total_weight += weight
    
    # Normalize to 0-1 range (max ~10 certs)
    return min(1.0, total_weight / 10.0)