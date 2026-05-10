"""Default constants for the CV-matching pipeline.

All numeric thresholds and weight defaults live here so they are never
scattered as magic numbers across multiple modules.  Config YAML values
override these at runtime.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fusion weight defaults
# ---------------------------------------------------------------------------

#: V1 — no BM25 channel (TF-IDF + SBERT + skills + experience)
FUSION_V1_WEIGHTS: dict[str, float] = {
    "tfidf": 0.35,
    "dense": 0.35,
    "skills": 0.20,
    "experience": 0.10,
}

#: V2 — includes BM25 channel (TF-IDF + SBERT + BM25 + skills + experience)
FUSION_V2_WEIGHTS: dict[str, float] = {
    "tfidf": 0.25,
    "dense": 0.25,
    "bm25": 0.20,
    "skills": 0.20,
    "experience": 0.10,
}

# ---------------------------------------------------------------------------
# Skill score computation
# ---------------------------------------------------------------------------

#: Weight for must-have skill coverage in the combined skill score.
SKILL_SCORE_MUST_WEIGHT: float = 0.75

#: Weight for nice-to-have skill coverage in the combined skill score.
SKILL_SCORE_NICE_WEIGHT: float = 0.25

# ---------------------------------------------------------------------------
# Explanation / suggestion thresholds
# ---------------------------------------------------------------------------

#: Semantic similarity below this value triggers a wording-improvement hint.
SEMANTIC_LOW_THRESHOLD: float = 0.35

#: Score difference above this value triggers a "CHECK>0.01" audit warning.
SCORE_AUDIT_WARN_THRESHOLD: float = 0.01

# ---------------------------------------------------------------------------
# CV quality score section weights
# ---------------------------------------------------------------------------

CV_QUALITY_WEIGHTS: dict[str, float] = {
    "skills": 0.20,
    "experience": 0.25,
    "education": 0.15,
    "projects": 0.10,
    "certificates": 0.10,
    "measurable": 0.20,
}

#: Minimum number of measurable achievement hits to score full marks.
CV_QUALITY_MEASURABLE_DIVISOR: float = 3.0

# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

#: Default blend: (1 - CROSS_ENCODER_BLEND) * base + CROSS_ENCODER_BLEND * ce_score
CROSS_ENCODER_BLEND: float = 0.3

#: Default cross-encoder model name.
CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Embedding model defaults
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_BATCH_SIZE: int = 32

# ---------------------------------------------------------------------------
# Weight optimiser
# ---------------------------------------------------------------------------

#: Number of random weight candidates to evaluate during weight search.
WEIGHT_SEARCH_N_TRIALS: int = 150
