"""Pydantic config schema for the CV-matching pipeline.

All top-level sections mirror config/config.yaml exactly.  Unknown keys are
*forbidden* so typos in YAML are caught at startup rather than silently ignored.

Usage (in main.py or any entry-point)::

    from src.config.schema import PipelineConfig
    from src.utils.helpers import load_config

    raw = load_config(args.config)
    cfg = PipelineConfig.model_validate(raw)   # raises ValidationError on bad input
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class PathsConfig(BaseModel):
    model_config = {"extra": "forbid"}

    processed_cvs: str
    processed_jobs: str
    tfidf_model: str
    output_rankings: str
    output_explanations: str = "data/gold/rankings/candidate_scores_explained.csv"
    ground_truth: str | None = None


class SkillsConfig(BaseModel):
    model_config = {"extra": "forbid"}

    path: str = "config/skills.yaml"


class IngestCvCorpusJsonlConfig(BaseModel):
    model_config = {"extra": "forbid"}

    enabled: bool = False
    path: str = ""
    id_field: str = "record_id"
    text_field: str = "text"
    id_prefix: str = "corpus_"
    max_rows: int = Field(default=4000, ge=1)


class IngestConfig(BaseModel):
    model_config = {"extra": "forbid"}

    raw_cvs_dir: str = "data/bronze/cvs"
    raw_jobs_dir: str = "data/bronze/job_descriptions"
    cv_corpus_jsonl: IngestCvCorpusJsonlConfig = Field(default_factory=IngestCvCorpusJsonlConfig)


class SilverConfig(BaseModel):
    model_config = {"extra": "forbid"}

    unified_resumes: str = "data/silver/unified_resumes.jsonl"
    write_unified_resumes: bool = False


class PrivacyConfig(BaseModel):
    model_config = {"extra": "forbid"}

    anonymize: bool = True


class PreprocessingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    language: str = "en"
    remove_stopwords: bool = True
    lemmatize: bool = True


class TfidfConfig(BaseModel):
    model_config = {"extra": "forbid"}

    max_features: int = Field(default=5000, ge=100)
    ngram_range: list[int] = Field(default=[1, 2])
    min_df: int = Field(default=1, ge=1)
    max_df: float = Field(default=0.95, gt=0.0, le=1.0)
    sublinear_tf: bool = True

    @field_validator("ngram_range")
    @classmethod
    def _check_ngram(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or v[0] < 1 or v[1] < v[0]:
            raise ValueError("ngram_range must be [min, max] with 1 <= min <= max")
        return v


class EmbeddingsConfig(BaseModel):
    model_config = {"extra": "forbid"}

    enabled: bool = True
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = Field(default=32, ge=1)
    device: str | None = None


class Bm25Config(BaseModel):
    model_config = {"extra": "forbid"}

    enabled: bool = False


class FusionWeightsConfig(BaseModel):
    """V1 fusion weights — no BM25 channel."""
    model_config = {"extra": "allow"}  # allow extra channel keys set by weight optimiser

    tfidf: float = Field(default=0.35, ge=0.0, le=1.0)
    dense: float = Field(default=0.35, ge=0.0, le=1.0)
    skills: float = Field(default=0.20, ge=0.0, le=1.0)
    experience: float = Field(default=0.10, ge=0.0, le=1.0)


class FusionConfig(BaseModel):
    model_config = {"extra": "forbid"}

    weights: FusionWeightsConfig = Field(default_factory=FusionWeightsConfig)


class FusionV2WeightsConfig(BaseModel):
    """V2 fusion weights — includes BM25 channel."""
    model_config = {"extra": "allow"}

    tfidf: float = Field(default=0.25, ge=0.0, le=1.0)
    dense: float = Field(default=0.25, ge=0.0, le=1.0)
    bm25: float = Field(default=0.20, ge=0.0, le=1.0)
    skills: float = Field(default=0.20, ge=0.0, le=1.0)
    experience: float = Field(default=0.10, ge=0.0, le=1.0)


class FusionV2Config(BaseModel):
    model_config = {"extra": "forbid"}

    weights: FusionV2WeightsConfig = Field(default_factory=FusionV2WeightsConfig)


class MatchingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    top_k: int = Field(default=10, ge=1)


class EvaluationConfig(BaseModel):
    model_config = {"extra": "forbid"}

    top_k_values: list[int] = Field(default=[1, 3, 5])

    @field_validator("top_k_values")
    @classmethod
    def _check_ks(cls, v: list[int]) -> list[int]:
        if not v or any(k < 1 for k in v):
            raise ValueError("top_k_values must be a non-empty list of positive integers")
        return v


class PipelineRunConfig(BaseModel):
    model_config = {"extra": "forbid"}

    write_explanations: bool = True


class ExperimentConfig(BaseModel):
    model_config = {"extra": "forbid"}

    write_manifest: bool = True


class LoggingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def _check_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"logging.level must be one of {valid}, got '{v}'")
        return v.upper()


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    model_config = {"extra": "forbid"}

    paths: PathsConfig
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    silver: SilverConfig = Field(default_factory=SilverConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    tfidf: TfidfConfig = Field(default_factory=TfidfConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    bm25: Bm25Config = Field(default_factory=Bm25Config)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    fusion_v2: FusionV2Config = Field(default_factory=FusionV2Config)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    pipeline: PipelineRunConfig = Field(default_factory=PipelineRunConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _warn_anonymize_off(self) -> "PipelineConfig":
        import logging as _logging
        if not self.privacy.anonymize:
            _logging.getLogger(__name__).warning(
                "CONFIG: privacy.anonymize=false — PII will NOT be redacted. "
                "Ensure compliance with GDPR/KVKK before running in production."
            )
        return self
