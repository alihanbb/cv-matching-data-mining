"""Standalone CV – Job Description Matching API.

Tek bir embedding kanalına (BAAI/bge-m3) dayanan, bağımsız çalışabilen
hafif bir REST API'dir. Tam pipeline (TF-IDF + BM25 + skill extraction +
fusion) için ``api/main.py``'yi (``uvicorn api.main:app``) kullanın.

Başlatmak için:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from functools import lru_cache
from typing import List, Optional
import re
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# src/ altyapısını kullan (projenin kökü sys.path'te değilse ekle)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.preprocessing.cleaner import TextCleaner
    from src.preprocessing.pii import anonymize_text
    from src.config.defaults import SEMANTIC_LOW_THRESHOLD

    _cleaner = TextCleaner(remove_stopwords=False, lemmatize=False)
    _USE_SRC = True
except ImportError:
    # Bağımsız kurulumda src/ yoksa basit fallback kullan
    _USE_SRC = False
    SEMANTIC_LOW_THRESHOLD = 0.35  # type: ignore[assignment]


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 8
CHUNK_WORD_SIZE = 350
CHUNK_OVERLAP = 70


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="CV - Job Description Matching API",
    description=(
        "JSON olarak gelen iş tanımı ve CV metnini embedding ile karşılaştırır "
        "ve uygunluk skorunu JSON olarak döner. "
        "Tam pipeline için api/main.py'yi kullanın."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================


class TextDocument(BaseModel):
    id: Optional[str] = Field(default=None, description="Doküman ID veya dosya adı")
    text: str = Field(..., min_length=1, description="CV veya iş tanımı metni")


class MatchRequest(BaseModel):
    job_description: TextDocument
    cv: TextDocument


class MatchResponse(BaseModel):
    job_description_id: Optional[str]
    cv_id: Optional[str]
    match_score: float
    match_percentage: float
    interpretation: str


class BatchMatchRequest(BaseModel):
    job_description: TextDocument
    cvs: List[TextDocument] = Field(..., min_length=1)
    top_n: Optional[int] = Field(default=None, ge=1, description="İstenirse en iyi N CV döndürülür")


class BatchMatchItem(BaseModel):
    rank: int
    cv_id: Optional[str]
    match_score: float
    match_percentage: float
    interpretation: str


class BatchMatchResponse(BaseModel):
    job_description_id: Optional[str]
    results: List[BatchMatchItem]


# ============================================================
# MODEL LOADING
# ============================================================


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Model sadece bir kez yüklenir (LRU cache ile)."""
    return SentenceTransformer(MODEL_NAME)


# ============================================================
# TEXT CLEANING / CHUNKING
# ============================================================


def clean_text(text: str) -> str:
    """Null byte ve gereksiz boşlukları temizler; PII'ı maskeler."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    cleaned = text.strip()

    if _USE_SRC:
        # PII anonymize (e-posta, telefon, URL, adres)
        cleaned = anonymize_text(cleaned)

    return cleaned


def chunk_text(
    text: str,
    chunk_word_size: int = CHUNK_WORD_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_word_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_word_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================


def average_embeddings(embeddings: np.ndarray) -> np.ndarray:
    averaged = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(averaged)
    return averaged if norm == 0 else averaged / norm


def encode_long_text(
    model: SentenceTransformer,
    text: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError("Text is empty after cleaning")

    chunks = chunk_text(cleaned)

    if not chunks:
        raise ValueError("No valid chunks created from text")

    chunk_embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return average_embeddings(np.array(chunk_embeddings))


def calculate_match_score(job_text: str, cv_text: str) -> float:
    model = get_model()
    job_embedding = encode_long_text(model, job_text)
    cv_embedding = encode_long_text(model, cv_text)
    score = cosine_similarity(
        job_embedding.reshape(1, -1),
        cv_embedding.reshape(1, -1),
    )[
        0
    ][0]
    return round(float(score), 5)


def interpret_score(score: float) -> str:
    """Skor yorumlama — src/config/defaults.py SEMANTIC_LOW_THRESHOLD ile hizalı."""
    if score >= 0.80:
        return "Çok yüksek uyum"
    if score >= 0.65:
        return "Yüksek uyum"
    if score >= 0.50:
        return "Orta uyum"
    if score >= SEMANTIC_LOW_THRESHOLD:
        return "Düşük uyum"
    return "Çok düşük uyum"


# ============================================================
# API ENDPOINTS
# ============================================================


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "src_infrastructure": _USE_SRC,
        "note": "Tam pipeline için /api/* endpoint'lerini (api/main.py) kullanın.",
    }


@app.post("/match", response_model=MatchResponse)
def match_cv_with_job_description(payload: MatchRequest) -> MatchResponse:
    """Tek bir iş tanımı ile tek bir CV'yi karşılaştırır."""
    try:
        score = calculate_match_score(
            job_text=payload.job_description.text,
            cv_text=payload.cv.text,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return MatchResponse(
        job_description_id=payload.job_description.id,
        cv_id=payload.cv.id,
        match_score=score,
        match_percentage=round(score * 100, 2),
        interpretation=interpret_score(score),
    )


@app.post("/match-batch", response_model=BatchMatchResponse)
def match_multiple_cvs_with_job_description(payload: BatchMatchRequest) -> BatchMatchResponse:
    """
    Tek bir iş tanımı ile birden fazla CV'yi karşılaştırır.
    Sonuçları skora göre büyükten küçüğe sıralar.
    """
    model = get_model()

    try:
        job_embedding = encode_long_text(model, payload.job_description.text)

        rows = []
        for cv in payload.cvs:
            cv_embedding = encode_long_text(model, cv.text)
            score = cosine_similarity(
                job_embedding.reshape(1, -1),
                cv_embedding.reshape(1, -1),
            )[
                0
            ][0]
            score = round(float(score), 5)
            rows.append(
                {
                    "cv_id": cv.id,
                    "match_score": score,
                    "match_percentage": round(score * 100, 2),
                    "interpretation": interpret_score(score),
                }
            )

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    rows = sorted(rows, key=lambda item: item["match_score"], reverse=True)

    if payload.top_n is not None:
        rows = rows[: payload.top_n]

    results = [
        BatchMatchItem(
            rank=index,
            cv_id=row["cv_id"],
            match_score=row["match_score"],
            match_percentage=row["match_percentage"],
            interpretation=row["interpretation"],
        )
        for index, row in enumerate(rows, start=1)
    ]

    return BatchMatchResponse(
        job_description_id=payload.job_description.id,
        results=results,
    )
