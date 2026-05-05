from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def try_load_semantic_encoder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed; dense channel disabled.")
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as e:  # pragma: no cover
        logger.warning("Could not load embedding model %s: %s", model_name, e)
        return None


def encode_normalized(model, texts: list[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def dense_cosine_similarity(cv_emb: np.ndarray, job_emb: np.ndarray) -> np.ndarray:
    """Embeddings L2-normalized → cosine similarity is dot product."""
    return (cv_emb @ job_emb.T).astype(np.float64)
