from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def try_load_semantic_encoder(
    model_name: str, *, device: str | None = None
) -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed; dense channel disabled.")
        return None
    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    try:
        model = SentenceTransformer(model_name, **kwargs)
        probe = model.encode(
            ["semantic probe"], normalize_embeddings=True, show_progress_bar=False
        )
        if probe is None or not np.isfinite(probe).all():
            logger.warning(
                "Embedding model produced non-finite probe output for %s", model_name
            )
            return None
        return model
    except Exception as e:  # pragma: no cover - environment specific
        logger.warning("Could not load embedding model %s: %s", model_name, e)
        return None


def encode_normalized(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    safe_texts = [(t or " ").strip() if t else " " for t in texts]
    emb = model.encode(
        safe_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    arr = np.asarray(emb, dtype=np.float32)
    if not np.isfinite(arr).all():
        logger.warning("Non-finite embeddings detected; replacing with zeros")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    row_norm = np.linalg.norm(arr, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    arr = arr / row_norm
    return arr


def dense_cosine_similarity(cv_emb: np.ndarray, job_emb: np.ndarray) -> np.ndarray:
    """Embeddings L2-normalized → cosine similarity is dot product."""
    sim = (cv_emb @ job_emb.T).astype(np.float64)
    sim = np.clip(sim, -1.0, 1.0)
    return sim
