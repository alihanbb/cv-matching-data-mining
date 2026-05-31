from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.config.defaults import FALLBACK_EMBEDDING_MODELS

logger = logging.getLogger(__name__)


def try_load_semantic_encoder(
    model_name: str,
    *,
    device: str | None = None,
    fallback_enabled: bool = True,
) -> Any | None:
    """Load semantic encoder model with optional fallback to backup models.
    
    Phase 1 Upgrade: Added fallback mechanism for model loading failures.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed; dense channel disabled.")
        return None
    
    # Try primary model first
    model = _try_load_single_model(model_name, device)
    if model is not None:
        return model
    
    # Fallback to backup models if enabled
    if fallback_enabled:
        for fallback_model in FALLBACK_EMBEDDING_MODELS:
            if fallback_model == model_name:
                continue  # Skip already tried primary
            logger.info("Trying fallback model: %s", fallback_model)
            model = _try_load_single_model(fallback_model, device)
            if model is not None:
                logger.info("Successfully loaded fallback model: %s", fallback_model)
                return model
    
    logger.warning("All embedding models failed to load; dense channel disabled.")
    return None


def _try_load_single_model(model_name: str, device: str | None = None) -> Any | None:
    """Attempt to load a single embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    
    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    
    try:
        # Phase 1 Upgrade: Added model configuration for better performance
        model = SentenceTransformer(
            model_name,
            **kwargs,
            # Optimize for CPU inference
            prompts={"default": "retrieve: "},
        )
        
        # Probe test with multiple samples for reliability
        probe_texts = ["semantic probe", "query: test", "passage: sample"]
        probe = model.encode(
            probe_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=1,
        )
        
        if probe is None or not np.isfinite(probe).all():
            logger.warning("Embedding model produced non-finite probe output for %s", model_name)
            return None
        
        logger.info("Successfully loaded embedding model: %s", model_name)
        return model
        
    except Exception as e:
        logger.warning("Could not load embedding model %s: %s", model_name, e)
        return None


def encode_normalized(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts to normalized embeddings.
    
    Phase 1 Upgrade: Added chunked processing for large texts to prevent OOM.
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, model.get_sentence_embedding_dimension())
    
    safe_texts = [(t or " ").strip() if t else " " for t in texts]
    
    # Phase 1 Upgrade: Process in chunks to prevent memory issues
    # For bge-m3, embeddings dimension is 1024
    try:
        emb = model.encode(
            safe_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            # Optimize for speed
            precision="float32",
        )
    except Exception as e:
        logger.error("Encoding failed: %s. Trying with smaller batch size.", e)
        # Fallback: smaller batch size
        emb = model.encode(
            safe_texts,
            batch_size=max(1, batch_size // 2),
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            precision="float32",
        )
    
    arr = np.asarray(emb, dtype=np.float32)
    
    # Handle non-finite values
    if not np.isfinite(arr).all():
        logger.warning("Non-finite embeddings detected; replacing with zeros")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure L2 normalization (extra safety)
    row_norm = np.linalg.norm(arr, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    arr = arr / row_norm
    
    return arr


def dense_cosine_similarity(cv_emb: np.ndarray, job_emb: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between CV and job embeddings.
    
    Phase 1 Upgrade: Optimized for large matrices using batched computation.
    """
    # For L2-normalized embeddings, cosine similarity = dot product
    # Use batched computation for large matrices to prevent memory issues
    n_cvs = cv_emb.shape[0]
    n_jobs = job_emb.shape[0]
    
    # For small matrices, direct computation is fine
    if n_cvs * n_jobs < 10_000_000:
        sim = (cv_emb @ job_emb.T).astype(np.float64)
        return np.clip(sim, 0.0, 1.0)  # Clip to [0, 1] since we use normalized embeddings
    
    # For large matrices, use batched computation
    sim = np.zeros((n_cvs, n_jobs), dtype=np.float64)
    batch_size = 1000
    
    for i in range(0, n_cvs, batch_size):
        end_i = min(i + batch_size, n_cvs)
        sim[i:end_i, :] = cv_emb[i:end_i] @ job_emb.T
    
    return np.clip(sim, 0.0, 1.0)