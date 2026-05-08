import numpy as np

from src.features.semantic_encoder import dense_cosine_similarity


def test_dense_cosine_in_unit_range() -> None:
    rng = np.random.default_rng(0)
    cv = rng.normal(size=(4, 32)).astype(np.float32)
    cv /= np.linalg.norm(cv, axis=1, keepdims=True)
    job = rng.normal(size=(3, 32)).astype(np.float32)
    job /= np.linalg.norm(job, axis=1, keepdims=True)
    sim = dense_cosine_similarity(cv, job)
    assert sim.shape == (4, 3)
    assert sim.min() >= -1.0 - 1e-6
    assert sim.max() <= 1.0 + 1e-6
