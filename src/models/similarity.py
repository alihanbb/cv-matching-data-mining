from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity


def cosine_pairs(cv_matrix, job_matrix):
    """Rows: CVs, columns: jobs — each cell is similarity(cv_i, job_j)."""
    return cosine_similarity(cv_matrix, job_matrix, dense_output=True)
