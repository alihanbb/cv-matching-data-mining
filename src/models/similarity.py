"""Thin compatibility shim — use sklearn directly in new code.

.. deprecated::
    Import ``cosine_similarity`` directly from
    ``sklearn.metrics.pairwise`` instead of using this wrapper.
"""

from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity


def cosine_pairs(cv_matrix, job_matrix):
    """Compute pairwise cosine similarity between CV and job matrices.

    .. deprecated::
        Use ``sklearn.metrics.pairwise.cosine_similarity`` directly.

    Rows: CVs, columns: jobs — each cell is similarity(cv_i, job_j).
    """
    return cosine_similarity(cv_matrix, job_matrix, dense_output=True)
