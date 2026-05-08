from __future__ import annotations

import numpy as np

from src.preprocessing.tokenizer import tokenize_unicode
from src.scoring.fusion import minmax_per_column


def bm25_matrix(job_queries: list[str], cv_docs: list[str]) -> np.ndarray:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:  # pragma: no cover
        raise ImportError("rank-bm25 is required for BM25 scoring. Install optional extra 'bm25'.") from e

    tokenized_cvs = [tokenize_unicode(d) for d in cv_docs]
    # BM25Okapi requires non-empty doc token lists; placeholder avoids all-zero columns.
    tokenized_cvs = [t if t else ["_empty_"] for t in tokenized_cvs]
    bm25 = BM25Okapi(tokenized_cvs)
    n_c, n_j = len(cv_docs), len(job_queries)
    raw = np.zeros((n_c, n_j), dtype=np.float64)
    for j, q in enumerate(job_queries):
        q_tokens = tokenize_unicode(q)
        if not q_tokens:
            # rare: job text became empty after token rules — match on placeholder corpus
            q_tokens = ["job"]
        scores = bm25.get_scores(q_tokens)
        raw[:, j] = np.asarray(scores, dtype=np.float64)
    return minmax_per_column(raw)
