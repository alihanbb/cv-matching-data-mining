"""Optional dependency management.

Centralised guards for libraries that are listed as optional extras in
``pyproject.toml``.  Every module that needs an optional package should
call the corresponding ``require_*`` function instead of wrapping its own
try/except ImportError block.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def require_sentence_transformers():
    """Return the ``sentence_transformers`` module or raise an informative error."""
    try:
        import sentence_transformers  # noqa: PLC0415
        return sentence_transformers
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for the dense/semantic channel.\n"
            "Install it with:  pip install -e '.[semantic]'"
        ) from exc


def require_rank_bm25():
    """Return the ``rank_bm25`` module or raise an informative error."""
    try:
        import rank_bm25  # noqa: PLC0415
        return rank_bm25
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 is required for the BM25 channel.\n"
            "Install it with:  pip install -e '.[bm25]'"
        ) from exc


def require_torch():
    """Return the ``torch`` module or raise an informative error."""
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for learned fusion training.\n"
            "Install it with:  pip install -e '.[training]'"
        ) from exc


def try_import_sentence_transformers():
    """Return ``sentence_transformers`` module, or ``None`` if unavailable."""
    try:
        import sentence_transformers  # noqa: PLC0415
        return sentence_transformers
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — dense/semantic channel disabled. "
            "Install with: pip install -e '.[semantic]'"
        )
        return None


def try_import_rank_bm25():
    """Return ``rank_bm25`` module, or ``None`` if unavailable."""
    try:
        import rank_bm25  # noqa: PLC0415
        return rank_bm25
    except ImportError:
        logger.warning(
            "rank-bm25 not installed — BM25 channel disabled. "
            "Install with: pip install -e '.[bm25]'"
        )
        return None
