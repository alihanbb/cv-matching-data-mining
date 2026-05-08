"""Shared I/O utilities for the pipeline.

Single source of truth for reading processed CSV files with validation.
Previously duplicated between orchestrator.py and matching_inputs.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_processed_csv(path: Path, id_col: str) -> pd.DataFrame:
    """Read a processed Silver CSV file, raising descriptive errors on failure.

    Parameters
    ----------
    path:
        Full path to the CSV file.
    id_col:
        Name of the required ID column (e.g. ``"cv_id"`` or ``"job_id"``).

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame (may still be empty — caller should check).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or missing the required ID column.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed table not found: {path}\n"
            "Run the ingest step first: python main.py --ingest"
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Processed table is empty (no header): {path}\n"
            "Place input files under data/bronze/, then re-run with --ingest."
        ) from None
    if id_col not in df.columns:
        raise ValueError(
            f"Processed table missing column '{id_col}': {path}\n"
            "Re-run ingest with valid bronze data."
        )
    logger.debug("Loaded %d rows from %s", len(df), path)
    return df
