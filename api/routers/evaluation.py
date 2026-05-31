"""Evaluation metrics endpoints."""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "gold" / "evaluation" / "evaluation_results.csv"
MODEL_COMP_PATH = PROJECT_ROOT / "data" / "gold" / "evaluation" / "model_comparison.csv"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "evaluation" / "ground_truth_filtered.csv"


def load_eval_results() -> pd.DataFrame:
    """Load evaluation results."""
    if not EVAL_RESULTS_PATH.is_file():
        return pd.DataFrame()
    return pd.read_csv(EVAL_RESULTS_PATH)


def load_model_comparison() -> pd.DataFrame:
    """Load model comparison data."""
    if not MODEL_COMP_PATH.is_file():
        return pd.DataFrame()
    return pd.read_csv(MODEL_COMP_PATH)


@router.get("/evaluation/metrics")
async def get_evaluation_metrics():
    """Get evaluation metrics (NDCG, Precision, Recall, MRR)."""
    df = load_eval_results()
    if df.empty:
        return {"metrics": [], "message": "No evaluation results found"}

    metrics = []
    for _, row in df.iterrows():
        metrics.append({
            "model": row["model"],
            "precision_at_1": float(row.get("precision_at_1", 0)),
            "precision_at_3": float(row.get("precision_at_3", 0)),
            "precision_at_5": float(row.get("precision_at_5", 0)),
            "recall_at_1": float(row.get("recall_at_1", 0)),
            "recall_at_3": float(row.get("recall_at_3", 0)),
            "recall_at_5": float(row.get("recall_at_5", 0)),
            "ndcg_at_1": float(row.get("ndcg_at_1", 0)),
            "ndcg_at_3": float(row.get("ndcg_at_3", 0)),
            "ndcg_at_5": float(row.get("ndcg_at_5", 0)),
            "mrr": float(row.get("mrr", 0)),
            "map": float(row.get("map", 0)),
        })

    return {"metrics": metrics}


@router.get("/evaluation/comparison")
async def get_model_comparison():
    """Get model comparison data."""
    df = load_model_comparison()
    if df.empty:
        return {"comparison": [], "message": "No model comparison found"}

    return {"comparison": df.to_dict(orient="records")}


@router.get("/evaluation/ground-truth")
async def get_ground_truth_stats():
    """Get ground truth statistics."""
    if not GROUND_TRUTH_PATH.is_file():
        return {"stats": None, "message": "No ground truth found"}

    df = pd.read_csv(GROUND_TRUTH_PATH)
    return {
        "total_rows": len(df),
        "unique_jobs": df["job_id"].nunique(),
        "unique_cvs": df["cv_id"].nunique(),
        "relevance_distribution": df["relevance"].value_counts().to_dict(),
    }


@router.get("/evaluation/summary")
async def get_evaluation_summary():
    """Get a summary of evaluation results."""
    df = load_eval_results()
    if df.empty:
        return {"summary": None, "message": "No evaluation results"}

    best_model = df.loc[df["ndcg_at_5"].idxmax()] if "ndcg_at_5" in df.columns else df.iloc[0]

    return {
        "best_model": best_model["model"] if "model" in best_model else "N/A",
        "best_ndcg5": float(best_model.get("ndcg_at_5", 0)),
        "best_mrr": float(best_model.get("mrr", 0)),
        "total_models_evaluated": len(df),
    }