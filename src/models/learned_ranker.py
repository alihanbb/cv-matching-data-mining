"""XGBoost/LightGBM based learned ranking for CV-Job matching.

Phase 3 Upgrade: Added gradient boosting based ranking model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Extended feature columns for better ranking
EXTENDED_FEATURE_COLS = [
    "tfidf_score",
    "semantic_score",
    "bm25_score",
    "skill_score",
    "experience_score",
    "must_have_coverage",
    "nice_to_have_coverage",
    "skill_jaccard_score",
    "cv_quality_score",
    # Phase 2 features (if available)
    "education_score",
    "certification_score",
]


def _normalize_relevance(v: object) -> float:
    """Convert relevance grades (0-3) to normalized scores (0.0-1.0)."""
    iv = int(v)
    mapping = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
    if iv not in mapping:
        raise ValueError("relevance must be in {0,1,2,3}")
    return mapping[iv]


def prepare_training_data_xgb(
    scores_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "relevant",
) -> pd.DataFrame:
    """Prepare training data for XGBoost ranking model."""
    feats = feature_cols or list(EXTENDED_FEATURE_COLS)
    
    # Filter available features
    available_feats = [f for f in feats if f in scores_df.columns]
    if len(available_feats) < 3:
        logger.warning("Not enough features for XGBoost training, falling back to defaults")
        available_feats = [
            "tfidf_score", "semantic_score", "skill_score", 
            "experience_score", "must_have_coverage"
        ]
        available_feats = [f for f in available_feats if f in scores_df.columns]
    
    score = scores_df.copy()
    score["job_id"] = score["job_id"].astype(str)
    score["cv_id"] = score["cv_id"].astype(str)
    
    # Clean feature columns
    for col in available_feats:
        if col in score.columns:
            score[col] = pd.to_numeric(score[col], errors="coerce").fillna(0.0)
    
    # Normalize ground truth
    gt = ground_truth_df.copy()
    if "cv_id" not in gt.columns and "resume_id" in gt.columns:
        gt = gt.rename(columns={"resume_id": "cv_id"})
    if target_col not in gt.columns:
        for alt in ["relevance", "score"]:
            if alt in gt.columns:
                target_col = alt
                break
    
    gt["job_id"] = gt["job_id"].astype(str)
    gt["cv_id"] = gt["cv_id"].astype(str)
    if target_col in gt.columns:
        gt[target_col] = gt[target_col].map(_normalize_relevance)
    
    # Merge
    merged = gt.merge(score, on=["job_id", "cv_id"], how="inner")
    merged = merged.dropna(subset=[target_col])
    
    return merged, available_feats


def train_xgboost_ranker(
    scores_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "relevant",
    *,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    objective: str = "rank:ndcg",
) -> Any:
    """Train XGBoost ranking model.
    
    Phase 3 Upgrade: Added XGBoost-based ranking.
    
    Args:
        scores_df: DataFrame with candidate scores
        ground_truth_df: Ground truth labels
        feature_cols: List of feature column names
        target_col: Target column name (e.g., 'relevant')
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        objective: XGBoost objective (rank:ndcg, rank:pairwise)
    
    Returns:
        Trained XGBoost model
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost required for XGBoost ranker") from e
    
    train_df, feats = prepare_training_data_xgb(
        scores_df, ground_truth_df, feature_cols, target_col
    )
    
    if train_df.empty:
        raise ValueError("No training data available")
    
    X = train_df[feats].values
    y = train_df[target_col].values
    
    # Create DMatrix with group information for ranking
    # Each job_id is a group
    train_df_sorted = train_df.sort_values("job_id")
    groups = train_df_sorted.groupby("job_id").size().values
    
    dtrain = xgb.DMatrix(X, label=y, feature_names=feats)
    dtrain.set_group(groups)
    
    params = {
        "objective": objective,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "eta": learning_rate,
        "seed": 42,
        "verbosity": 0,
    }
    
    # Train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        verbose_eval=False,
    )
    
    logger.info("XGBoost ranker trained with %d features, %d rounds", len(feats), n_estimators)
    return model, feats


def predict_xgboost_ranker(
    scores_df: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
) -> pd.Series:
    """Predict scores using trained XGBoost model."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost required")
    
    data = scores_df.copy()
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    
    X = data[feature_cols].values
    dtest = xgb.DMatrix(X, feature_names=feature_cols)
    
    predictions = model.predict(dtest)
    return pd.Series(predictions, index=scores_df.index, name="xgboost_score").clip(0.0, 1.0)


def train_lightgbm_ranker(
    scores_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "relevant",
    *,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
) -> Any:
    """Train LightGBM ranking model.
    
    Phase 3 Upgrade: Added LightGBM alternative.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm required for LightGBM ranker") from e
    
    train_df, feats = prepare_training_data_xgb(
        scores_df, ground_truth_df, feature_cols, target_col
    )
    
    if train_df.empty:
        raise ValueError("No training data available")
    
    # Prepare group data
    train_df_sorted = train_df.sort_values("job_id")
    groups = train_df_sorted.groupby("job_id").size().values
    
    X = train_df_sorted[feats].values
    y = train_df_sorted[target_col].values
    
    # Create dataset
    dataset = lgb.Dataset(X, label=y, feature_name=feats, group=groups)
    
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "verbosity": -1,
        "seed": 42,
    }
    
    model = lgb.train(
        params,
        dataset,
        num_boost_round=n_estimators,
    )
    
    logger.info("LightGBM ranker trained with %d features", len(feats))
    return model, feats


def predict_lightgbm_ranker(
    scores_df: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
) -> pd.Series:
    """Predict scores using trained LightGBM model."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm required")
    
    data = scores_df.copy()
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    
    X = data[feature_cols].values
    predictions = model.predict(X)
    return pd.Series(predictions, index=scores_df.index, name="lightgbm_score").clip(0.0, 1.0)


# Ensemble prediction
def ensemble_predict(
    scores_df: pd.DataFrame,
    models: list[tuple[Any, str]],  # (model, predict_func_name)
    feature_cols: list[str],
) -> pd.Series:
    """Combine predictions from multiple models.
    
    Phase 3 Upgrade: Model ensemble support.
    """
    predictions = []
    
    for model, func_name in models:
        if func_name == "xgboost":
            pred = predict_xgboost_ranker(scores_df, model, feature_cols)
        elif func_name == "lightgbm":
            pred = predict_lightgbm_ranker(scores_df, model, feature_cols)
        else:
            continue
        predictions.append(pred)
    
    if not predictions:
        return pd.Series(0.5, index=scores_df.index)
    
    # Average ensemble
    ensemble = pd.concat(predictions, axis=1).mean(axis=1)
    return ensemble


def save_model_xgb(model: Any, path: Path) -> None:
    """Save XGBoost model to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model_xgb(path: Path) -> Any:
    """Load XGBoost model from file."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost required")
    return xgb.Booster()


# Feature importance analysis
def get_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Get feature importance from trained model."""
    try:
        import xgboost as xgb
    except ImportError:
        return pd.DataFrame()
    
    if hasattr(model, "get_score"):
        scores = model.get_score(importance_type="gain")
        importance = [{"feature": f"f{i}", "importance": float(scores.get(f"f{i}", 0))} 
                      for i in range(len(feature_names))]
        for i, name in enumerate(feature_names):
            for imp in importance:
                if imp["feature"] == f"f{i}":
                    imp["feature"] = name
                    break
        return pd.DataFrame(importance).sort_values("importance", ascending=False)
    return pd.DataFrame()