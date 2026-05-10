from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.id_normalization import normalize_cv_id, normalize_job_id

DEFAULT_FEATURE_COLS: list[str] = [
    "tfidf_score",
    "semantic_score",
    "bm25_score",
    "skill_score",
    "experience_score",
    "must_have_coverage",
]


def _normalize_relevance(v: object) -> float:
    iv = int(v)
    mapping = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
    if iv not in mapping:
        raise ValueError("relevance must be in {0,1,2,3}")
    return mapping[iv]


def _normalize_gt_df(ground_truth_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    gt = ground_truth_df.copy()
    if "cv_id" not in gt.columns and "resume_id" in gt.columns:
        gt = gt.rename(columns={"resume_id": "cv_id"})
    if target_col not in gt.columns and "relevant" in gt.columns:
        gt = gt.rename(columns={"relevant": target_col})
    if target_col not in gt.columns and "relevance" in gt.columns:
        gt = gt.rename(columns={"relevance": target_col})
    required = {"job_id", "cv_id", target_col}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"ground_truth_df missing columns: {sorted(missing)}")
    gt = gt[list(required)].copy()
    gt["job_id"] = gt["job_id"].map(normalize_job_id)
    gt["cv_id"] = gt["cv_id"].map(normalize_cv_id)
    gt = gt[(gt["job_id"] != "") & (gt["cv_id"] != "")]
    gt[target_col] = gt[target_col].map(_normalize_relevance)
    gt = gt.groupby(["job_id", "cv_id"], as_index=False, sort=False)[target_col].max()
    return gt


def prepare_training_data(
    scores_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "relevance",
) -> pd.DataFrame:
    feats = feature_cols or list(DEFAULT_FEATURE_COLS)
    missing_feats = [c for c in feats if c not in scores_df.columns]
    if missing_feats:
        raise ValueError(f"scores_df missing feature columns: {missing_feats}")

    score = scores_df.copy()
    score["job_id"] = score["job_id"].map(normalize_job_id)
    score["cv_id"] = score["cv_id"].map(normalize_cv_id)
    score = score[(score["job_id"] != "") & (score["cv_id"] != "")]
    keep = ["job_id", "cv_id", *feats]
    score = score[keep]
    for col in feats:
        score[col] = pd.to_numeric(score[col], errors="coerce").fillna(0.0)

    gt = _normalize_gt_df(ground_truth_df, target_col=target_col)
    merged = gt.merge(score, on=["job_id", "cv_id"], how="inner")
    merged = merged.dropna(subset=[target_col])
    return merged


def train_learned_fusion(
    scores_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "relevance",
    *,
    epochs: int = 100,
    lr: float = 0.01,
):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for learned fusion training.") from exc

    train_df = prepare_training_data(
        scores_df,
        ground_truth_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )
    if train_df.empty:
        raise ValueError("No overlapping ground-truth rows found for learned fusion training.")

    X = torch.tensor(train_df[feature_cols].to_numpy(dtype="float32"), dtype=torch.float32)
    y = torch.tensor(train_df[target_col].to_numpy(dtype="float32"), dtype=torch.float32).unsqueeze(1)

    class LearnedFusionLinear(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.linear(x))

    model = LearnedFusionLinear(input_dim=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    return model


def predict_learned_fusion(
    scores_df: pd.DataFrame,
    model,
    feature_cols: list[str] | None = None,
) -> pd.Series:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for learned fusion inference.") from exc

    feats = feature_cols or list(DEFAULT_FEATURE_COLS)
    data = scores_df.copy()
    for col in feats:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    X = torch.tensor(data[feats].to_numpy(dtype="float32"), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model(X).detach().cpu().numpy().reshape(-1)
    return pd.Series(pred, index=scores_df.index, name="learned_fusion_score").clip(0.0, 1.0)


def save_learned_fusion_model(model, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    in_features = int(model.linear.in_features)
    payload = {"input_dim": in_features, "state_dict": model.state_dict()}
    torch.save(payload, path)


def load_learned_fusion_model(path: Path):
    import torch
    import torch.nn as nn

    class LearnedFusionLinear(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.linear(x))

    payload = torch.load(path, map_location="cpu")
    model = LearnedFusionLinear(int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def export_learned_fusion_weights_json(model, feature_cols: list[str], path: Path) -> dict[str, float]:
    weights = model.linear.weight.detach().cpu().numpy().reshape(-1).tolist()
    bias = float(model.linear.bias.detach().cpu().numpy().reshape(-1)[0])
    payload = {name: float(w) for name, w in zip(feature_cols, weights, strict=True)}
    payload["bias"] = bias
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload

