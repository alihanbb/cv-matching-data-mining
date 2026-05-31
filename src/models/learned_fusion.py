from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.id_normalization import normalize_cv_id, normalize_job_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model definition — module-level so torch.save / torch.load can resolve the
# class by its fully-qualified name regardless of call site.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    class LearnedFusionLinear(nn.Module):
        """Single-layer linear model for learned score fusion."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.linear(x))

except ImportError:
    LearnedFusionLinear = None  # type: ignore[assignment,misc]

DEFAULT_FEATURE_COLS: list[str] = [
    "tfidf_score",
    "semantic_score",
    "bm25_score",
    "skill_score",
    "experience_score",
    "must_have_coverage",
]

MIN_ALIGNMENT_RATIO = 0.5


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


def _alignment_stats(gt_df: pd.DataFrame, score_df: pd.DataFrame) -> dict[str, int | float]:
    pairs = score_df[["job_id", "cv_id"]].drop_duplicates()
    merged = gt_df[["job_id", "cv_id"]].merge(pairs, on=["job_id", "cv_id"], how="left", indicator=True)
    total_gt = int(len(gt_df))
    matched_gt = int((merged["_merge"] == "both").sum())
    unmatched_gt = total_gt - matched_gt
    match_ratio = (matched_gt / total_gt) if total_gt else 0.0
    return {
        "total_ground_truth_rows": total_gt,
        "matched_ground_truth_rows": matched_gt,
        "unmatched_ground_truth_rows": unmatched_gt,
        "match_ratio": float(match_ratio),
    }


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
    stats = _alignment_stats(gt, score)
    total_gt = int(stats["total_ground_truth_rows"])
    matched_gt = int(stats["matched_ground_truth_rows"])
    unmatched_gt = int(stats["unmatched_ground_truth_rows"])
    match_ratio = float(stats["match_ratio"])
    logger.info(
        "Learned fusion training alignment: total_ground_truth_rows=%d matched_ground_truth_rows=%d "
        "unmatched_ground_truth_rows=%d match_ratio=%.4f",
        total_gt,
        matched_gt,
        unmatched_gt,
        match_ratio,
    )
    if total_gt > 0 and match_ratio < MIN_ALIGNMENT_RATIO:
        logger.warning("Learned fusion training skipped: ground truth alignment too low.")
        raise ValueError("ground truth alignment too low.")

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
    val_ratio: float = 0.2,
    patience: int = 10,
    weight_decay: float = 1e-4,
):
    """Train a linear fusion model with train/val split and early stopping.

    Args:
        scores_df: DataFrame with feature columns for each (job_id, cv_id) pair.
        ground_truth_df: Ground truth relevance labels.
        feature_cols: Feature column names to use as inputs.
        target_col: Name of the relevance label column.
        epochs: Maximum number of training epochs.
        lr: Adam learning rate.
        val_ratio: Fraction of data to use for validation (early stopping).
        patience: Stop training if val loss does not improve for this many epochs.
        weight_decay: L2 regularization coefficient for Adam optimizer.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for learned fusion training.") from exc

    if LearnedFusionLinear is None:  # pragma: no cover
        raise ImportError("PyTorch is required for learned fusion training.")

    train_df = prepare_training_data(
        scores_df,
        ground_truth_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )
    if train_df.empty:
        raise ValueError("No overlapping ground-truth rows found for learned fusion training.")

    # --- Train / validation split ---
    n_total = len(train_df)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    shuffled = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_part = shuffled.iloc[:n_train]
    val_part = shuffled.iloc[n_train:]

    X_train = torch.tensor(train_part[feature_cols].to_numpy(dtype="float32"), dtype=torch.float32)
    y_train = torch.tensor(train_part[target_col].to_numpy(dtype="float32"), dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(val_part[feature_cols].to_numpy(dtype="float32"), dtype=torch.float32)
    y_val = torch.tensor(val_part[target_col].to_numpy(dtype="float32"), dtype=torch.float32).unsqueeze(1)

    model = LearnedFusionLinear(input_dim=len(feature_cols))
    # weight_decay provides L2 regularization to prevent overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict = {}
    no_improve = 0
    n_epochs = int(epochs)

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        # Validation pass (no gradient)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
        model.train()

        val_loss_val = float(val_loss.item())
        if epoch == 1 or epoch == n_epochs or (epoch % 10 == 0):
            logger.info(
                "Learned fusion epoch %d/%d train_loss=%.6f val_loss=%.6f",
                epoch,
                n_epochs,
                float(loss.item()),
                val_loss_val,
            )

        # Early stopping
        if val_loss_val < best_val_loss - 1e-6:
            best_val_loss = val_loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (val_loss=%.6f)", epoch, best_val_loss)
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
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
    # Save only primitive types + tensors — never pickle arbitrary objects.
    # Use weights_only-compatible payload: plain dict of tensors and scalars.
    payload = {
        "input_dim": in_features,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)


def load_learned_fusion_model(path: Path):
    import torch

    if LearnedFusionLinear is None:  # pragma: no cover
        raise ImportError("PyTorch is required for learned fusion inference.")

    # weights_only=False is required because the payload contains a state_dict
    # with tensor objects. The file is written exclusively by save_learned_fusion_model
    # above, so the source is trusted. For untrusted files, validate the path
    # before calling this function.
    payload = torch.load(path, map_location="cpu", weights_only=False)
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


def export_learned_fusion_weights(model, feature_cols: list[str], path: Path) -> dict[str, float]:
    return export_learned_fusion_weights_json(model, feature_cols, path)
