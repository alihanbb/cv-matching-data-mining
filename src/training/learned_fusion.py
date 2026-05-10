from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.matching_inputs import build_matching_matrices
from src.schemas.documents import validate_ground_truth_df
from src.utils.helpers import resolve_path

logger = logging.getLogger(__name__)


def train_learned_fusion(
    root: Path,
    cfg: dict[str, Any],
    *,
    semantic: bool = True,
    epochs: int = 200,
    lr: float = 0.05,
) -> dict[str, float]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("PyTorch is required for --train-fusion. Install torch.") from e

    gt_path = resolve_path(
        root, cfg["paths"].get("ground_truth", "data/evaluation/ground_truth.csv")
    )
    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Ground truth is required for learned fusion: {gt_path}\n"
            "Create data/evaluation/ground_truth.csv with columns job_id,cv_id,relevance."
        )
    gt = validate_ground_truth_df(pd.read_csv(gt_path))
    if gt.empty:
        raise ValueError("Ground truth is empty.")

    mats = build_matching_matrices(root, cfg, semantic=semantic, bm25=True)
    cv_index = {cid: i for i, cid in enumerate(mats.cv_ids)}
    job_index = {jid: j for j, jid in enumerate(mats.job_ids)}

    dense = mats.dense_sim if mats.dense_sim is not None else np.zeros_like(mats.sim_lex)
    bm = mats.bm25 if mats.bm25 is not None else np.zeros_like(mats.sim_lex)

    xs: list[list[float]] = []
    ys: list[float] = []
    for _, r in gt.iterrows():
        i = cv_index.get(str(r["cv_id"]))
        j = job_index.get(str(r["job_id"]))
        if i is None or j is None:
            continue
        feat = [
            float(mats.sim_lex[i, j]),
            float(dense[i, j]),
            float(bm[i, j]),
            float(mats.skill_score[i, j]),
            float(mats.exp_mat[i, j]),
        ]
        xs.append(feat)
        ys.append(float(r["relevant"]) / 3.0)

    if len(xs) < 3:
        raise ValueError(
            f"Need at least 3 labeled pairs overlapping processed CV/job IDs; found {len(xs)} usable rows."
        )

    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

    class FusionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = torch.softmax(self.logits, dim=0)
            return (x * w).sum(dim=1, keepdim=True)

    model = FusionNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = torch.softmax(model.logits, dim=0).cpu().tolist()
    names = ["tfidf", "semantic", "bm25", "skills", "experience"]
    out = dict(zip(names, [float(v) for v in w], strict=True))
    art_dir = resolve_path(root, "artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)
    out_path = art_dir / "learned_fusion_weights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote learned fusion weights to %s: %s", out_path, out)
    return out
