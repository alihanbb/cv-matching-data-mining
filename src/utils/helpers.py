from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    cfg_path = Path(path) if path else root / "config" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def resolve_path(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute():
        return p
    return (root / p).resolve()
