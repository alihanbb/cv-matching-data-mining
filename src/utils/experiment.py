from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_cfg(cfg: dict[str, Any]) -> bytes:
    return yaml.dump(cfg, sort_keys=True, allow_unicode=True).encode("utf-8")


def write_run_manifest(
    root: Path,
    cfg: dict[str, Any],
    artifact_paths: dict[str, str],
    metrics: dict[str, float],
    notes: str | None = None,
) -> Path:
    runs = root / "artifacts" / "runs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = hashlib.sha256(_canonical_cfg(cfg)).hexdigest()
    data_hashes = {k: _sha256_file(Path(v)) for k, v in artifact_paths.items() if k.startswith("input_")}
    manifest = {
        "run_id": stamp,
        "config_sha256": cfg_hash,
        "metrics": metrics,
        "artifacts": artifact_paths,
        "input_file_sha256": {k: v for k, v in data_hashes.items() if v},
        "notes": notes or "",
    }
    out = run_dir / "manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return run_dir
