"""Read/merge helpers for reports/metrics.json (single source of all reported numbers)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import METRICS_PATH


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars and containers to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    return obj


def load_metrics(path: Path = METRICS_PATH) -> dict:
    """Return reports/metrics.json as a dict ({} if the file does not exist yet)."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_metrics(section: str, data: dict, path: Path = METRICS_PATH) -> None:
    """Replace one top-level ``section`` of metrics.json, keeping every other key."""
    metrics = load_metrics(path)
    metrics[section] = _to_builtin(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
