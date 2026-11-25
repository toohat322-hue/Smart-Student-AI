"""Reproducibility fingerprint of everything train_all.py produces.

Hashes reports/metrics.json and the *behaviour* of every saved model on the full datasets:
classifier P(pass) and labels, K-Means labels and centroids, Q-tables. File bytes are reported
too, but two pickles of identical models may legitimately differ in bytes.

Usage:
    python scripts/fingerprint.py out.json            # write a fingerprint
    python scripts/fingerprint.py out.json --compare old.json   # write and compare with an older one
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import CLUSTER_FEATURES  # noqa: E402
from src.preprocessing import SCENARIOS, feature_columns  # noqa: E402
from src.supervised import prepare_subject  # noqa: E402


def h(data: bytes) -> str:
    """Short SHA-256 of some bytes."""
    return hashlib.sha256(data).hexdigest()[:16]


def arr(a) -> bytes:
    """Array as float64 bytes (for hashing)."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()


def fingerprint() -> dict:
    """Hashes of metrics.json sections, model behaviour and file bytes."""
    fp: dict = {"metrics.json": h((ROOT_DIR / "reports/metrics.json").read_bytes()),
                "metrics_sections": {}, "behaviour": {}, "file_bytes": {}}
    metrics = json.loads((ROOT_DIR / "reports/metrics.json").read_text(encoding="utf-8"))
    for k, v in metrics.items():
        fp["metrics_sections"][k] = h(json.dumps(v, sort_keys=True).encode())
    for s in ("mat", "por"):
        df = prepare_subject(s)
        for sc in SCENARIOS:
            m = joblib.load(ROOT_DIR / f"models/clf_{s}_{sc}.joblib")
            X = df[feature_columns(sc)]
            fp["behaviour"][f"clf_{s}_{sc}"] = h(arr(m.predict_proba(X)) + arr(m.predict(X)) + arr([m.threshold]))
        km = joblib.load(ROOT_DIR / f"models/kmeans_{s}.joblib")
        fp["behaviour"][f"kmeans_{s}"] = h(arr(km.predict(df[CLUSTER_FEATURES])) + arr(km[-1].cluster_centers_))
    q = joblib.load(ROOT_DIR / "models/q_tables.joblib")
    fp["behaviour"]["q_tables"] = h(b"".join(str(k).encode() + arr(q[k]) for k in sorted(q)))
    for f in sorted((ROOT_DIR / "models").glob("*.joblib")):
        fp["file_bytes"][f.name] = h(f.read_bytes())
    for f in sorted((ROOT_DIR / "reports/figures").glob("*.png")):
        fp["file_bytes"][f.name] = h(f.read_bytes())
    return fp


def compare(a: dict, b: dict) -> list[str]:
    """Names of everything that differs between two fingerprints."""
    diffs = []
    for group in ("metrics_sections", "behaviour", "file_bytes"):
        for k in sorted(set(a[group]) | set(b[group])):
            if a[group].get(k) != b[group].get(k):
                diffs.append(f"{group}/{k}")
    if a["metrics.json"] != b["metrics.json"]:
        diffs.append("metrics.json (whole file)")
    return diffs


def main() -> int:
    """Write a fingerprint and optionally compare it with an older one."""
    out = Path(sys.argv[1])
    fp = fingerprint()
    out.write_text(json.dumps(fp, indent=2), encoding="utf-8")
    print(f"metrics.json {fp['metrics.json']}; behaviour:", ", ".join(f"{k}={v[:8]}" for k, v in fp["behaviour"].items()))
    if "--compare" in sys.argv:
        old = json.loads(Path(sys.argv[sys.argv.index("--compare") + 1]).read_text(encoding="utf-8"))
        diffs = compare(old, fp)
        print("IDENTICAL" if not diffs else "DIFFERENCES:\n  " + "\n  ".join(diffs))
        return 1 if diffs else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
