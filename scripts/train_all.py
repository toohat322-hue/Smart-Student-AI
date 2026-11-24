"""Phase 7 — End-to-end training: regenerates models/ and reports/ with fixed seeds.

Steps (each runs as its own process, in order, stopping at the first failure):
  1. data         download the CSVs if missing, then validate both subjects' schema
  2. eda          reports/figures/01–06, metrics.json["eda"]
  3. supervised   models/clf_*.joblib, figures 07–12, metrics.json["supervised"]          (~8 min)
  4. clustering   models/kmeans_*.joblib, figures 14–16, metrics.json["clustering"]
  5. rl           models/q_tables.joblib, figures 17–20, metrics.json["rl"]                (~2 min)
  6. repeated     (only with --full) figure 13, metrics.json["supervised_repeated_splits"] (~30 min)

Without --full the existing metrics.json["supervised_repeated_splits"] is left untouched.

Usage:
    python scripts/train_all.py           # everything except the 20-split evaluation (~7–10 min)
    python scripts/train_all.py --full    # also the 20-split evaluation (~40 min)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PY = sys.executable

VALIDATE = (
    "from src.data import load_subject, add_target, validate_schema\n"
    "for s in ('mat', 'por'):\n"
    "    df = add_target(load_subject(s)); validate_schema(df)\n"
    "    print(f'  {s}: {df.shape[0]} rows x {df.shape[1] - 1} columns, schema OK')\n"
)

STEPS = [
    ("data", [[PY, "scripts/download_data.py"], [PY, "-c", VALIDATE]]),
    ("eda", [[PY, "scripts/eda.py"]]),
    ("supervised", [[PY, "scripts/train_supervised.py"]]),
    ("clustering", [[PY, "scripts/train_clustering.py"]]),
    ("rl", [[PY, "scripts/train_rl.py"]]),
]
FULL_STEPS = [("repeated", [[PY, "scripts/repeated_splits.py"]])]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Regenerate all models and reports.")
    parser.add_argument("--full", action="store_true", help="also run the ~30 min 20-split evaluation")
    args = parser.parse_args(argv)

    steps = STEPS + (FULL_STEPS if args.full else [])
    timings: list[tuple[str, float]] = []
    t_all = time.time()
    for name, commands in steps:
        print(f"\n{'=' * 80}\n[{len(timings) + 1}/{len(steps)}] {name}\n{'=' * 80}", flush=True)
        t0 = time.time()
        for cmd in commands:
            proc = subprocess.run(cmd, cwd=ROOT_DIR, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            if proc.returncode != 0:
                print(f"\nFAILED at step '{name}' (exit code {proc.returncode}): {' '.join(cmd[:2])}", flush=True)
                return proc.returncode
        timings.append((name, time.time() - t0))
    if not args.full:
        print("\n(skipped the 20-split evaluation; existing metrics.json['supervised_repeated_splits'] kept. "
              "Use --full to rerun it.)")

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    for name, secs in timings:
        print(f"  {name:12s} {secs:7.1f} s")
    print(f"  {'total':12s} {time.time() - t_all:7.1f} s")

    print("\nSaved files:")
    models = sorted((ROOT_DIR / "models").glob("*.joblib"))
    for f in models:
        print(f"  models/{f.name:28s} {human(f.stat().st_size):>9s}")
    print(f"  {'models total':35s} {human(sum(f.stat().st_size for f in models)):>9s}")
    metrics = ROOT_DIR / "reports" / "metrics.json"
    print(f"  reports/metrics.json                {human(metrics.stat().st_size):>9s}")
    figs = sorted((ROOT_DIR / "reports" / "figures").glob("*.png"))
    print(f"  reports/figures/ ({len(figs)} PNG)          {human(sum(f.stat().st_size for f in figs)):>9s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
