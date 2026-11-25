"""Phase 4b — Repeated-split evaluation of the full supervised procedure.

For each subject x scenario and each seed in SEEDS: stratified 80/20 split with that seed,
Dummy + LogReg baselines, monotonic RF tuned with GridSearchCV, alert cut-off from training
out-of-fold predictions, then test metrics at 0.5 and at the tuned cut-off.

Writes reports/metrics.json["supervised_repeated_splits"] (mean ± std per metric, per-seed rows,
and where the seed-42 split used by the saved models falls in that distribution).

Usage:
    python scripts/repeated_splits.py            # ~30–40 min on 4 cores
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import FIGURES_DIR  # noqa: E402
from src.metrics_io import load_metrics, update_metrics  # noqa: E402
from src.preprocessing import SCENARIOS  # noqa: E402
from src.supervised import REPEATED_KEYS, run_split  # noqa: E402

SEEDS = list(range(20))
SUBJECT_ORDER = ["mat", "por"]
SUBJECT_NAMES = {"mat": "Mathematics", "por": "Portuguese"}
SCENARIO_NAMES = {"early": "Early", "midterm": "Mid-term"}
MODEL_KEYS = ("dummy", "logreg", "rf_default", "rf_tuned")  # rf_tuned = final cut-off (with 0.5 floor)
C_BLUE, C_GRAY, C_ORANGE, C_TEXT2 = "#2a78d6", "#8a8984", "#eb6834", "#52514e"


def summarise(rows: list[dict], seed42: dict | None) -> dict:
    """Mean ± std per model and metric, pooled fairness, and where seed 42 falls."""
    out: dict = {"n_splits": len(rows), "seeds": [r["seed"] for r in rows]}
    for mk in MODEL_KEYS:
        out[mk] = {}
        for k in REPEATED_KEYS:
            vals = np.array([r[mk][k] for r in rows], dtype=float)
            out[mk][k] = {"mean": round(vals.mean(), 4), "std": round(vals.std(ddof=1), 4),
                          "min": round(vals.min(), 4), "max": round(vals.max(), 4)}
    cv = np.array([r["rf_cv_f1_macro"] for r in rows])
    test = np.array([r["rf_default"]["f1_macro"] for r in rows])
    gap = cv - test
    out["rf_cv_f1_macro"] = {"mean": round(cv.mean(), 4), "std": round(cv.std(ddof=1), 4)}
    out["rf_cv_minus_test_f1_macro"] = {"mean": round(gap.mean(), 4), "std": round(gap.std(ddof=1), 4),
                                        "min": round(gap.min(), 4), "max": round(gap.max(), 4)}
    thr = np.array([r["rf_threshold"] for r in rows])
    out["rf_threshold"] = {"mean": round(thr.mean(), 4), "std": round(thr.std(ddof=1), 4),
                           "min": round(thr.min(), 4), "max": round(thr.max(), 4)}
    raw = np.array([r["rf_threshold_raw"] for r in rows])
    out["rf_threshold_raw"] = {"mean": round(raw.mean(), 4), "std": round(raw.std(ddof=1), 4),
                               "share_floored": round(float((raw < 0.5).mean()), 4)}
    # Fairness: pool caught / At-Risk counts over all test sets (not an average of percentages).
    out["fairness_pooled_final"] = {}
    for g in ("F", "M"):
        caught = int(sum(r["fairness_final"][g]["n_caught"] for r in rows))
        total = int(sum(r["fairness_final"][g]["n_at_risk"] for r in rows))
        out["fairness_pooled_final"][g] = {"n_caught": caught, "n_at_risk": total,
                                           "at_risk_recall": round(caught / total, 4) if total else None}
    out["per_seed"] = [{k: v for k, v in r.items() if k not in ("subject", "scenario")} for r in rows]
    if seed42 is not None:
        f1_42 = seed42["test_default"]["f1_macro"]
        out["seed42"] = {
            "rf_default_f1_macro": f1_42,
            "rf_default_at_risk_recall": seed42["test_default"]["at_risk_recall"],
            "rf_tuned_f1_macro": seed42["test_tuned"]["f1_macro"],
            "rf_tuned_at_risk_recall": seed42["test_tuned"]["at_risk_recall"],
            "threshold": seed42["threshold_tuning"]["threshold"],
            "rf_cv_f1_macro": seed42["cv"]["mean"],
            "rf_default_f1_macro_z": round((f1_42 - test.mean()) / test.std(ddof=1), 2),
            "share_of_seeds_with_lower_f1_macro": round(float((test < f1_42).mean()), 4),
        }
    return out


def fig_repeated(summary: dict) -> str:
    """Figure 13: per-split results with mean ± std."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = [f"{SUBJECT_NAMES[s]}\n{SCENARIO_NAMES[sc]}" for s in SUBJECT_ORDER for sc in SCENARIOS]
    for ax, metric, title in ((axes[0], "f1_macro", "Test macro-F1"),
                              (axes[1], "at_risk_recall", "Test At-Risk recall")):
        for i, (s, sc) in enumerate((s, sc) for s in SUBJECT_ORDER for sc in SCENARIOS):
            e = summary["subjects"][s][sc]
            for mk, dx, color, name in (("rf_default", -0.15, C_GRAY, "RF @0.5"),
                                        ("rf_tuned", 0.15, C_BLUE, "RF @final cut-off")):
                vals = [r[mk][metric] for r in e["per_seed"]]
                jitter = np.linspace(-0.05, 0.05, len(vals))
                ax.scatter(i + dx + jitter, vals, s=14, color=color, alpha=0.55, edgecolors="none",
                           label=name if i == 0 else None)
                m, sd = e[mk][metric]["mean"], e[mk][metric]["std"]
                ax.errorbar(i + dx, m, yerr=sd, fmt="o", color="#0b0b0b", markersize=5, capsize=4, linewidth=1.2)
            if "seed42" in e:
                key = "rf_default_f1_macro" if metric == "f1_macro" else "rf_default_at_risk_recall"
                ax.scatter(i - 0.15, e["seed42"][key], marker="D", s=40, color=C_ORANGE, zorder=5,
                           label="seed 42 (saved split), RF @0.5" if i == 0 else None)
        ax.set_xticks(range(len(labels)), labels=labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Monotonic RF over {len(SEEDS)} random stratified splits (dots = splits, black = mean ± std)",
                 fontweight="bold")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "13_repeated_splits.png")
    plt.close(fig)
    return "reports/figures/13_repeated_splits.png"


def main() -> int:
    """Run all subject × scenario × seed splits in parallel and save the summary."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    jobs = [(s, sc, seed) for s in SUBJECT_ORDER for sc in SCENARIOS for seed in SEEDS]
    print(f"Running {len(jobs)} splits in parallel ...", flush=True)
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(run_split)(s, sc, seed) for s, sc, seed in jobs)

    sup = load_metrics().get("supervised", {}).get("subjects", {})
    summary: dict = {
        "config": {"seeds": SEEDS, "procedure": "same as saved models: stratified 80/20 split (seed varies), "
                   "GridSearchCV monotonic RF, cut-off = max(0.5, OOF cut-off at >= 70% At-Risk recall), "
                   "single test evaluation",
                   "fairness": "At-Risk recall by sex at the final cut-off, caught/total counts pooled over all splits"},
        "subjects": {},
    }
    for s in SUBJECT_ORDER:
        summary["subjects"][s] = {}
        for sc in SCENARIOS:
            r = [x for x in rows if x["subject"] == s and x["scenario"] == sc]
            seed42 = sup.get(s, {}).get(sc, {}).get("models", {}).get("rf")
            summary["subjects"][s][sc] = summarise(r, seed42)
    summary["figure"] = fig_repeated(summary)
    update_metrics("supervised_repeated_splits", summary)

    print("\n" + "=" * 100)
    print(f"REPEATED SPLITS ({len(SEEDS)} seeds): test mean ± std")
    print("=" * 100)
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            e = summary["subjects"][s][sc]
            print(f"\n[{s} / {sc}]  RF CV macro-F1 {e['rf_cv_f1_macro']['mean']:.3f} ± {e['rf_cv_f1_macro']['std']:.3f}"
                  f"   CV - test gap {e['rf_cv_minus_test_f1_macro']['mean']:+.3f} ± "
                  f"{e['rf_cv_minus_test_f1_macro']['std']:.3f}   cut-off {e['rf_threshold']['mean']:.3f} ± "
                  f"{e['rf_threshold']['std']:.3f}")
            for mk in MODEL_KEYS:
                print(f"  {mk:11s} " + "  ".join(
                    f"{k} {e[mk][k]['mean']:.3f}±{e[mk][k]['std']:.3f}" for k in REPEATED_KEYS))
            fp = e["fairness_pooled_final"]
            print(f"  pooled fairness @final: F {fp['F']['at_risk_recall']:.3f} ({fp['F']['n_caught']}/{fp['F']['n_at_risk']})"
                  f"   M {fp['M']['at_risk_recall']:.3f} ({fp['M']['n_caught']}/{fp['M']['n_at_risk']})"
                  f"   floored in {e['rf_threshold_raw']['share_floored']:.0%} of splits")
            if "seed42" in e:
                z = e["seed42"]
                print(f"  seed 42 (saved): RF@0.5 macro-F1 {z['rf_default_f1_macro']:.3f} (z = {z['rf_default_f1_macro_z']:+.2f}, "
                      f"{z['share_of_seeds_with_lower_f1_macro']:.0%} of seeds lower), AR-recall "
                      f"{z['rf_default_at_risk_recall']:.3f};  RF@final macro-F1 {z['rf_tuned_f1_macro']:.3f}, "
                      f"AR-recall {z['rf_tuned_at_risk_recall']:.3f}")
    print(f"\nSaved metrics.json['supervised_repeated_splits'], {summary['figure']}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
