"""Phase 4 — Supervised learning runner.

Trains Dummy / LogisticRegression / tuned RandomForest for 2 subjects x 2 scenarios,
saves the 4 main RF pipelines to models/, runs the G3 = 0 sensitivity check and the
absences partial-dependence check, writes reports/metrics.json["supervised"] and figures.

Usage:
    python scripts/train_supervised.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE  # noqa: E402
from src.data import TARGET_COL  # noqa: E402
from src.metrics_io import update_metrics  # noqa: E402
from src.preprocessing import SCENARIOS, feature_columns  # noqa: E402
from src.supervised import (  # noqa: E402
    CV_FOLDS, PERM_REPEATS, RF_PARAM_GRID, SCORING, TEST_SIZE,
    absences_partial_dependence, evaluate, fairness_check, model_path, permutation_top,
    prepare_subject, split_indices, train_reference, train_scenario,
)

SUBJECT_ORDER = ["mat", "por"]
SUBJECT_NAMES = {"mat": "Mathematics", "por": "Portuguese"}
SCENARIO_NAMES = {"early": "Early", "midterm": "Mid-term"}
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
C_TEXT2 = "#52514e"
PD_GAP = 0.02  # PD(0) lower than PD(1–5) by more than this => "0 absences = higher risk"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)
    return f"reports/figures/{name}"


# --------------------------------------------------------------------------- training
def run() -> dict:
    results: dict = {
        "config": {
            "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "cv_folds": CV_FOLDS,
            "cv_scoring": SCORING, "rf_param_grid": {k.removeprefix("clf__"): v for k, v in RF_PARAM_GRID.items()},
            "permutation_repeats": PERM_REPEATS, "permutation_scoring": SCORING,
            "label_names": {"0": "At Risk", "1": "Likely to Pass"}, "headline_metric": "at_risk_recall",
            "features": {s: feature_columns(s) for s in SCENARIOS},
        },
        "subjects": {},
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for subject in SUBJECT_ORDER:
        df = prepare_subject(subject)
        train_idx, test_idx = split_indices(df)
        nonzero = (df["G3"] > 0).to_numpy()
        train_nz, test_nz = train_idx[nonzero[train_idx]], test_idx[nonzero[test_idx]]
        y = df[TARGET_COL]
        subj: dict = {
            "split": {
                "n_train": len(train_idx), "n_test": len(test_idx),
                "train_at_risk": int((y.iloc[train_idx] == 0).sum()),
                "test_at_risk": int((y.iloc[test_idx] == 0).sum()),
                "test_g3_zero": int((~nonzero[test_idx]).sum()),
                "n_train_excl_g3_zero": len(train_nz), "n_test_excl_g3_zero": len(test_nz),
            }
        }
        for scenario in SCENARIOS:
            t0 = time.time()
            features = feature_columns(scenario)
            X = df[features]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            models, metrics = train_scenario(df, train_idx, test_idx, scenario)
            rf = models["rf"]
            joblib.dump(rf, model_path(subject, scenario))

            # Sensitivity: same procedure without G3 = 0 students (not saved).
            sens_models, sens_metrics = train_scenario(df, train_nz, test_nz, scenario, with_baselines=False)
            rf_sens = sens_models["rf"]

            entry = {
                "features": features,
                "models": metrics,
                "permutation_importance": permutation_top(rf, X_te, y_te),
                "fairness": fairness_check(rf, X_te, y_te, df["sex"].iloc[test_idx]),
                "train_reference": train_reference(X.iloc[train_idx]),
                "sensitivity_excl_g3_zero": {
                    "note": "RF retrained and tested without G3 = 0 students; metrics only, model not saved.",
                    "rf_main_on_test_excl_g3_zero": evaluate(rf, X.iloc[test_nz], y.iloc[test_nz]),
                    "rf_excl_g3_zero": sens_metrics["rf"],
                },
                "absences_partial_dependence": {
                    "note": "Mean P(pass) over all students of the subject with absences set to each value.",
                    "rf_main": absences_partial_dependence(rf, X),
                    "rf_excl_g3_zero": absences_partial_dependence(rf_sens, X),
                },
            }
            subj[scenario] = entry
            print(f"  {subject}/{scenario}: done in {time.time() - t0:.0f}s "
                  f"(RF best {metrics['rf']['best_params']})", flush=True)
        results["subjects"][subject] = subj
    return results


# --------------------------------------------------------------------------- figures
def grid_axes(figsize=(10, 8.5), **kw):
    fig, axes = plt.subplots(2, 2, figsize=figsize, **kw)
    return fig, {(s, sc): axes[i, j] for i, s in enumerate(SUBJECT_ORDER) for j, sc in enumerate(SCENARIOS)}


def fig_confusion(res: dict) -> str:
    fig, axes = grid_axes(figsize=(9, 8))
    for (s, sc), ax in axes.items():
        cm = np.array(res["subjects"][s][sc]["models"]["rf"]["test"]["confusion_matrix"]["matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%} of row)", ha="center", va="center",
                        fontsize=10, color="white" if cm[i, j] > cm.max() * 0.6 else "#0b0b0b")
        ax.set_xticks([0, 1], labels=["At Risk", "Likely to Pass"])
        ax.set_yticks([0, 1], labels=["At Risk", "Likely to Pass"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        rec = res["subjects"][s][sc]["models"]["rf"]["test"]["at_risk_recall"]
        ax.set_title(f"{SUBJECT_NAMES[s]} — {SCENARIO_NAMES[sc]}\nAt-Risk recall {rec:.0%}")
    fig.suptitle("Random Forest confusion matrices (held-out test set)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "07_confusion_matrices.png")


def fig_roc(res: dict) -> str:
    fig, axes = grid_axes(figsize=(10, 9), sharex=True, sharey=True)
    styles = {"rf": ("Random Forest", C_BLUE, "-"), "logreg": ("Logistic Regression", C_ORANGE, "-"),
              "dummy": ("Dummy (most frequent)", C_GRAY, "--")}
    for (s, sc), ax in axes.items():
        m = res["subjects"][s][sc]["models"]
        for key in ("rf", "logreg", "dummy"):
            name, color, ls = styles[key]
            t = m[key]["test"]
            ax.plot(t["roc_curve"]["fpr"], t["roc_curve"]["tpr"], color=color, linestyle=ls, linewidth=2,
                    label=f"{name} (AUC {t['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=0.8, linestyle=":")
        ax.set_title(f"{SUBJECT_NAMES[s]} — {SCENARIO_NAMES[sc]}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (Likely to Pass)")
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("ROC curves on the held-out test set", fontweight="bold")
    fig.tight_layout()
    return save(fig, "08_roc_curves.png")


def fig_importance(res: dict) -> str:
    fig, axes = grid_axes(figsize=(11, 9))
    for (s, sc), ax in axes.items():
        imp = res["subjects"][s][sc]["permutation_importance"][::-1]
        names = [d["feature"] for d in imp]
        ax.barh(names, [d["mean"] for d in imp], xerr=[d["std"] for d in imp], color=C_BLUE, height=0.65,
                error_kw={"ecolor": C_TEXT2, "elinewidth": 1, "capsize": 2})
        ax.axvline(0, color=C_TEXT2, linewidth=0.8)
        ax.set_title(f"{SUBJECT_NAMES[s]} — {SCENARIO_NAMES[sc]}")
        ax.set_xlabel("Drop in test macro-F1 when shuffled")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Random Forest permutation importance, top 10 (test set, {PERM_REPEATS} repeats, ± std)",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, "09_feature_importance.png")


def fig_early_vs_midterm(res: dict) -> str:
    keys = [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"), ("at_risk_recall", "At-Risk recall"),
            ("roc_auc", "ROC-AUC")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    x = np.arange(len(keys))
    w = 0.38
    for ax, s in zip(axes, SUBJECT_ORDER):
        for i, (sc, color) in enumerate((("early", C_BLUE), ("midterm", C_AQUA))):
            t = res["subjects"][s][sc]["models"]["rf"]["test"]
            vals = [t[k] for k, _ in keys]
            xs = x + (i - 0.5) * (w + 0.02)
            ax.bar(xs, vals, width=w, color=color, label=SCENARIO_NAMES[sc])
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x, labels=[n for _, n in keys])
        ax.set_ylim(0, 1.12)
        ax.set_title(SUBJECT_NAMES[s])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score (test set)")
    axes[0].legend(loc="upper left", ncol=2)
    fig.suptitle("Random Forest: Early (no grades) vs Mid-term (with G1, G2)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "10_early_vs_midterm.png")


def fig_absences_pd(res: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i, s in enumerate(SUBJECT_ORDER):
        for j, sc in enumerate(SCENARIOS):
            ax = axes[i, j]
            pdd = res["subjects"][s][sc]["absences_partial_dependence"]
            for key, name, color in (("rf_main", "RF main (G3 = 0 kept)", C_BLUE),
                                     ("rf_excl_g3_zero", "RF without G3 = 0", C_AQUA)):
                curve = pdd[key]["curve_0_30"]
                xs = [int(k) for k in curve]
                ax.plot(xs, list(curve.values()), color=color, linewidth=2, marker="o", markersize=3, label=name)
            ax.set_title(f"{SUBJECT_NAMES[s]} — {SCENARIO_NAMES[sc]}")
            ax.set_xlabel("absences (set for every student)")
            ax.set_ylabel("Mean predicted P(pass)")
            ax.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Partial dependence of predicted pass probability on absences", fontweight="bold")
    fig.tight_layout()
    return save(fig, "11_absences_partial_dependence.png")


# --------------------------------------------------------------------------- summary
def print_summary(res: dict) -> None:
    print("\n" + "=" * 100)
    print("TEST-SET COMPARISON  (RF w/o G3=0 is trained and tested without G3 = 0 students;"
          " 'RF main*' = main RF on that same reduced test set)")
    print("=" * 100)
    hdr = f"{'subject/scenario':18s} {'model':22s} {'n_test':>6s} {'acc':>6s} {'F1mac':>6s} {'AR-rec':>7s} {'AUC':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            e = res["subjects"][s][sc]
            rows = [("Dummy", e["models"]["dummy"]["test"]), ("LogReg", e["models"]["logreg"]["test"]),
                    ("RF main", e["models"]["rf"]["test"]),
                    ("RF main* (test w/o 0)", e["sensitivity_excl_g3_zero"]["rf_main_on_test_excl_g3_zero"]),
                    ("RF w/o G3=0", e["sensitivity_excl_g3_zero"]["rf_excl_g3_zero"]["test"])]
            for k, (name, t) in enumerate(rows):
                tag = f"{s}/{sc}" if k == 0 else ""
                print(f"{tag:18s} {name:22s} {t['n']:6d} {t['accuracy']:6.3f} {t['f1_macro']:6.3f} "
                      f"{t['at_risk_recall']:7.3f} {t['roc_auc']:6.3f}")
            print()

    print("CV macro-F1 (train, 5-fold) and RF best params:")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            m = res["subjects"][s][sc]["models"]
            cv = {k: f"{m[k]['cv']['mean']:.3f}±{m[k]['cv']['std']:.3f}" for k in ("dummy", "logreg", "rf")}
            print(f"  {s}/{sc:8s} dummy {cv['dummy']}  logreg {cv['logreg']}  rf {cv['rf']}  {m['rf']['best_params']}")

    print("\nTop 5 permutation importance (RF main):")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            imp = res["subjects"][s][sc]["permutation_importance"][:5]
            print(f"  {s}/{sc:8s} " + ", ".join(f"{d['feature']} {d['mean']:.3f}" for d in imp))

    print("\nFairness — At-Risk recall by sex (test set, RF main):")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            f = res["subjects"][s][sc]["fairness"]
            print(f"  {s}/{sc:8s} F {f['F']['at_risk_recall']:.2f} (n={f['F']['n_at_risk']})   "
                  f"M {f['M']['at_risk_recall']:.2f} (n={f['M']['n_at_risk']})")

    print("\nAbsences partial dependence — mean P(pass) by absences bin:")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            for key in ("rf_main", "rf_excl_g3_zero"):
                b = res["subjects"][s][sc]["absences_partial_dependence"][key]["by_bin"]
                flag = "  <-- 0 absences => higher risk" if b["0"] < b["1–5"] - PD_GAP else ""
                print(f"  {s}/{sc:8s} {key:16s} " + "  ".join(f"{k}: {v:.3f}" for k, v in b.items()) + flag)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    t0 = time.time()
    print("Training supervised models ...", flush=True)
    res = run()
    res["figures"] = [fig_confusion(res), fig_roc(res), fig_importance(res), fig_early_vs_midterm(res),
                      fig_absences_pd(res)]
    update_metrics("supervised", res)
    print_summary(res)
    print("\nSaved models:", ", ".join(f"models/{model_path(s, sc).name}" for s in SUBJECT_ORDER for sc in SCENARIOS))
    print("Figures:", ", ".join(res["figures"]))
    print(f"Saved metrics.json['supervised']  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
