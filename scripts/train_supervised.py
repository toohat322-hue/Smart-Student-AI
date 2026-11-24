"""Phase 4 — Supervised learning runner.

Trains Dummy / LogisticRegression / tuned RandomForest for 2 subjects x 2 scenarios.
Main RF = monotonic in absences + alert cut-off tuned on training out-of-fold predictions;
it is wrapped in FixedThresholdClassifier and saved to models/. For comparison (not saved):
the unconstrained RF, and the RF retrained without G3 = 0 students.
Writes reports/metrics.json["supervised"] and figures 07–12.

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
from src.preprocessing import MONOTONIC_DECREASING, SCENARIOS, feature_columns  # noqa: E402
from src.supervised import (  # noqa: E402
    CV_FOLDS, DEFAULT_THRESHOLD, PERM_REPEATS, RF_PARAM_GRID, SCORING, TARGET_AT_RISK_RECALL, TEST_SIZE,
    THRESHOLD_FLOOR,
    absences_partial_dependence, evaluate, fairness_check, model_path, permutation_top,
    prepare_subject, split_indices, train_reference, train_scenario,
)

SUBJECT_ORDER = ["mat", "por"]
SUBJECT_NAMES = {"mat": "Mathematics", "por": "Portuguese"}
SCENARIO_NAMES = {"early": "Early", "midterm": "Mid-term"}
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
C_TEXT2 = "#52514e"
THRESHOLD_RULE = (f"cut-off = max({THRESHOLD_FLOOR}, smallest cut-off with out-of-fold At-Risk recall >= "
                  f"{TARGET_AT_RISK_RECALL:.0%} on the training set (5-fold, best RF params)); "
                  "Likely to Pass iff P(pass) >= cut-off")
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
            "monotonic_decreasing_features": MONOTONIC_DECREASING,
            "default_threshold": DEFAULT_THRESHOLD,
            "threshold_rule": THRESHOLD_RULE,
            "saved_model": "FixedThresholdClassifier(Pipeline(preprocessor, RandomForest with monotonic_cst))",
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
            sex_te = df["sex"].iloc[test_idx]

            models, metrics = train_scenario(df, train_idx, test_idx, scenario)
            rf, rf_old = models["rf"], models["rf_unconstrained"]
            joblib.dump(rf, model_path(subject, scenario))

            # Sensitivity: same procedure without G3 = 0 students (not saved).
            sens_models, sens_metrics = train_scenario(df, train_nz, test_nz, scenario, with_baselines=False,
                                                       with_unconstrained=False)
            X_nz, y_nz = X.iloc[test_nz], y.iloc[test_nz]

            entry = {
                "features": features,
                "threshold": rf.threshold,
                "models": metrics,
                "permutation_importance": permutation_top(rf, X_te, y_te),
                "fairness": {
                    "rf_default": fairness_check(rf, X_te, y_te, sex_te, DEFAULT_THRESHOLD),
                    "rf_tuned": fairness_check(rf, X_te, y_te, sex_te, rf.threshold),
                    "rf_unconstrained_default": fairness_check(rf_old, X_te, y_te, sex_te, DEFAULT_THRESHOLD),
                },
                "train_reference": train_reference(X.iloc[train_idx]),
                "sensitivity_excl_g3_zero": {
                    "note": "RF retrained and tested without G3 = 0 students; metrics only, model not saved.",
                    "rf_main_on_test_excl_g3_zero": {
                        "test_default": evaluate(rf, X_nz, y_nz, DEFAULT_THRESHOLD, include_curves=False),
                        "test_tuned": evaluate(rf, X_nz, y_nz, rf.threshold, include_curves=False),
                    },
                    "rf_excl_g3_zero": sens_metrics["rf"],
                },
                "absences_partial_dependence": {
                    "note": "Mean P(pass) over all students of the subject with absences set to each value.",
                    "rf_main": absences_partial_dependence(rf, X),
                    "rf_unconstrained": absences_partial_dependence(rf_old, X),
                    "rf_excl_g3_zero": absences_partial_dependence(sens_models["rf"], X),
                },
            }
            subj[scenario] = entry
            print(f"  {subject}/{scenario}: done in {time.time() - t0:.0f}s (RF best {metrics['rf']['best_params']}, "
                  f"cut-off {rf.threshold})", flush=True)
        results["subjects"][subject] = subj
    return results


# --------------------------------------------------------------------------- figures
def grid_axes(figsize=(10, 8.5), **kw):
    fig, axes = plt.subplots(2, 2, figsize=figsize, **kw)
    return fig, {(s, sc): axes[i, j] for i, s in enumerate(SUBJECT_ORDER) for j, sc in enumerate(SCENARIOS)}


def fig_confusion(res: dict) -> str:
    fig, axes = grid_axes(figsize=(9, 8))
    for (s, sc), ax in axes.items():
        t = res["subjects"][s][sc]["models"]["rf"]["test_tuned"]
        cm = np.array(t["confusion_matrix"]["matrix"])
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
        ax.set_title(f"{SUBJECT_NAMES[s]} — {SCENARIO_NAMES[sc]} (cut-off {t['threshold']:.3f})\n"
                     f"At-Risk recall {t['at_risk_recall']:.0%}, false alerts {t['false_alerts']}")
    fig.suptitle("Random Forest confusion matrices at the final alert cut-off (held-out test set)",
                 fontweight="bold")
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
            t = m[key]["test_default"]
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
    fig.suptitle(f"Random Forest permutation importance, top 10 (test set, final cut-off, {PERM_REPEATS} "
                 "repeats, ± std)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "09_feature_importance.png")


def fig_early_vs_midterm(res: dict) -> str:
    keys = [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"), ("at_risk_recall", "At-Risk recall"),
            ("at_risk_precision", "At-Risk precision"), ("roc_auc", "ROC-AUC")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    x = np.arange(len(keys))
    w = 0.38
    for ax, s in zip(axes, SUBJECT_ORDER):
        for i, (sc, color) in enumerate((("early", C_BLUE), ("midterm", C_AQUA))):
            t = res["subjects"][s][sc]["models"]["rf"]["test_tuned"]
            vals = [t[k] for k, _ in keys]
            xs = x + (i - 0.5) * (w + 0.02)
            ax.bar(xs, vals, width=w, color=color, label=f"{SCENARIO_NAMES[sc]} (cut-off {t['threshold']:.2f})")
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x, labels=[n for _, n in keys], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(SUBJECT_NAMES[s])
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", ncol=2, fontsize=9)
    axes[0].set_ylabel("Score (test set)")
    fig.suptitle("Random Forest at final cut-off: Early (no grades) vs Mid-term (with G1, G2)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "10_early_vs_midterm.png")


def fig_absences_pd(res: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    lines = (("rf_unconstrained", "RF unconstrained (old)", C_GRAY, "--"),
             ("rf_main", "RF monotonic (saved)", C_BLUE, "-"),
             ("rf_excl_g3_zero", "RF monotonic, without G3 = 0", C_AQUA, "-"))
    for i, s in enumerate(SUBJECT_ORDER):
        for j, sc in enumerate(SCENARIOS):
            ax = axes[i, j]
            pdd = res["subjects"][s][sc]["absences_partial_dependence"]
            for key, name, color, ls in lines:
                curve = pdd[key]["curve_0_30"]
                ax.plot([int(k) for k in curve], list(curve.values()), color=color, linestyle=ls, linewidth=2,
                        marker="o", markersize=3, label=name)
            ax.set_title(f"{SUBJECT_NAMES[s]} — {SCENARIO_NAMES[sc]}")
            ax.set_xlabel("absences (set for every student)")
            ax.set_ylabel("Mean predicted P(pass)")
            ax.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Partial dependence of predicted pass probability on absences", fontweight="bold")
    fig.tight_layout()
    return save(fig, "11_absences_partial_dependence.png")


def fig_threshold_effect(res: dict) -> str:
    """At-Risk recall and false alerts at 0.5 vs tuned cut-off."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = [f"{s} / {sc}" for s in SUBJECT_ORDER for sc in SCENARIOS]
    x = np.arange(len(labels))
    w = 0.38
    for ax, key, ylabel in ((axes[0], "at_risk_recall", "At-Risk recall"),
                            (axes[1], "false_alert_rate", "False-alert rate (share of passing students)")):
        for i, (mode, name, color) in enumerate((("test_default", "cut-off 0.5", C_GRAY),
                                                 ("test_tuned", "final cut-off", C_BLUE))):
            vals = [res["subjects"][s][sc]["models"]["rf"][mode][key] for s in SUBJECT_ORDER for sc in SCENARIOS]
            xs = x + (i - 0.5) * (w + 0.02)
            ax.bar(xs, vals, width=w, color=color, label=name)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x, labels=labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", ncol=2)
    axes[0].axhline(TARGET_AT_RISK_RECALL, color=C_TEXT2, linestyle=":", linewidth=1)
    fig.suptitle("Effect of the final alert cut-off on the test set (monotonic RF)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "12_threshold_effect.png")


# --------------------------------------------------------------------------- summary
def row(name: str, t: dict) -> str:
    return (f"  {name:34s} {t['threshold']:6.3f} {t['accuracy']:6.3f} {t['f1_macro']:6.3f} "
            f"{t['at_risk_recall']:7.3f} {t['at_risk_precision']:7.3f} {t['roc_auc']:6.3f} "
            f"{t['false_alerts']:4d}/{t['n'] - t['n_at_risk']:<4d}")


def print_summary(res: dict) -> None:
    print("\n" + "=" * 104)
    print("TEST-SET COMPARISON (seed-42 split)")
    print("=" * 104)
    hdr = (f"  {'model':34s} {'cut':>6s} {'acc':>6s} {'F1mac':>6s} {'AR-rec':>7s} {'AR-prec':>7s} {'AUC':>6s} "
           f"{'false alerts':>12s}")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            e = res["subjects"][s][sc]
            m = e["models"]
            print(f"\n[{s} / {sc}]   n_test={m['rf']['test_default']['n']}, "
                  f"At Risk={m['rf']['test_default']['n_at_risk']}")
            print(hdr)
            print(row("Dummy", m["dummy"]["test_default"]))
            print(row("LogReg", m["logreg"]["test_default"]))
            print(row("RF old (unconstrained) @0.5", m["rf_unconstrained"]["test_default"]))
            print(row("RF old (unconstrained) @tuned", m["rf_unconstrained"]["test_tuned"]))
            print(row("RF monotonic @0.5", m["rf"]["test_default"]))
            print(row("RF monotonic @final  [SAVED]", m["rf"]["test_tuned"]))
            sens = e["sensitivity_excl_g3_zero"]
            print(row("RF monotonic @tuned, test w/o G3=0", sens["rf_main_on_test_excl_g3_zero"]["test_tuned"]))
            print(row("RF w/o G3=0 @tuned, test w/o G3=0", sens["rf_excl_g3_zero"]["test_tuned"]))

    print("\nCut-off tuning (training out-of-fold) and CV macro-F1 (GridSearchCV mean_test_score):")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            for key in ("rf", "rf_unconstrained"):
                m = res["subjects"][s][sc]["models"][key]
                t = m["threshold_tuning"]
                print(f"  {s}/{sc:8s} {key:17s} cut-off {t['threshold']:.3f}  OOF AR-recall "
                      f"{t['oof_at_risk_recall_at_default']:.2f} -> {t['oof_at_risk_recall']:.2f}  "
                      f"(raw cut-off {t['threshold_raw']:.3f})  CV F1 {m['cv']['mean']:.3f}±{m['cv']['std']:.3f}  "
                      f"{m['best_params']}")

    print("\nTop 5 permutation importance (saved model):")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            imp = res["subjects"][s][sc]["permutation_importance"][:5]
            print(f"  {s}/{sc:8s} " + ", ".join(f"{d['feature']} {d['mean']:.3f}" for d in imp))

    print("\nFairness — At-Risk recall by sex on the test set (caught / At-Risk):")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            f = res["subjects"][s][sc]["fairness"]
            parts = []
            for key, name in (("rf_unconstrained_default", "old@0.5"), ("rf_default", "mono@0.5"),
                              ("rf_tuned", "mono@tuned")):
                g = f[key]
                parts.append(f"{name}: F {g['F']['at_risk_recall']:.2f} ({g['F']['n_caught']}/{g['F']['n_at_risk']})"
                             f"  M {g['M']['at_risk_recall']:.2f} ({g['M']['n_caught']}/{g['M']['n_at_risk']})")
            print(f"  {s}/{sc:8s} " + "  |  ".join(parts))

    print("\nAbsences partial dependence — mean P(pass) by absences bin:")
    for s in SUBJECT_ORDER:
        for sc in SCENARIOS:
            for key in ("rf_unconstrained", "rf_main", "rf_excl_g3_zero"):
                b = res["subjects"][s][sc]["absences_partial_dependence"][key]["by_bin"]
                flag = "  <-- 0 absences => higher risk" if b["0"] < b["1–5"] - PD_GAP else ""
                print(f"  {s}/{sc:8s} {key:17s} " + "  ".join(f"{k}: {v:.3f}" for k, v in b.items()) + flag)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    t0 = time.time()
    print("Training supervised models ...", flush=True)
    res = run()
    res["figures"] = [fig_confusion(res), fig_roc(res), fig_importance(res), fig_early_vs_midterm(res),
                      fig_absences_pd(res), fig_threshold_effect(res)]
    update_metrics("supervised", res)
    print_summary(res)
    print("\nSaved models:", ", ".join(f"models/{model_path(s, sc).name}" for s in SUBJECT_ORDER for sc in SCENARIOS))
    print("Figures:", ", ".join(res["figures"]))
    print(f"Saved metrics.json['supervised']  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
