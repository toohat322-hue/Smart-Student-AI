"""Phase 5 — Student clustering runner (spec §7).

Per subject: k = 2..6 with and without log1p(absences) (inertia + silhouette), choose k,
fit the final log1p -> StandardScaler -> KMeans pipeline, profile and name the clusters,
check where G3 = 0 students land, 2-D PCA. Saves models/kmeans_{subject}.joblib,
reports/metrics.json["clustering"] and figures 14–16.

Usage:
    python scripts/train_clustering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import CLUSTER_FEATURES, FIGURES_DIR, K_RANGE, MODELS_DIR, RANDOM_STATE  # noqa: E402
from src.metrics_io import update_metrics  # noqa: E402
from src.supervised import prepare_subject  # noqa: E402
from src.unsupervised import (  # noqa: E402
    FEATURE_LABELS, K_TOLERANCE, NAME_MAX_TRAITS, NAME_MIN_Z, PREFERRED_K, build_pipeline, choose_k,
    evaluate_k, g3_zero_concentration, model_path, pca_2d, profile_clusters, transformed,
)

SUBJECT_ORDER = ["mat", "por"]
SUBJECT_NAMES = {"mat": "Mathematics", "por": "Portuguese"}
# Reference categorical palette, first slots (fixed order; never cycled). k <= 6.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
C_GRAY, C_TEXT2 = "#8a8984", "#52514e"

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


def run() -> dict:
    res: dict = {
        "config": {
            "features": CLUSTER_FEATURES, "k_range": list(K_RANGE), "random_state": RANDOM_STATE, "n_init": 10,
            "pipeline": "log1p(absences) -> StandardScaler -> KMeans",
            "k_rule": f"best silhouette; if k={PREFERRED_K} is within {K_TOLERANCE} of the best, prefer k={PREFERRED_K}",
            "naming_rule": (f"up to {NAME_MAX_TRAITS} features with |z| >= {NAME_MIN_Z} from the overall mean "
                            "(standardized, after log1p), strongest first; 'High'/'Low' + feature"),
            "grades_used_for_clustering": False,
        },
        "subjects": {},
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for s in SUBJECT_ORDER:
        df = prepare_subject(s)
        X = df[CLUSTER_FEATURES]
        rows_log, rows_raw = evaluate_k(X, log_absences=True), evaluate_k(X, log_absences=False)
        choice = choose_k(rows_log)
        pipe = build_pipeline(choice["k"]).fit(X)
        joblib.dump(pipe, model_path(s))
        labels = pipe[-1].labels_
        res["subjects"][s] = {
            "n_students": len(df),
            "k_selection": {"log1p": rows_log, "raw": rows_raw},
            "choice": choice,
            "k": choice["k"],
            "silhouette": next(r["silhouette"] for r in rows_log if r["k"] == choice["k"]),
            "silhouette_without_log1p_same_k": next(r["silhouette"] for r in rows_raw if r["k"] == choice["k"]),
            "profile": profile_clusters(df, labels, transformed(pipe, X)),
            "g3_zero": g3_zero_concentration(df, labels),
            "pca": pca_2d(pipe, X),
        }
    return res


# --------------------------------------------------------------------------- figures
def fig_k_selection(res: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for i, s in enumerate(SUBJECT_ORDER):
        e = res["subjects"][s]
        for variant, color, ls, label in (("log1p", "#2a78d6", "-", "log1p(absences)"),
                                          ("raw", C_GRAY, "--", "raw absences")):
            rows = e["k_selection"][variant]
            ks = [r["k"] for r in rows]
            axes[i, 0].plot(ks, [r["inertia"] for r in rows], color=color, linestyle=ls, marker="o", label=label)
            axes[i, 1].plot(ks, [r["silhouette"] for r in rows], color=color, linestyle=ls, marker="o", label=label)
        k = e["k"]
        for ax in axes[i]:
            ax.axvline(k, color=C_TEXT2, linestyle=":", linewidth=1)
            ax.set_xticks(list(K_RANGE))
            ax.set_xlabel("k (number of clusters)")
        best = max(e["k_selection"]["log1p"], key=lambda r: r["silhouette"])["silhouette"]
        axes[i, 1].axhspan(best - K_TOLERANCE, best, color="#2a78d6", alpha=0.08, label=f"within {K_TOLERANCE} of best")
        axes[i, 0].set_ylabel(f"{SUBJECT_NAMES[s]}\nInertia (within-cluster SS)")
        axes[i, 1].set_ylabel("Silhouette score")
        axes[i, 0].set_title(f"{SUBJECT_NAMES[s]}: elbow (chosen k = {k})")
        axes[i, 1].set_title(f"{SUBJECT_NAMES[s]}: silhouette (chosen k = {k})")
        axes[i, 0].legend(fontsize=8.5)
        axes[i, 1].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Choosing k for K-Means (dotted line = chosen k)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "14_k_selection.png")


def fig_pca(res: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, s in zip(axes, SUBJECT_ORDER):
        e = res["subjects"][s]
        coords = np.array(e["pca"]["coords"])
        labels = np.array(e["pca"]["labels"])
        for c in e["profile"]["clusters"]:
            m = labels == c["cluster"]
            ax.scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.6, color=CLUSTER_COLORS[c["cluster"]],
                       edgecolors="white", linewidths=0.3, label=f"{c['cluster']}: {c['name']} (n={c['size']})")
        ev = e["pca"]["explained_variance_ratio"]
        ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
        ax.set_title(f"{SUBJECT_NAMES[s]} — k = {e['k']}, silhouette {e['silhouette']:.3f}, "
                     f"PCA explains {e['pca']['explained_variance_total']:.0%}")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, markerscale=1.8)
    fig.suptitle("Student clusters projected on the first two principal components", fontweight="bold")
    fig.tight_layout()
    return save(fig, "15_clusters_pca.png")


def fig_profiles(res: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, s in zip(axes, SUBJECT_ORDER):
        cl = res["subjects"][s]["profile"]["clusters"]
        Z = np.array([[c["feature_z"][f] for f in CLUSTER_FEATURES] for c in cl])
        lim = max(1.5, float(np.abs(Z).max()))
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, f"{Z[i, j]:+.1f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(Z[i, j]) > lim * 0.6 else "#0b0b0b")
        ax.set_xticks(range(len(CLUSTER_FEATURES)), labels=[FEATURE_LABELS[f] for f in CLUSTER_FEATURES], fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(cl)), labels=[f"{c['cluster']}: {c['name']}\n(n={c['size']}, pass {c['pass_rate']:.0%})"
                                              for c in cl], fontsize=8.5)
        ax.grid(False)
        ax.set_title(SUBJECT_NAMES[s])
        fig.colorbar(im, ax=ax, fraction=0.035, label="SD from overall mean")
    fig.suptitle("Cluster profiles: mean of each feature in standardized units (after log1p on absences)",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, "16_cluster_profiles.png")


# --------------------------------------------------------------------------- summary
def print_summary(res: dict) -> None:
    for s in SUBJECT_ORDER:
        e = res["subjects"][s]
        print(f"\n=== {SUBJECT_NAMES[s]} (n={e['n_students']}) ===")
        print(f"  {'k':>2s} {'sil log1p':>10s} {'sil raw':>8s} {'inertia log1p':>14s} {'min size log1p':>15s}")
        for a, b in zip(e["k_selection"]["log1p"], e["k_selection"]["raw"]):
            mark = "  <== chosen" if a["k"] == e["k"] else ""
            print(f"  {a['k']:2d} {a['silhouette']:10.3f} {b['silhouette']:8.3f} {a['inertia']:14.1f} "
                  f"{a['min_cluster_size']:15d}{mark}")
        print(f"  choice: {e['choice']['reason']}  (within tolerance: {e['choice']['within_tolerance']})")
        print(f"  PCA explained variance: {e['pca']['explained_variance_ratio']} "
              f"(total {e['pca']['explained_variance_total']:.1%})")
        p = e["profile"]
        hdr = "  " + f"{'cluster':38s} {'n':>4s} " + " ".join(f"{f[:9]:>9s}" for f in CLUSTER_FEATURES) + \
              f" {'mean G3':>8s} {'pass':>6s} {'G3=0':>5s}"
        print(hdr)
        for c in p["clusters"]:
            print(f"  {str(c['cluster']) + ': ' + c['name']:38s} {c['size']:4d} "
                  + " ".join(f"{c['feature_means'][f]:9.2f}" for f in CLUSTER_FEATURES)
                  + f" {c['mean_G3']:8.2f} {c['pass_rate']:6.1%} {c['n_g3_zero']:5d}")
        print(f"  {'overall':38s} {e['n_students']:4d} "
              + " ".join(f"{p['overall_means'][f]:9.2f}" for f in CLUSTER_FEATURES)
              + f" {p['overall_mean_G3']:8.2f} {p['overall_pass_rate']:6.1%} {e['g3_zero']['n_g3_zero']:5d}")
        g = e["g3_zero"]
        print(f"  G3=0: {g['counts_by_cluster']} by cluster; {g['share_in_top_cluster']:.0%} in cluster "
              f"{g['top_cluster']} (which holds {g['top_cluster_share_of_all_students']:.0%} of students, "
              f"x{g['over_representation']}); piled up: {g['piled_up']}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    res["figures"] = [fig_k_selection(res), fig_pca(res), fig_profiles(res)]
    update_metrics("clustering", res)
    print_summary(res)
    print("\nSaved:", ", ".join(f"models/{model_path(s).name}" for s in SUBJECT_ORDER), "| figures:",
          ", ".join(res["figures"]), "| metrics.json['clustering']")
    return 0


if __name__ == "__main__":
    sys.exit(main())
