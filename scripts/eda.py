"""Phase 3 — Exploratory data analysis.

Generates PNG figures into reports/figures/ and writes key numbers to
reports/metrics.json["eda"] (other keys in metrics.json are preserved).

Usage:
    python scripts/eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import FIGURES_DIR, PASS_THRESHOLD, RANDOM_STATE  # noqa: E402
from src.data import TARGET_COL, add_target, load_subject, validate_schema  # noqa: E402
from src.metrics_io import update_metrics  # noqa: E402

SUBJECT_ORDER = ["mat", "por"]
SUBJECT_NAMES = {"mat": "Mathematics", "por": "Portuguese"}

# Reference categorical palette (dataviz skill), slots 1–2: CVD-safe pair.
C_PASS = "#2a78d6"      # Likely to Pass
C_RISK = "#eb6834"      # At Risk
C_NEUTRAL = "#52514e"   # reference lines / annotations
C_GRID = "#e4e3df"
LABELS = {1: "Likely to Pass (G3 ≥ 10)", 0: "At Risk (G3 < 10)"}
COLORS = {1: C_PASS, 0: C_RISK}

# Numeric + ordinal columns (spec §2.3). Binary/nominal columns are excluded.
NUMERIC_ORDINAL = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2",
]
ABSENCE_BINS = [-0.5, 0.5, 5.5, 10.5, 20.5, np.inf]
ABSENCE_LABELS = ["0", "1–5", "6–10", "11–20", ">20"]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8984",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- figures
def fig_g3_distribution(dfs: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    edges = np.arange(0, 22)  # bin [g, g+1) holds grade g, so x=10 is the pass boundary
    for ax, s in zip(axes, SUBJECT_ORDER):
        g3 = dfs[s]["G3"]
        counts, _ = np.histogram(g3, bins=edges)
        colors = [C_RISK if g < PASS_THRESHOLD else C_PASS for g in edges[:-1]]
        ax.bar(edges[:-1], counts, width=0.9, align="edge", color=colors)
        ax.axvline(PASS_THRESHOLD, color=C_NEUTRAL, linestyle="--", linewidth=1.5)
        ax.text(PASS_THRESHOLD - 0.3, counts.max() * 0.97, f"pass threshold = {PASS_THRESHOLD}",
                color=C_NEUTRAL, va="top", ha="right", fontsize=9)
        ax.set_xticks(edges[:-1] + 0.45, labels=[str(g) for g in edges[:-1]], fontsize=8)
        ax.set_title(f"{SUBJECT_NAMES[s]} (n = {len(g3)})")
        ax.set_xlabel("Final grade G3 (0–20)")
        ax.set_ylabel("Number of students")
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in (1, 0)]
    fig.legend(handles, [LABELS[1], LABELS[0]], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Distribution of final grade G3 by subject", y=1.08, fontweight="bold")
    return save(fig, "01_g3_distribution.png")


def fig_pass_counts(dfs: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(SUBJECT_ORDER))
    width = 0.38
    for i, k in enumerate((1, 0)):
        counts = [int((dfs[s][TARGET_COL] == k).sum()) for s in SUBJECT_ORDER]
        totals = [len(dfs[s]) for s in SUBJECT_ORDER]
        xs = x + (i - 0.5) * (width + 0.02)
        ax.bar(xs, counts, width=width, color=COLORS[k], label=LABELS[k])
        for xi, c, t in zip(xs, counts, totals):
            ax.text(xi, c + 6, f"{c}\n({c / t:.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels=[f"{SUBJECT_NAMES[s]}\n(n = {len(dfs[s])})" for s in SUBJECT_ORDER])
    ax.set_ylabel("Number of students")
    ax.set_ylim(0, max(len(d) for d in dfs.values()) * 1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Likely to Pass vs At Risk by subject")
    return save(fig, "02_pass_vs_at_risk_counts.png")


def correlations_with_g3(df: pd.DataFrame) -> pd.Series:
    return df[NUMERIC_ORDINAL].corrwith(df["G3"]).sort_values()


def fig_correlations(corrs: dict[str, pd.Series]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, s in zip(axes, SUBJECT_ORDER):
        c = corrs[s]
        ax.barh(c.index, c.values, color=[C_PASS if v >= 0 else C_RISK for v in c.values], height=0.7)
        for i, v in enumerate(c.values):
            ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color=C_NEUTRAL, linewidth=1)
        ax.set_xlim(-0.55, 1.05)
        ax.set_title(SUBJECT_NAMES[s])
        ax.set_xlabel("Pearson correlation with G3")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Correlation of numeric / ordinal features with final grade G3\n"
                 "(blue = positive, orange = negative)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "03_correlation_with_g3.png")


def pass_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (df.groupby(col, observed=True)[TARGET_COL]
              .agg(pass_rate="mean", n="size")
              .reset_index())


def fig_pass_rate_factors(dfs: dict[str, pd.DataFrame]) -> tuple[Path, dict]:
    factors = [
        ("studytime", "Weekly study time category",
         {1: "<2h", 2: "2–5h", 3: "5–10h", 4: ">10h"}),
        ("failures", "Past class failures", {0: "0", 1: "1", 2: "2", 3: "3+"}),
        ("absence_bin", "Number of absences", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    tables: dict = {}
    for r, s in enumerate(SUBJECT_ORDER):
        df = dfs[s].assign(absence_bin=pd.cut(dfs[s]["absences"], ABSENCE_BINS, labels=ABSENCE_LABELS))
        overall = df[TARGET_COL].mean()
        tables[s] = {}
        for c, (col, xlabel, names) in enumerate(factors):
            ax = axes[r, c]
            t = pass_rate_table(df, col)
            labels = [names.get(v, str(v)) if names else str(v) for v in t[col]]
            ax.bar(labels, t["pass_rate"] * 100, color=C_PASS, width=0.6)
            for i, (pr, n) in enumerate(zip(t["pass_rate"], t["n"])):
                ax.text(i, pr * 100 + 2, f"{pr:.0%}\nn={n}", ha="center", va="bottom", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
            ax.axhline(overall * 100, color=C_NEUTRAL, linestyle="--", linewidth=1, zorder=0)
            ax.set_ylim(0, 118)
            ax.set_yticks(range(0, 101, 20))
            ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(f"{SUBJECT_NAMES[s]} (overall {overall:.0%})\nPass rate (%)")
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(f"Pass rate by {xlabel.lower()}")
            tables[s][col] = {
                (names.get(v, str(v)) if names else str(v)): {"pass_rate": round(float(pr), 4), "n": int(n)}
                for v, pr, n in zip(t[col], t["pass_rate"], t["n"])
            }
    fig.suptitle("Pass rate by study time, past failures and absences "
                 "(dashed line = subject's overall pass rate)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "04_pass_rate_by_factors.png"), tables


def fig_grade_scatter(dfs: dict[str, pd.DataFrame]) -> Path:
    rng = np.random.default_rng(RANDOM_STATE)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for r, s in enumerate(SUBJECT_ORDER):
        df = dfs[s]
        for c, g in enumerate(["G1", "G2"]):
            ax = axes[r, c]
            for k in (0, 1):
                d = df[df[TARGET_COL] == k]
                jx = rng.uniform(-0.25, 0.25, len(d))
                jy = rng.uniform(-0.25, 0.25, len(d))
                ax.scatter(d[g] + jx, d["G3"] + jy, s=16, alpha=0.6, color=COLORS[k],
                           edgecolors="white", linewidths=0.4, label=LABELS[k])
            ax.plot([0, 20], [0, 20], color=C_NEUTRAL, linewidth=0.8, linestyle=":")
            ax.axhline(PASS_THRESHOLD - 0.5, color=C_NEUTRAL, linewidth=1, linestyle="--")
            rho = df[g].corr(df["G3"])
            ax.set_title(f"{SUBJECT_NAMES[s]}: {g} vs G3 (r = {rho:.2f})")
            ax.set_xlabel(f"{g} (period {g[1]} grade)")
            ax.set_ylabel("G3 (final grade)")
            ax.set_xlim(-1, 21)
            ax.set_ylim(-1, 21)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               markerscale=2)
    fig.suptitle("Period grades vs final grade (points jittered ±0.25; dotted = y = x, "
                 "dashed = pass threshold)", y=1.03, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, "05_g1_g2_vs_g3.png")


# --------------------------------------------------------------------------- G3 = 0 check
def g3_zero_check(df: pd.DataFrame) -> dict:
    at_risk = df[df[TARGET_COL] == 0]
    zero = at_risk[at_risk["G3"] == 0]
    other = at_risk[at_risk["G3"] > 0]

    def describe(d: pd.DataFrame) -> dict:
        return {
            "n": len(d),
            "G1_mean": round(d["G1"].mean(), 2) if len(d) else None,
            "G2_mean": round(d["G2"].mean(), 2) if len(d) else None,
            "absences_mean": round(d["absences"].mean(), 2) if len(d) else None,
            "absences_median": float(d["absences"].median()) if len(d) else None,
            "share_G2_eq_0": round((d["G2"] == 0).mean(), 4) if len(d) else None,
            "share_absences_eq_0": round((d["absences"] == 0).mean(), 4) if len(d) else None,
            "share_G1_ge_10": round((d["G1"] >= PASS_THRESHOLD).mean(), 4) if len(d) else None,
            "share_G2_ge_8": round((d["G2"] >= 8).mean(), 4) if len(d) else None,
            "share_failures_gt_0": round((d["failures"] > 0).mean(), 4) if len(d) else None,
        }

    abs0 = df[df["absences"] == 0]
    g2_zero = df[df["G2"] == 0]
    return {
        "n_absences_eq_0": len(abs0),
        "share_of_absences_eq_0_with_G3_eq_0": round((abs0["G3"] == 0).mean(), 4),
        "pass_rate_absences_eq_0": round(abs0[TARGET_COL].mean(), 4),
        "pass_rate_absences_eq_0_excl_g3_zero": round(abs0.loc[abs0["G3"] > 0, TARGET_COL].mean(), 4),
        "n_G2_eq_0": len(g2_zero),
        "all_G2_eq_0_have_G3_eq_0": bool((g2_zero["G3"] == 0).all()),
        "g3_zero": describe(zero),
        "other_at_risk": describe(other),
        "g3_zero_G1_distribution": zero["G1"].value_counts().sort_index().to_dict(),
        "g3_zero_absences_distribution": zero["absences"].value_counts().sort_index().to_dict(),
        "corr_absences_G3_all": round(df["absences"].corr(df["G3"]), 4),
        "corr_absences_G3_excl_g3_zero": round(df.loc[df["G3"] > 0, "absences"].corr(
            df.loc[df["G3"] > 0, "G3"]), 4),
    }


def fig_g3_zero(dfs: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    c_zero, c_other = "#6250d6", C_RISK  # violet (slot 7) vs orange; groups also named on x-axis
    for r, s in enumerate(SUBJECT_ORDER):
        at_risk = dfs[s][dfs[s][TARGET_COL] == 0]
        groups = {"G3 = 0": at_risk[at_risk["G3"] == 0], "G3 1–9": at_risk[at_risk["G3"] > 0]}
        for c, col in enumerate(["G1", "G2", "absences"]):
            ax = axes[r, c]
            data = [g[col].to_numpy() for g in groups.values()]
            bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                            medianprops={"color": "#0b0b0b", "linewidth": 1.5},
                            flierprops={"markersize": 4, "markeredgecolor": C_NEUTRAL})
            for patch, color in zip(bp["boxes"], (c_zero, c_other)):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticks([1, 2], labels=[f"{k}\n(n={len(g)})" for k, g in groups.items()])
            ax.set_ylabel(col if c else f"{SUBJECT_NAMES[s]}\n{col}")
            if r == 0:
                ax.set_title(f"{col} of At-Risk students")
            ax.grid(axis="x", visible=False)
    fig.suptitle("G3 = 0 students vs other At-Risk students (G3 1–9)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "06_g3_zero_check.png")


# --------------------------------------------------------------------------- main
def main() -> int:
    dfs: dict[str, pd.DataFrame] = {}
    for s in SUBJECT_ORDER:
        df = add_target(load_subject(s))
        validate_schema(df)
        dfs[s] = df

    corrs = {s: correlations_with_g3(dfs[s]) for s in SUBJECT_ORDER}
    figures = [
        fig_g3_distribution(dfs),
        fig_pass_counts(dfs),
        fig_correlations(corrs),
    ]
    p, pass_rate_tables = fig_pass_rate_factors(dfs)
    figures += [p, fig_grade_scatter(dfs), fig_g3_zero(dfs)]

    eda: dict = {"pass_threshold": PASS_THRESHOLD, "subjects": {}}
    for s in SUBJECT_ORDER:
        df, c = dfs[s], corrs[s]
        ranked = c.reindex(c.abs().sort_values(ascending=False).index)
        no_grades = ranked.drop(["G1", "G2"])
        eda["subjects"][s] = {
            "rows": len(df),
            "columns": df.shape[1] - 1,  # excluding the derived `pass`
            "n_pass": int(df[TARGET_COL].sum()),
            "n_at_risk": int((df[TARGET_COL] == 0).sum()),
            "pass_rate": round(df[TARGET_COL].mean(), 4),
            "g3_zero_count": int((df["G3"] == 0).sum()),
            "g3_zero_share": round((df["G3"] == 0).mean(), 4),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": int(df.isna().sum().sum()),
            "top5_corr_with_g3": {k: round(v, 4) for k, v in ranked.head(5).items()},
            "top5_corr_with_g3_excl_G1_G2": {k: round(v, 4) for k, v in no_grades.head(5).items()},
            "corr_with_g3": {k: round(v, 4) for k, v in c.sort_values(ascending=False).items()},
            "pass_rate_by": pass_rate_tables[s],
            "g3_zero_check": g3_zero_check(df),
        }
    eda["figures"] = [str(p.relative_to(ROOT_DIR)).replace("\\", "/") for p in figures]
    update_metrics("eda", eda)

    # ----------------------------------------------------------------- console summary
    print("=" * 72)
    print("EDA SUMMARY")
    print("=" * 72)
    for s in SUBJECT_ORDER:
        e = eda["subjects"][s]
        z = e["g3_zero_check"]
        print(f"\n[{s}] {SUBJECT_NAMES[s]}: {e['rows']} rows x {e['columns']} cols, "
              f"missing={e['missing_values']}, duplicates={e['duplicate_rows']}")
        print(f"  pass rate {e['pass_rate']:.1%}  (pass {e['n_pass']} / at risk {e['n_at_risk']})")
        print(f"  G3 = 0: {e['g3_zero_count']} ({e['g3_zero_share']:.1%} of all, "
              f"{e['g3_zero_count'] / e['n_at_risk']:.0%} of At Risk)")
        print("  top 5 |corr| with G3:           "
              + ", ".join(f"{k} {v:+.2f}" for k, v in e["top5_corr_with_g3"].items()))
        print("  top 5 |corr| excl. G1/G2:       "
              + ", ".join(f"{k} {v:+.2f}" for k, v in e["top5_corr_with_g3_excl_G1_G2"].items()))
        print("  G3=0 vs other At-Risk (G3 1-9):")
        for label, key in (("G3 = 0", "g3_zero"), ("G3 1-9", "other_at_risk")):
            d = z[key]
            print(f"    {label:7s} n={d['n']:3d}  G1 mean {d['G1_mean']:5.2f}  G2 mean {d['G2_mean']:5.2f}"
                  f"  G2=0 {d['share_G2_eq_0']:6.1%}  absences mean {d['absences_mean']:5.2f}"
                  f" (median {d['absences_median']:g})  absences=0 {d['share_absences_eq_0']:6.1%}"
                  f"  G1>=10 {d['share_G1_ge_10']:6.1%}")
        print(f"    G3=0 G1 values: {z['g3_zero_G1_distribution']}")
        print(f"    corr(absences, G3): all {z['corr_absences_G3_all']:+.3f}  |  "
              f"excluding G3=0 {z['corr_absences_G3_excl_g3_zero']:+.3f}")
    print("\nFigures:")
    for p in eda["figures"]:
        print(f"  {p}")
    print("\nSaved metrics.json['eda']")
    return 0


if __name__ == "__main__":
    sys.exit(main())
