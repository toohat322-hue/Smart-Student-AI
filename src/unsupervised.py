"""Unsupervised learning (spec §7): K-Means clustering of study / lifestyle behaviour.

Pipeline per subject: log1p(absences) -> StandardScaler -> KMeans(n_init=10, random_state=42).
Grades are NOT features; G3 / pass are only joined afterwards to describe the clusters.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import CLUSTER_FEATURES, K_RANGE, MODELS_DIR, RANDOM_STATE, SUBJECTS
from src.data import TARGET_COL
from src.metrics_io import load_metrics
from src.supervised import InputValidationError, validate_input

K_TOLERANCE = 0.02       # silhouettes within this of the best count as a tie
PREFERRED_K = 3          # tie-break: one cluster per RL learner profile (spec §8)
NAME_MIN_Z = 0.5         # a feature must differ by >= 0.5 SD from the overall mean to name a cluster
NAME_MAX_TRAITS = 2
FEATURE_LABELS = {
    "studytime": "study time", "absences": "absences", "failures": "past failures",
    "freetime": "free time", "health": "health", "goout": "going out",
}


# --------------------------------------------------------------------------- pipeline
def log1p_absences(X: pd.DataFrame) -> pd.DataFrame:
    """absences is heavily right-skewed (0–93); log1p compresses the tail before scaling."""
    X = X.copy()
    X["absences"] = np.log1p(X["absences"].astype(float))
    return X


def build_pipeline(k: int, log_absences: bool = True) -> Pipeline:
    steps = []
    if log_absences:
        steps.append(("log", FunctionTransformer(log1p_absences, feature_names_out="one-to-one")))
    steps += [("scale", StandardScaler()),
              ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE))]
    return Pipeline(steps)


def transformed(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Data in the space K-Means sees (after log1p + scaling)."""
    return pipe[:-1].transform(X)


# --------------------------------------------------------------------------- k selection
def evaluate_k(X: pd.DataFrame, log_absences: bool = True) -> list[dict[str, Any]]:
    rows = []
    for k in K_RANGE:
        pipe = build_pipeline(k, log_absences).fit(X)
        Z = transformed(pipe, X)
        labels = pipe[-1].labels_
        rows.append({
            "k": k,
            "inertia": round(float(pipe[-1].inertia_), 2),
            "silhouette": round(float(silhouette_score(Z, labels)), 4),
            "min_cluster_size": int(np.bincount(labels).min()),
        })
    return rows


def choose_k(rows: list[dict[str, Any]], tol: float = K_TOLERANCE, preferred: int = PREFERRED_K) -> dict[str, Any]:
    """Best silhouette, unless PREFERRED_K is within `tol` of it (then prefer it: interpretability)."""
    best = max(rows, key=lambda r: r["silhouette"])
    ties = [r["k"] for r in rows if best["silhouette"] - r["silhouette"] <= tol + 1e-12]
    if preferred in ties and preferred != best["k"]:
        k, reason = preferred, (f"k={best['k']} has the best silhouette ({best['silhouette']:.3f}) but k={preferred} "
                                f"is within {tol} ({next(r['silhouette'] for r in rows if r['k'] == preferred):.3f}); "
                                f"k={preferred} is preferred for interpretability (one cluster per RL learner profile).")
    else:
        k, reason = best["k"], f"best silhouette ({best['silhouette']:.3f})."
    return {"k": k, "best_silhouette_k": best["k"], "within_tolerance": ties, "tolerance": tol, "reason": reason}


# --------------------------------------------------------------------------- profiles & names
def cluster_names(z_means: pd.DataFrame) -> dict[int, str]:
    """Name each cluster by its features furthest from the overall mean (standardized units).

    Up to NAME_MAX_TRAITS features with |z| >= NAME_MIN_Z, strongest first; more traits are
    added only if needed to make names unique.
    """
    def name_for(c: int, n_traits: int) -> str:
        z = z_means.loc[c].reindex(z_means.columns)
        order = z.abs().sort_values(ascending=False).index
        traits = [f"{'High' if z[f] > 0 else 'Low'} {FEATURE_LABELS[f]}" for f in order
                  if abs(z[f]) >= NAME_MIN_Z][:n_traits]
        return " · ".join(traits) if traits else "Close to average"

    names = {c: name_for(c, NAME_MAX_TRAITS) for c in z_means.index}
    for n in range(NAME_MAX_TRAITS + 1, len(z_means.columns) + 1):
        if len(set(names.values())) == len(names):
            break
        dup = {v for v in names.values() if list(names.values()).count(v) > 1}
        names.update({c: name_for(c, n) for c, v in names.items() if v in dup})
    for c, v in list(names.items()):  # still duplicated (e.g. two "Close to average"): add the id
        if list(names.values()).count(v) > 1:
            names[c] = f"{v} ({c})"
    return names


def profile_clusters(df: pd.DataFrame, labels: np.ndarray, Z: np.ndarray) -> dict[str, Any]:
    X = df[CLUSTER_FEATURES]
    k = int(labels.max()) + 1
    z_means = pd.DataFrame(Z, columns=CLUSTER_FEATURES).groupby(labels).mean()  # overall mean = 0 in Z space
    names = cluster_names(z_means)
    g3_zero = (df["G3"] == 0).to_numpy()
    profiles = []
    for c in range(k):
        m = labels == c
        profiles.append({
            "cluster": c,
            "name": names[c],
            "size": int(m.sum()),
            "share": round(float(m.mean()), 4),
            "feature_means": {f: round(float(X.loc[m, f].mean()), 3) for f in CLUSTER_FEATURES},
            "feature_z": {f: round(float(z_means.loc[c, f]), 3) for f in CLUSTER_FEATURES},
            # after the fact (NOT used for clustering)
            "mean_G3": round(float(df.loc[m, "G3"].mean()), 3),
            "pass_rate": round(float(df.loc[m, TARGET_COL].mean()), 4),
            "n_g3_zero": int((m & g3_zero).sum()),
        })
    return {
        "overall_means": {f: round(float(X[f].mean()), 3) for f in CLUSTER_FEATURES},
        "overall_mean_G3": round(float(df["G3"].mean()), 3),
        "overall_pass_rate": round(float(df[TARGET_COL].mean()), 4),
        "clusters": profiles,
    }


def g3_zero_concentration(df: pd.DataFrame, labels: np.ndarray, threshold: float = 0.6) -> dict[str, Any]:
    """Where do G3 = 0 students land? 'Piled up' if one cluster holds >= threshold of them."""
    z = (df["G3"] == 0).to_numpy()
    n = int(z.sum())
    counts = np.bincount(labels[z], minlength=int(labels.max()) + 1)
    top = int(counts.argmax())
    share_top = float(counts[top] / n) if n else 0.0
    cluster_share = float((labels == top).mean())
    return {
        "n_g3_zero": n,
        "counts_by_cluster": counts.tolist(),
        "top_cluster": top,
        "share_in_top_cluster": round(share_top, 4),
        "top_cluster_share_of_all_students": round(cluster_share, 4),
        "over_representation": round(share_top / cluster_share, 2) if cluster_share else None,
        "share_of_top_cluster_that_is_g3_zero": round(float(counts[top] / (labels == top).sum()), 4),
        "piled_up": bool(share_top >= threshold),
        "rule": f"piled up if one cluster holds >= {threshold:.0%} of the G3 = 0 students",
    }


def pca_2d(pipe: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
    Z = transformed(pipe, X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
    coords = pca.transform(Z)
    return {
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "explained_variance_total": round(float(pca.explained_variance_ratio_.sum()), 4),
        "loadings": {f"PC{i + 1}": {f: round(float(w), 3) for f, w in zip(CLUSTER_FEATURES, comp)}
                     for i, comp in enumerate(pca.components_)},
        "coords": np.round(coords, 4).tolist(),
        "labels": pipe[-1].labels_.tolist(),
    }


def model_path(subject: str):
    return MODELS_DIR / f"kmeans_{subject}.joblib"


# --------------------------------------------------------------------------- assignment
@lru_cache(maxsize=None)
def load_kmeans(subject: str) -> Pipeline:
    path = model_path(subject)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/train_clustering.py` first.")
    return joblib.load(path)


def assign_cluster(subject: str, input_dict: dict[str, Any]) -> dict[str, Any]:
    """Assign one student (the 6 CLUSTER_FEATURES) to a cluster of the given subject.

    Returns {"cluster", "name", "profile"}; profile is that cluster's entry from metrics.json
    (size = "you are similar to N students", feature means, and after-the-fact G3 / pass rate).
    """
    if subject not in SUBJECTS:
        raise InputValidationError(f"Unknown subject {subject!r}; expected one of {sorted(SUBJECTS)}")
    X = validate_input(input_dict, CLUSTER_FEATURES)
    pipe = load_kmeans(subject)
    c = int(pipe.predict(X)[0])
    try:
        profiles = load_metrics()["clustering"]["subjects"][subject]["profile"]["clusters"]
    except KeyError as exc:
        raise FileNotFoundError("reports/metrics.json has no clustering results. "
                                "Run `python scripts/train_clustering.py`.") from exc
    profile = next(p for p in profiles if p["cluster"] == c)
    return {"cluster": c, "name": profile["name"], "profile": profile}

