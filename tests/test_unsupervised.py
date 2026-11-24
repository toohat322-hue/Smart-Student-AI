"""Tests for src/unsupervised.py (spec §10: test_unsupervised.py)."""
import numpy as np
import pandas as pd
import pytest

from src.config import CLUSTER_FEATURES, GRADE_COLS, SUBJECTS
from src.metrics_io import load_metrics
from src.supervised import InputValidationError, prepare_subject
from src.unsupervised import (
    K_TOLERANCE,
    assign_cluster,
    build_pipeline,
    choose_k,
    cluster_names,
    evaluate_k,
    fit_pipeline,
    load_kmeans,
    log1p_absences,
    model_path,
)

SUBJ = sorted(SUBJECTS)


@pytest.fixture(scope="module")
def data():
    return {s: prepare_subject(s) for s in SUBJ}


@pytest.fixture(scope="module")
def meta():
    return load_metrics()["clustering"]


def student(df, i=0):
    return {f: int(df.iloc[i][f]) for f in CLUSTER_FEATURES}


def test_cluster_features_exclude_grades():
    assert not set(CLUSTER_FEATURES) & set(GRADE_COLS)
    assert CLUSTER_FEATURES == ["studytime", "absences", "failures", "freetime", "health", "goout"]


def test_log1p_applied_only_to_absences():
    X = pd.DataFrame({f: [0, 9] for f in CLUSTER_FEATURES})
    out = log1p_absences(X)
    assert np.allclose(out["absences"], np.log1p([0, 9]))
    assert all((out[f] == X[f]).all() for f in CLUSTER_FEATURES if f != "absences")
    assert (X["absences"] == [0, 9]).all()  # input not mutated


@pytest.mark.parametrize("subject", SUBJ)
def test_saved_pipeline_uses_log1p_scaler_kmeans(subject):
    pipe = load_kmeans(subject)
    assert [name for name, _ in pipe.steps] == ["log", "scale", "kmeans"]
    assert model_path(subject).is_file()


@pytest.mark.parametrize("subject", SUBJ)
def test_label_count_equals_chosen_k_and_clusters_non_empty(subject, data, meta):
    k = meta["subjects"][subject]["k"]
    labels = load_kmeans(subject).predict(data[subject][CLUSTER_FEATURES])
    assert len(np.unique(labels)) == k == load_kmeans(subject)[-1].n_clusters
    assert np.bincount(labels, minlength=k).min() > 0
    sizes = [c["size"] for c in meta["subjects"][subject]["profile"]["clusters"]]
    assert sizes == np.bincount(labels, minlength=k).tolist() and sum(sizes) == len(data[subject])


@pytest.mark.parametrize("subject", SUBJ)
def test_silhouettes_in_range(subject, meta):
    e = meta["subjects"][subject]
    for variant in ("log1p", "raw"):
        assert [r["k"] for r in e["k_selection"][variant]] == list(range(2, 7))
        assert all(-1 <= r["silhouette"] <= 1 for r in e["k_selection"][variant])
    assert -1 <= e["silhouette"] <= 1


@pytest.mark.parametrize("subject", SUBJ)
def test_deterministic_with_fixed_seed(subject, data, meta):
    X = data[subject][CLUSTER_FEATURES]
    k = meta["subjects"][subject]["k"]
    a = build_pipeline(k).fit(X)
    b = build_pipeline(k).fit(X)
    assert np.array_equal(a[-1].labels_, b[-1].labels_)
    assert np.allclose(a[-1].cluster_centers_, b[-1].cluster_centers_)
    assert np.array_equal(a.predict(X), load_kmeans(subject).predict(X))  # saved model reproducible
    assert evaluate_k(X) == meta["subjects"][subject]["k_selection"]["log1p"]



@pytest.mark.parametrize("subject", SUBJ)
def test_centroids_bit_identical_across_fits(subject, data, meta):
    """Single-threaded fit => centroids identical to the last bit (they feed the RL environments)."""
    X = data[subject][CLUSTER_FEATURES]
    k = meta["subjects"][subject]["k"]
    fits = [fit_pipeline(k, X)[-1].cluster_centers_ for _ in range(3)]
    assert all(np.array_equal(f, fits[0]) for f in fits)
    assert np.array_equal(fits[0], load_kmeans(subject)[-1].cluster_centers_)


def test_choose_k_rule():
    rows = [{"k": 2, "silhouette": 0.20}, {"k": 3, "silhouette": 0.195}, {"k": 4, "silhouette": 0.21}]
    assert choose_k(rows)["k"] == 3  # 3 within tolerance of best (4)
    rows[1]["silhouette"] = 0.21 - K_TOLERANCE - 0.001
    assert choose_k(rows)["k"] == 4  # 3 just outside tolerance -> best wins


def test_cluster_names_from_rule():
    z = pd.DataFrame({"studytime": [1.2, -0.1, 0.0], "absences": [0.0, 0.9, 0.1], "failures": [0.0, -0.7, 0.2],
                      "freetime": [0.0, 0.0, 0.1], "health": [0.0, 0.0, 0.0], "goout": [0.0, 0.0, -0.2]})
    names = cluster_names(z)
    assert names[0] == "High study time"
    assert names[1] == "High absences · Low past failures"
    assert names[2] == "Close to average"


@pytest.mark.parametrize("subject", SUBJ)
def test_saved_names_unique_and_match_rule(subject, meta):
    clusters = meta["subjects"][subject]["profile"]["clusters"]
    names = [c["name"] for c in clusters]
    assert len(set(names)) == len(names)
    z = pd.DataFrame([c["feature_z"] for c in clusters])
    assert list(cluster_names(z).values()) == names


@pytest.mark.parametrize("subject", SUBJ)
def test_assign_cluster(subject, data):
    df = data[subject]
    out = assign_cluster(subject, student(df, 3))
    assert set(out) == {"cluster", "name", "profile"}
    assert out["cluster"] == int(load_kmeans(subject).predict(df[CLUSTER_FEATURES].iloc[[3]])[0])
    assert out["profile"]["cluster"] == out["cluster"] and out["profile"]["name"] == out["name"]
    assert out["profile"]["size"] > 0


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: {k: v for k, v in d.items() if k != "goout"}, "missing fields"),
        (lambda d: {**d, "absences": 200}, "out of range"),
        (lambda d: {**d, "health": 0}, "out of range"),
        (lambda d: {**d, "studytime": 1.5}, "whole number"),
        (lambda d: {**d, "hobby": "chess"}, "unknown fields"),
    ],
    ids=["missing", "absences-high", "health-low", "float", "unknown"],
)
def test_assign_cluster_rejects_bad_input(mutate, message, data):
    with pytest.raises(InputValidationError, match=message):
        assign_cluster("por", mutate(student(data["por"])))


def test_assign_cluster_unknown_subject():
    with pytest.raises(InputValidationError):
        assign_cluster("bio", {})


@pytest.mark.parametrize("subject", SUBJ)
def test_g3_zero_and_pca_reported(subject, data, meta):
    e = meta["subjects"][subject]
    g = e["g3_zero"]
    assert g["n_g3_zero"] == int((data[subject]["G3"] == 0).sum()) == sum(g["counts_by_cluster"])
    assert sum(c["n_g3_zero"] for c in e["profile"]["clusters"]) == g["n_g3_zero"]
    ev = e["pca"]["explained_variance_ratio"]
    assert len(ev) == 2 and 0 < sum(ev) <= 1
    assert len(e["pca"]["coords"]) == len(data[subject])
