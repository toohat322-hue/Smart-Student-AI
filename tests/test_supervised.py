"""Tests for src/preprocessing.py and src/supervised.py (spec §10: test_supervised.py).

Uses the committed pipelines in models/ (produced by scripts/train_supervised.py).
"""
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from src.config import SUBJECTS
from src.data import TARGET_COL
from src.preprocessing import SCENARIOS, build_preprocessor, feature_columns
from src.supervised import (
    InputValidationError,
    load_model,
    model_path,
    predict,
    prepare_subject,
    proba_pass,
    split_indices,
)

CASES = [(s, sc) for s in sorted(SUBJECTS) for sc in SCENARIOS]


@pytest.fixture(scope="module")
def data():
    out = {}
    for s in SUBJECTS:
        df = prepare_subject(s)
        out[s] = (df, *split_indices(df))
    return out


def student(df, scenario, i=0):
    """A real row as a plain-Python dict with exactly the scenario's features."""
    row = df.iloc[i][feature_columns(scenario)]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


# --------------------------------------------------------------------------- feature sets
def test_early_excludes_grades_and_sex():
    early = feature_columns("early")
    for col in ("G1", "G2", "G3", "sex"):
        assert col not in early
    assert len(early) == 33 - 4


def test_midterm_is_early_plus_g1_g2_and_never_g3():
    mid = feature_columns("midterm")
    assert "G3" not in mid and "sex" not in mid
    assert set(mid) == set(feature_columns("early")) | {"G1", "G2"}


def test_unknown_scenario():
    with pytest.raises(ValueError):
        feature_columns("final")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_preprocessor_covers_every_feature(scenario, data):
    df, train_idx, _ = data["por"]
    Xt = build_preprocessor(feature_columns(scenario)).fit_transform(df[feature_columns(scenario)])
    assert Xt.shape[0] == len(df)
    assert not np.isnan(np.asarray(Xt, dtype=float)).any()


# --------------------------------------------------------------------------- split
@pytest.mark.parametrize("subject", sorted(SUBJECTS))
def test_split_is_stratified_and_deterministic(subject, data):
    df, train_idx, test_idx = data[subject]
    tr2, te2 = split_indices(df)
    assert np.array_equal(train_idx, tr2) and np.array_equal(test_idx, te2)
    assert not set(train_idx) & set(test_idx)
    assert len(test_idx) == round(0.2 * len(df))
    y = df[TARGET_COL]
    assert abs(y.iloc[train_idx].mean() - y.iloc[test_idx].mean()) < 0.02


# --------------------------------------------------------------------------- saved pipelines
@pytest.mark.parametrize("subject,scenario", CASES)
def test_saved_pipeline_predicts_single_dict(subject, scenario, data):
    assert model_path(subject, scenario).is_file()
    df, _, _ = data[subject]
    import pandas as pd

    X = pd.DataFrame([student(df, scenario)])
    model = load_model(subject, scenario)
    assert model.predict(X).shape == (1,)
    p = model.predict_proba(X)
    assert p.shape == (1, 2) and np.allclose(p.sum(axis=1), 1)


@pytest.mark.parametrize("subject,scenario", CASES)
def test_probabilities_in_unit_interval(subject, scenario, data):
    df, _, test_idx = data[subject]
    p = proba_pass(load_model(subject, scenario), df[feature_columns(scenario)].iloc[test_idx])
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("subject,scenario", CASES)
def test_rf_macro_f1_beats_dummy(subject, scenario, data):
    df, train_idx, test_idx = data[subject]
    X, y = df[feature_columns(scenario)], df[TARGET_COL]
    dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[train_idx], y.iloc[train_idx])
    f1_dummy = f1_score(y.iloc[test_idx], dummy.predict(X.iloc[test_idx]), average="macro")
    f1_rf = f1_score(y.iloc[test_idx], load_model(subject, scenario).predict(X.iloc[test_idx]), average="macro")
    assert f1_rf >= f1_dummy


# --------------------------------------------------------------------------- predict()
@pytest.mark.parametrize("subject,scenario", CASES)
def test_predict_output_contract(subject, scenario, data):
    df, _, _ = data[subject]
    out = predict(subject, scenario, student(df, scenario, i=5))
    assert set(out) == {"label", "proba_pass", "top_factors"}
    assert 0 <= out["proba_pass"] <= 1
    assert out["label"] == ("Likely to Pass" if out["proba_pass"] >= 0.5 else "At Risk")
    assert len(out["top_factors"]) == 3
    for f in out["top_factors"]:
        assert f["feature"] in feature_columns(scenario)
        assert f["effect"] in {"better", "worse", "neutral"}
        assert {"value", "reference", "reference_type", "position", "delta_proba_pass"} <= set(f)


def test_predict_ignores_known_unused_fields(data):
    df, _, _ = data["por"]
    base = student(df, "early")
    extra = {**base, "sex": "M", "G1": 3, "G2": 4}  # not used by the Early model
    assert predict("por", "early", base) == predict("por", "early", extra)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: {k: v for k, v in d.items() if k != "failures"}, "missing fields"),
        (lambda d: {**d, "shoe_size": 42}, "unknown fields"),
        (lambda d: {**d, "G3": 12}, "target"),
        (lambda d: {**d, "age": 30}, "out of range"),
        (lambda d: {**d, "absences": -1}, "out of range"),
        (lambda d: {**d, "failures": 4}, "out of range"),
        (lambda d: {**d, "studytime": 2.5}, "whole number"),
        (lambda d: {**d, "studytime": "2"}, "whole number"),
        (lambda d: {**d, "health": True}, "whole number"),
        (lambda d: {**d, "absences": float("nan")}, "whole number"),
        (lambda d: {**d, "Mjob": "astronaut"}, "must be one of"),
        (lambda d: {**d, "internet": True}, "must be one of"),
    ],
    ids=["missing", "unknown", "g3", "age-high", "absences-neg", "failures-4", "float",
         "str-number", "bool-number", "nan", "bad-category", "bool-category"],
)
def test_predict_rejects_bad_input(mutate, message, data):
    df, _, _ = data["por"]
    with pytest.raises(InputValidationError, match=message):
        predict("por", "early", mutate(student(df, "early")))


def test_predict_accepts_integer_valued_float(data):
    df, _, _ = data["por"]
    d = student(df, "early")
    assert predict("por", "early", {**d, "absences": float(d["absences"])}) == predict("por", "early", d)


def test_predict_midterm_requires_grades(data):
    df, _, _ = data["mat"]
    with pytest.raises(InputValidationError, match="missing fields"):
        predict("mat", "midterm", student(df, "early"))


@pytest.mark.parametrize("subject,scenario", [("bio", "early"), ("por", "final")])
def test_predict_rejects_unknown_subject_or_scenario(subject, scenario):
    with pytest.raises(InputValidationError):
        predict(subject, scenario, {})


def test_predict_rejects_non_dict():
    with pytest.raises(InputValidationError):
        predict("por", "early", ["not", "a", "dict"])
