"""Tests for src/preprocessing.py and src/supervised.py (spec §10: test_supervised.py).

Uses the committed pipelines in models/ (produced by scripts/train_supervised.py).
"""
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import FixedThresholdClassifier, cross_val_predict

from src.config import SUBJECTS
from src.data import FEATURE_SPEC, TARGET_COL
from src.metrics_io import load_metrics
from src.preprocessing import (
    SCENARIOS,
    build_preprocessor,
    feature_columns,
    monotonic_constraints,
    output_feature_names,
)
from src.supervised import (
    AT_RISK,
    PASS,
    InputValidationError,
    cv_splitter,
    evaluate,
    labels_at,
    load_model,
    make_logreg,
    model_path,
    predict,
    prepare_subject,
    proba_pass,
    split_indices,
    THRESHOLD_FLOOR,
    tune_threshold,
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
    threshold = load_model(subject, scenario).threshold
    assert out["label"] == ("Likely to Pass" if out["proba_pass"] >= threshold else "At Risk")
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


# --------------------------------------------------------------------------- monotonic absences
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_output_layout_is_known_before_fit(scenario, data):
    features = feature_columns(scenario)
    df, _, _ = data["mat"]
    fitted = build_preprocessor(features).fit(df[features])
    assert list(fitted.get_feature_names_out()) == output_feature_names(features)
    cst = monotonic_constraints(features)
    assert len(cst) == len(output_feature_names(features))
    assert [n for n, c in zip(output_feature_names(features), cst) if c != 0] == ["absences"]
    assert cst[output_feature_names(features).index("absences")] == -1


@pytest.mark.parametrize("subject,scenario", CASES)
def test_saved_rf_has_absences_constraint_on_right_column(subject, scenario):
    model = load_model(subject, scenario)
    pipe = model.estimator_
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    cst = np.asarray(pipe.named_steps["clf"].monotonic_cst)
    assert len(cst) == len(names) == pipe.named_steps["clf"].n_features_in_
    assert names[int(np.flatnonzero(cst)[0])] == "absences"
    assert cst[names.index("absences")] == -1 and np.count_nonzero(cst) == 1


@pytest.mark.parametrize("subject,scenario", CASES)
def test_pass_probability_never_rises_with_absences(subject, scenario, data):
    """Only absences changes (0..93, the data-dictionary range); P(pass) must be non-increasing."""
    df, _, test_idx = data[subject]
    X = df[feature_columns(scenario)].iloc[test_idx]
    model = load_model(subject, scenario)
    lo, hi = FEATURE_SPEC["absences"]
    grid = np.arange(lo, hi + 1)
    probs = np.column_stack([proba_pass(model, X.assign(absences=int(v))) for v in grid])  # rows x values
    assert (np.diff(probs, axis=1) <= 1e-12).all()


# --------------------------------------------------------------------------- alert cut-off
@pytest.mark.parametrize("subject,scenario", CASES)
def test_saved_model_carries_tuned_threshold(subject, scenario):
    model = load_model(subject, scenario)
    assert isinstance(model, FixedThresholdClassifier)
    meta = load_metrics()["supervised"]["subjects"][subject][scenario]
    tt = meta["models"]["rf"]["threshold_tuning"]
    assert 0.5 <= model.threshold < 1
    assert model.threshold == meta["threshold"] == tt["threshold"] == max(THRESHOLD_FLOOR, tt["threshold_raw"])
    assert tt["oof_at_risk_recall"] >= 0.70 and tt["oof_at_risk_recall_raw"] >= 0.70


def test_threshold_floor_never_lowers_recall_below_default():
    """The final cut-off is never below 0.5, so it can only add alerts compared with the default."""
    meta = load_metrics()["supervised"]["subjects"]
    for subject, scenario in CASES:
        m = meta[subject][scenario]["models"]["rf"]
        assert m["threshold_tuning"]["threshold"] >= THRESHOLD_FLOOR == 0.5
        assert m["test_tuned"]["at_risk_recall"] >= m["test_default"]["at_risk_recall"]


@pytest.mark.parametrize("subject,scenario", CASES)
def test_model_predict_uses_threshold(subject, scenario, data):
    df, _, test_idx = data[subject]
    X = df[feature_columns(scenario)].iloc[test_idx]
    model = load_model(subject, scenario)
    assert np.array_equal(model.predict(X), labels_at(proba_pass(model, X), model.threshold))


def test_labels_at_boundary():
    assert labels_at(np.array([0.59, 0.6, 0.61]), 0.6).tolist() == [AT_RISK, PASS, PASS]


def test_tune_threshold_picks_smallest_cutoff_reaching_target(data):
    df, train_idx, _ = data["por"]
    features = feature_columns("early")
    X, y = df[features].iloc[train_idx], df[TARGET_COL].iloc[train_idx]
    est = make_logreg(features)
    res = tune_threshold(est, X, y, target_recall=0.7, n_jobs=1)
    oof = cross_val_predict(est, X, y, cv=cv_splitter(), method="predict_proba")[:, PASS]
    risk = (y == AT_RISK).to_numpy()

    def recall(t):
        return (labels_at(oof, t)[risk] == AT_RISK).mean()

    raw = res["threshold_raw"]
    assert recall(raw) >= 0.7
    assert abs(recall(raw) - res["oof_at_risk_recall_raw"]) < 1e-4
    assert recall(round(raw - 0.005, 3)) < 0.7  # one grid step lower misses the target
    assert res["threshold"] == max(THRESHOLD_FLOOR, raw)
    assert abs(recall(res["threshold"]) - res["oof_at_risk_recall"]) < 1e-4


def test_evaluate_false_alerts_match_confusion_matrix(data):
    df, _, test_idx = data["mat"]
    X, y = df[feature_columns("early")].iloc[test_idx], df[TARGET_COL].iloc[test_idx]
    model = load_model("mat", "early")
    for t in (0.5, model.threshold):
        m = evaluate(model, X, y, t, include_curves=False)
        cm = np.array(m["confusion_matrix"]["matrix"])  # rows true [AR, Pass], cols predicted
        assert m["false_alerts"] == cm[1, 0]
        assert m["at_risk_recall"] == round(cm[0, 0] / cm[0].sum(), 4)
        assert m["at_risk_precision"] == round(cm[0, 0] / cm[:, 0].sum(), 4)


def test_split_seed_changes_split(data):
    df, train_idx, _ = data["por"]
    other, _ = split_indices(df, seed=0)
    assert not np.array_equal(train_idx, other)


# --------------------------------------------------------------------------- repeated splits
@pytest.mark.parametrize("subject,scenario", CASES)
def test_repeated_split_results_present(subject, scenario):
    rep = load_metrics()["supervised_repeated_splits"]["subjects"][subject][scenario]
    assert rep["n_splits"] == 20 and sorted(rep["seeds"]) == list(range(20))
    for mk in ("dummy", "logreg", "rf_default", "rf_tuned"):
        for k in ("f1_macro", "at_risk_recall"):
            assert 0 <= rep[mk][k]["mean"] <= 1 and rep[mk][k]["std"] >= 0
    assert all(r["rf_threshold"] >= THRESHOLD_FLOOR for r in rep["per_seed"])
    # pooled fairness = summed counts over the 20 test sets, not averaged percentages
    for g in ("F", "M"):
        pooled = rep["fairness_pooled_final"][g]
        caught = sum(r["fairness_final"][g]["n_caught"] for r in rep["per_seed"])
        total = sum(r["fairness_final"][g]["n_at_risk"] for r in rep["per_seed"])
        assert (pooled["n_caught"], pooled["n_at_risk"]) == (caught, total)
        assert pooled["at_risk_recall"] == round(caught / total, 4)
