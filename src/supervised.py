"""Supervised learning (spec §6): pass / At-Risk classification.

Label convention: 0 = At Risk (G3 < 10), 1 = Likely to Pass (G3 >= 10).

Saved models are ``FixedThresholdClassifier(Pipeline(preprocessor, RandomForest))``: the RF
has a monotonic constraint (P(pass) never rises with absences) and the wrapper carries the
alert cut-off chosen from out-of-fold predictions on the training set. A student is predicted
"Likely to Pass" when P(pass) >= threshold, otherwise "At Risk".
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    FixedThresholdClassifier,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RANDOM_STATE, SUBJECTS
from src.data import FEATURE_SPEC, TARGET_COL, add_target, load_subject, validate_schema
from src.metrics_io import load_metrics
from src.preprocessing import (
    CATEGORICAL_COLS,
    SCENARIOS,
    build_preprocessor,
    feature_columns,
    monotonic_constraints,
)

AT_RISK, PASS = 0, 1
LABEL_NAMES = {AT_RISK: "At Risk", PASS: "Likely to Pass"}
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = "f1_macro"
RF_PARAM_GRID = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_leaf": [1, 3, 5],
}
DEFAULT_THRESHOLD = 0.5
TARGET_AT_RISK_RECALL = 0.70
THRESHOLD_FLOOR = 0.5  # the alert cut-off never goes below the default (would lower At-Risk recall)
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.005), 3)
PERM_REPEATS = 20
TOP_N_IMPORTANCE = 10
N_TOP_FACTORS = 3
NEUTRAL_DELTA = 0.01  # |Δ proba_pass| below this counts as "neutral"
ABSENCE_BINS = {"0": (0, 0), "1–5": (1, 5), "6–10": (6, 10), ">10": (11, 10_000)}


# --------------------------------------------------------------------------- data & split
def prepare_subject(subject: str) -> pd.DataFrame:
    df = add_target(load_subject(subject))
    validate_schema(df)
    return df


def split_indices(df: pd.DataFrame, seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 split; identical for every model of a subject (seed 42 by default)."""
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=seed)
    return np.sort(train_idx), np.sort(test_idx)


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- models
def make_dummy(features: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(features)),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])


def make_logreg(features: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(features, scale_numeric=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def make_rf_search(features: list[str], monotonic: bool = True, n_jobs: int = -1) -> GridSearchCV:
    rf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
        monotonic_cst=monotonic_constraints(features) if monotonic else None,
    )
    pipe = Pipeline([("prep", build_preprocessor(features)), ("clf", rf)])
    return GridSearchCV(pipe, RF_PARAM_GRID, scoring=SCORING, cv=cv_splitter(), n_jobs=n_jobs, refit=True)


# --------------------------------------------------------------------------- threshold
def proba_pass(model, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(PASS)]


def labels_at(p_pass: np.ndarray, threshold: float) -> np.ndarray:
    """Likely to Pass iff P(pass) >= threshold (same rule as FixedThresholdClassifier)."""
    return np.where(p_pass >= threshold, PASS, AT_RISK)


def tune_threshold(estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                   target_recall: float = TARGET_AT_RISK_RECALL, n_jobs: int = -1) -> dict[str, Any]:
    """Alert cut-off from out-of-fold predictions on the TRAINING set (test set never used).

    threshold_raw = smallest cut-off whose OOF At-Risk recall is >= target (fewest false alerts).
    threshold     = max(THRESHOLD_FLOOR, threshold_raw): when recall at 0.5 already exceeds the
                    target, the raw rule would pick a cut-off below 0.5 and lower recall, so the
                    floor keeps the default instead.
    """
    oof = cross_val_predict(clone(estimator), X_train, y_train, cv=cv_splitter(),
                            method="predict_proba", n_jobs=n_jobs)[:, PASS]
    y = np.asarray(y_train)
    is_risk = y == AT_RISK
    for t in THRESHOLD_GRID:
        pred = labels_at(oof, t)
        rec = float((pred[is_risk] == AT_RISK).mean())
        if rec >= target_recall:
            break
    raw = float(t)
    final = max(THRESHOLD_FLOOR, raw)
    pred_raw = labels_at(oof, raw)
    return {
        "threshold": final,
        "threshold_raw": raw,
        "threshold_floor": THRESHOLD_FLOOR,
        "target_at_risk_recall": target_recall,
        "oof_at_risk_recall": round(float((labels_at(oof, final)[is_risk] == AT_RISK).mean()), 4),
        "oof_at_risk_recall_raw": round(rec, 4),
        "oof_at_risk_precision_raw": round(precision_score(y, pred_raw, pos_label=AT_RISK, zero_division=0), 4),
        "oof_f1_macro_raw": round(f1_score(y, pred_raw, average="macro"), 4),
        "oof_at_risk_recall_at_default": round(float((labels_at(oof, DEFAULT_THRESHOLD)[is_risk] == AT_RISK).mean()), 4),
    }


# --------------------------------------------------------------------------- evaluation
def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float = DEFAULT_THRESHOLD,
             include_curves: bool = True) -> dict[str, Any]:
    """Test-set metrics at a given cut-off. At-Risk (class 0) recall is the headline number."""
    p = proba_pass(model, X)
    y = np.asarray(y)
    y_pred = labels_at(p, threshold)
    report = classification_report(
        y, y_pred, labels=[AT_RISK, PASS], target_names=[LABEL_NAMES[AT_RISK], LABEL_NAMES[PASS]],
        output_dict=True, zero_division=0,
    )
    false_alerts = int(((y == PASS) & (y_pred == AT_RISK)).sum())
    out = {
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_at_risk": int((y == AT_RISK).sum()),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "at_risk_recall": round(recall_score(y, y_pred, pos_label=AT_RISK, zero_division=0), 4),
        "at_risk_precision": round(precision_score(y, y_pred, pos_label=AT_RISK, zero_division=0), 4),
        "f1_macro": round(f1_score(y, y_pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "false_alerts": false_alerts,
        "false_alert_rate": round(false_alerts / max(int((y == PASS).sum()), 1), 4),
        "per_class": {
            name: {k: round(report[name][k], 4) for k in ("precision", "recall", "f1-score", "support")}
            for name in (LABEL_NAMES[AT_RISK], LABEL_NAMES[PASS])
        },
        "confusion_matrix": {
            "labels": [LABEL_NAMES[AT_RISK], LABEL_NAMES[PASS]],  # rows = true, cols = predicted
            "matrix": confusion_matrix(y, y_pred, labels=[AT_RISK, PASS]).tolist(),
        },
    }
    if include_curves:
        fpr, tpr, _ = roc_curve(y, p, pos_label=PASS)
        out["roc_curve"] = {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()}
    return out


def baseline_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series, n_jobs: int = -1) -> dict[str, float]:
    scores = cross_val_score(clone(model), X, y, scoring=SCORING, cv=cv_splitter(), n_jobs=n_jobs)
    return {"metric": SCORING, "mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}


def search_cv(search: GridSearchCV) -> dict[str, float]:
    """GridSearchCV's validation score (mean_test_score over held-out folds) for the best params."""
    i = search.best_index_
    return {
        "metric": SCORING,
        "source": "GridSearchCV.cv_results_['mean_test_score'] (held-out folds)",
        "mean": round(float(search.cv_results_["mean_test_score"][i]), 4),
        "std": round(float(search.cv_results_["std_test_score"][i]), 4),
    }


def best_params(search: GridSearchCV) -> dict[str, Any]:
    return {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}


def permutation_top(model, X: pd.DataFrame, y: pd.Series, top_n: int = TOP_N_IMPORTANCE) -> list[dict]:
    """Permutation importance on raw input columns (drop in test macro-F1, at the model's cut-off)."""
    r = permutation_importance(
        model, X, y, scoring=SCORING, n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1
    )
    order = np.argsort(-r.importances_mean)
    return [
        {"feature": X.columns[i], "mean": round(float(r.importances_mean[i]), 4),
         "std": round(float(r.importances_std[i]), 4)}
        for i in order[:top_n]
    ]


def fairness_check(model, X: pd.DataFrame, y: pd.Series, sex: pd.Series,
                   threshold: float = DEFAULT_THRESHOLD) -> dict[str, dict]:
    """At-Risk recall per sex on the test set (sex is NOT a model feature)."""
    y_pred = pd.Series(labels_at(proba_pass(model, X), threshold), index=X.index)
    out = {}
    for group in ("F", "M"):
        mask = (sex == group) & (y == AT_RISK)
        n = int(mask.sum())
        caught = int((y_pred[mask] == AT_RISK).sum())
        out[group] = {"n_test": int((sex == group).sum()), "n_at_risk": n, "n_caught": caught,
                      "at_risk_recall": round(caught / n, 4) if n else None}
    return out


def absences_partial_dependence(model, X: pd.DataFrame) -> dict[str, Any]:
    """Average predicted P(pass) when every row's `absences` is set to a value, others as in data.

    Bin values are the observed absence values in that bin, weighted by how often they occur.
    """
    cache: dict[int, float] = {}

    def pd_at(v: int) -> float:
        if v not in cache:
            cache[v] = float(proba_pass(model, X.assign(absences=v)).mean())
        return cache[v]

    observed = X["absences"]
    bins = {}
    for name, (lo, hi) in ABSENCE_BINS.items():
        counts = observed[(observed >= lo) & (observed <= hi)].value_counts()
        if counts.empty:
            continue
        bins[name] = round(sum(pd_at(int(v)) * c for v, c in counts.items()) / counts.sum(), 4)
    curve = {str(v): round(pd_at(v), 4) for v in range(0, 31)}
    return {"by_bin": bins, "curve_0_30": curve}


# --------------------------------------------------------------------------- training
def fit_rf(features: list[str], X_train: pd.DataFrame, y_train: pd.Series, monotonic: bool = True,
           n_jobs: int = -1) -> tuple[FixedThresholdClassifier, dict[str, Any]]:
    """GridSearchCV -> best pipeline -> OOF cut-off on train -> FixedThresholdClassifier."""
    search = make_rf_search(features, monotonic=monotonic, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    thr = tune_threshold(search.best_estimator_, X_train, y_train, n_jobs=n_jobs)
    model = FixedThresholdClassifier(search.best_estimator_, threshold=thr["threshold"], pos_label=PASS,
                                     response_method="predict_proba")
    model.fit(X_train, y_train)  # refits a clone with the same params and seed => identical forest
    info = {"cv": search_cv(search), "best_params": best_params(search), "monotonic_absences": monotonic,
            "threshold_tuning": thr}
    return model, info


def train_scenario(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, scenario: str,
                   with_baselines: bool = True, with_unconstrained: bool = True, n_jobs: int = -1,
                   include_curves: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit baselines + tuned RFs on train rows; evaluate once on test rows at 0.5 and at the cut-off."""
    features = feature_columns(scenario)
    X, y = df[features], df[TARGET_COL]
    X_tr, y_tr, X_te, y_te = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]

    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if with_baselines:
        for name, model in (("dummy", make_dummy(features)), ("logreg", make_logreg(features))):
            cv = baseline_cv(model, X_tr, y_tr, n_jobs=n_jobs)
            model.fit(X_tr, y_tr)
            models[name] = model
            metrics[name] = {"cv": cv, "test_default": evaluate(model, X_te, y_te, include_curves=include_curves)}

    variants = [("rf", True)] + ([("rf_unconstrained", False)] if with_unconstrained else [])
    for name, monotonic in variants:
        model, info = fit_rf(features, X_tr, y_tr, monotonic=monotonic, n_jobs=n_jobs)
        models[name] = model
        metrics[name] = {
            **info,
            "test_default": evaluate(model, X_te, y_te, DEFAULT_THRESHOLD, include_curves=include_curves),
            "test_tuned": evaluate(model, X_te, y_te, model.threshold, include_curves=False),
        }
    return models, metrics


def train_reference(X_train: pd.DataFrame) -> dict[str, Any]:
    """Median for numeric features, mode for categoricals (used by predict() top_factors)."""
    ref = {}
    for c in X_train.columns:
        if c in CATEGORICAL_COLS:
            ref[c] = str(X_train[c].mode().iloc[0])
        else:
            ref[c] = float(X_train[c].median())
    return ref


def model_path(subject: str, scenario: str):
    return MODELS_DIR / f"clf_{subject}_{scenario}.joblib"


# --------------------------------------------------------------------------- repeated splits
REPEATED_KEYS = ("accuracy", "f1_macro", "at_risk_recall", "at_risk_precision", "roc_auc", "false_alerts")


def run_split(subject: str, scenario: str, seed: int) -> dict[str, Any]:
    """Full procedure on one stratified split (seed): baselines + constrained RF + OOF cut-off.

    Runs single-threaded so many seeds can run in parallel.
    """
    df = prepare_subject(subject)
    train_idx, test_idx = split_indices(df, seed=seed)
    models, m = train_scenario(df, train_idx, test_idx, scenario, with_unconstrained=False, n_jobs=1,
                               include_curves=False)
    rf = models["rf"]
    X_te, y_te = df[feature_columns(scenario)].iloc[test_idx], df[TARGET_COL].iloc[test_idx]
    fair = fairness_check(rf, X_te, y_te, df["sex"].iloc[test_idx], rf.threshold)

    def pick(t: dict) -> dict:
        return {k: t[k] for k in REPEATED_KEYS}

    return {
        "subject": subject, "scenario": scenario, "seed": seed,
        "dummy": pick(m["dummy"]["test_default"]),
        "logreg": pick(m["logreg"]["test_default"]),
        "rf_default": pick(m["rf"]["test_default"]),
        "rf_tuned": pick(m["rf"]["test_tuned"]),
        "rf_cv_f1_macro": m["rf"]["cv"]["mean"],
        "rf_threshold": m["rf"]["threshold_tuning"]["threshold"],
        "rf_threshold_raw": m["rf"]["threshold_tuning"]["threshold_raw"],
        "fairness_final": {g: {"n_at_risk": fair[g]["n_at_risk"], "n_caught": fair[g]["n_caught"]} for g in ("F", "M")},
        "n_test_at_risk": m["rf"]["test_default"]["n_at_risk"],
    }


# --------------------------------------------------------------------------- prediction
class InputValidationError(ValueError):
    """Raised when predict() receives missing, unknown or out-of-range values."""


@lru_cache(maxsize=None)
def load_model(subject: str, scenario: str) -> FixedThresholdClassifier:
    path = model_path(subject, scenario)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/train_supervised.py` first.")
    return joblib.load(path)


def _supervised_meta(subject: str, scenario: str) -> dict[str, Any]:
    try:
        return load_metrics()["supervised"]["subjects"][subject][scenario]
    except KeyError as exc:
        raise FileNotFoundError(
            "reports/metrics.json has no supervised results. Run `python scripts/train_supervised.py`."
        ) from exc


def validate_input(input_dict: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """Check presence, type and range of every feature against the data dictionary.

    Known dataset columns that the scenario does not use (e.g. `sex`, or G1/G2 in Early) are
    ignored. G3 is rejected because it is the target.
    """
    if not isinstance(input_dict, dict):
        raise InputValidationError(f"input must be a dict, got {type(input_dict).__name__}")
    problems: list[str] = []
    if "G3" in input_dict:
        problems.append("'G3' is the prediction target and cannot be an input")
    unknown = sorted(k for k in input_dict if k not in FEATURE_SPEC)
    if unknown:
        problems.append(f"unknown fields: {unknown}")
    missing = [f for f in features if f not in input_dict]
    if missing:
        problems.append(f"missing fields: {missing}")

    row: dict[str, Any] = {}
    for f in features:
        if f not in input_dict:
            continue
        v, spec = input_dict[f], FEATURE_SPEC[f]
        if isinstance(spec, list):
            if not isinstance(v, str) or v not in spec:
                problems.append(f"{f}={v!r} must be one of {spec}")
                continue
            row[f] = v
        else:
            lo, hi = spec
            is_num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            if not is_num or not np.isfinite(float(v)) or float(v) != int(v):
                problems.append(f"{f}={v!r} must be a whole number in [{lo}, {hi}]")
                continue
            if not lo <= int(v) <= hi:
                problems.append(f"{f}={v!r} is out of range [{lo}, {hi}]")
                continue
            row[f] = int(v)
    if problems:
        raise InputValidationError("Invalid input: " + "; ".join(problems))
    return pd.DataFrame([row], columns=features)


def predict(subject: str, scenario: str, input_dict: dict[str, Any]) -> dict[str, Any]:
    """Predict one student.

    Returns {"label", "proba_pass", "top_factors"}. The label uses the model's tuned alert
    cut-off (Likely to Pass iff proba_pass >= threshold). top_factors are the 3 globally most
    important features (test-set permutation importance) with this student's value, the
    training median (numeric) or most common value (categorical), and the effect of the
    student's value vs that reference on P(pass), all other inputs held fixed.
    """
    if subject not in SUBJECTS:
        raise InputValidationError(f"Unknown subject {subject!r}; expected one of {sorted(SUBJECTS)}")
    if scenario not in SCENARIOS:
        raise InputValidationError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    features = feature_columns(scenario)
    X = validate_input(input_dict, features)
    model = load_model(subject, scenario)
    meta = _supervised_meta(subject, scenario)

    p = float(proba_pass(model, X)[0])
    label = LABEL_NAMES[int(model.predict(X)[0])]

    factors = []
    for item in meta["permutation_importance"][:N_TOP_FACTORS]:
        f = item["feature"]
        ref = meta["train_reference"][f]
        value = X.at[0, f]
        p_ref = float(proba_pass(model, X.assign(**{f: ref}))[0])
        delta = p - p_ref
        effect = "neutral" if abs(delta) < NEUTRAL_DELTA else ("better" if delta > 0 else "worse")
        if f in CATEGORICAL_COLS:
            position = "same as most common" if value == ref else "differs from most common"
        else:
            position = "equal to median" if value == ref else ("above median" if value > ref else "below median")
        factors.append({
            "feature": f,
            "value": value.item() if hasattr(value, "item") else value,
            "reference": ref,
            "reference_type": "most common" if f in CATEGORICAL_COLS else "median",
            "position": position,
            "effect": effect,
            "delta_proba_pass": round(delta, 4),
            "importance": item["mean"],
        })
    return {"label": label, "proba_pass": round(p, 4), "top_factors": factors}
