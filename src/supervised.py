"""Supervised learning (spec §6): pass / At-Risk classification.

Label convention: 0 = At Risk (G3 < 10), 1 = Likely to Pass (G3 >= 10).
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
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RANDOM_STATE, SUBJECTS
from src.data import FEATURE_SPEC, TARGET_COL, add_target, load_subject, validate_schema
from src.metrics_io import load_metrics
from src.preprocessing import CATEGORICAL_COLS, SCENARIOS, build_preprocessor, feature_columns

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


def split_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 split; identical for every model of a subject."""
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )
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


def make_rf_search(features: list[str]) -> GridSearchCV:
    pipe = Pipeline([
        ("prep", build_preprocessor(features)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
    ])
    return GridSearchCV(pipe, RF_PARAM_GRID, scoring=SCORING, cv=cv_splitter(), n_jobs=-1, refit=True)


# --------------------------------------------------------------------------- evaluation
def proba_pass(model, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(PASS)]


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Test-set metrics. At-Risk (class 0) recall is the headline number."""
    y_pred = model.predict(X)
    p = proba_pass(model, X)
    report = classification_report(
        y, y_pred, labels=[AT_RISK, PASS], target_names=[LABEL_NAMES[AT_RISK], LABEL_NAMES[PASS]],
        output_dict=True, zero_division=0,
    )
    fpr, tpr, _ = roc_curve(y, p, pos_label=PASS)
    return {
        "n": int(len(y)),
        "n_at_risk": int((y == AT_RISK).sum()),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "at_risk_recall": round(recall_score(y, y_pred, pos_label=AT_RISK, zero_division=0), 4),
        "f1_macro": round(f1_score(y, y_pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "per_class": {
            name: {k: round(report[name][k], 4) for k in ("precision", "recall", "f1-score", "support")}
            for name in (LABEL_NAMES[AT_RISK], LABEL_NAMES[PASS])
        },
        "confusion_matrix": {
            "labels": [LABEL_NAMES[AT_RISK], LABEL_NAMES[PASS]],  # rows = true, cols = predicted
            "matrix": confusion_matrix(y, y_pred, labels=[AT_RISK, PASS]).tolist(),
        },
        "roc_curve": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
    }


def baseline_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    scores = cross_val_score(clone(model), X, y, scoring=SCORING, cv=cv_splitter(), n_jobs=-1)
    return {"metric": SCORING, "mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}


def search_cv(search: GridSearchCV) -> dict[str, float]:
    i = search.best_index_
    return {
        "metric": SCORING,
        "mean": round(float(search.cv_results_["mean_test_score"][i]), 4),
        "std": round(float(search.cv_results_["std_test_score"][i]), 4),
    }


def best_params(search: GridSearchCV) -> dict[str, Any]:
    return {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}


def permutation_top(model, X: pd.DataFrame, y: pd.Series, top_n: int = TOP_N_IMPORTANCE) -> list[dict]:
    """Permutation importance on raw input columns (drop in test macro-F1)."""
    r = permutation_importance(
        model, X, y, scoring=SCORING, n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1
    )
    order = np.argsort(-r.importances_mean)
    return [
        {"feature": X.columns[i], "mean": round(float(r.importances_mean[i]), 4),
         "std": round(float(r.importances_std[i]), 4)}
        for i in order[:top_n]
    ]


def fairness_check(model, X: pd.DataFrame, y: pd.Series, sex: pd.Series) -> dict[str, dict]:
    """At-Risk recall per sex on the test set (sex is NOT a model feature)."""
    y_pred = pd.Series(model.predict(X), index=X.index)
    out = {}
    for group in ("F", "M"):
        mask = (sex == group) & (y == AT_RISK)
        n = int(mask.sum())
        rec = float((y_pred[mask] == AT_RISK).mean()) if n else None
        out[group] = {"n_test": int((sex == group).sum()), "n_at_risk": n,
                      "at_risk_recall": round(rec, 4) if rec is not None else None}
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
def train_scenario(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, scenario: str,
                   with_baselines: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit baselines + tuned RF on train rows; evaluate once on test rows."""
    features = feature_columns(scenario)
    X, y = df[features], df[TARGET_COL]
    X_tr, y_tr, X_te, y_te = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]

    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if with_baselines:
        for name, model in (("dummy", make_dummy(features)), ("logreg", make_logreg(features))):
            cv = baseline_cv(model, X_tr, y_tr)
            model.fit(X_tr, y_tr)
            models[name] = model
            metrics[name] = {"cv": cv, "test": evaluate(model, X_te, y_te)}

    search = make_rf_search(features)
    search.fit(X_tr, y_tr)
    rf = search.best_estimator_
    models["rf"] = rf
    metrics["rf"] = {"cv": search_cv(search), "best_params": best_params(search),
                     "test": evaluate(rf, X_te, y_te)}
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


# --------------------------------------------------------------------------- prediction
class InputValidationError(ValueError):
    """Raised when predict() receives missing, unknown or out-of-range values."""


@lru_cache(maxsize=None)
def load_model(subject: str, scenario: str) -> Pipeline:
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

    Returns {"label", "proba_pass", "top_factors"}. top_factors are the 3 globally most
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
    label = LABEL_NAMES[PASS] if p >= 0.5 else LABEL_NAMES[AT_RISK]

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
