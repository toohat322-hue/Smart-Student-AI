"""Feature sets and ColumnTransformer builders (spec §6)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import GRADE_COLS, SENSITIVE
from src.data import EXPECTED_COLUMNS, FEATURE_SPEC

SCENARIOS = ("early", "midterm")

BINARY_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
]
NOMINAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]
NUMERIC_COLS = [  # numeric + ordinal; trees use them as-is
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2",
]
CATEGORICAL_COLS = BINARY_COLS + NOMINAL_COLS


def feature_columns(scenario: str) -> list[str]:
    """Early: all columns except G1, G2, G3 and sensitive ones. Mid-term: Early + G1, G2.

    G3 is never a feature.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    excluded = set(GRADE_COLS) | set(SENSITIVE)
    early = [c for c in EXPECTED_COLUMNS if c not in excluded]
    if scenario == "early":
        return early
    return early + ["G1", "G2"]


def _groups(features: list[str]) -> tuple[list[str], list[str], list[str]]:
    numeric = [c for c in NUMERIC_COLS if c in features]
    binary = [c for c in BINARY_COLS if c in features]
    nominal = [c for c in NOMINAL_COLS if c in features]
    unknown = set(features) - set(binary) - set(nominal) - set(numeric)
    if unknown:
        raise ValueError(f"No preprocessing rule for columns: {sorted(unknown)}")
    return numeric, binary, nominal


def _categories(cols: list[str]) -> list[list[str]]:
    # Fixed categories from the data dictionary => output width never depends on the data.
    return [sorted(FEATURE_SPEC[c]) for c in cols]


def build_preprocessor(features: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    """Numeric/ordinal first (passthrough, or scaled for linear models), then one-hot categoricals.

    Output layout is fixed: [numeric..., one column per binary, one per nominal category], so
    the index of any numeric column (e.g. `absences`) is known before fitting.
    """
    numeric, binary, nominal = _groups(features)
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler() if scale_numeric else "passthrough", numeric),
            ("binary", OneHotEncoder(categories=_categories(binary), drop="if_binary",
                                     handle_unknown="error"), binary),
            ("nominal", OneHotEncoder(categories=_categories(nominal), handle_unknown="ignore"), nominal),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def output_feature_names(features: list[str]) -> list[str]:
    """Column names produced by build_preprocessor(features), computable without fitting."""
    numeric, binary, nominal = _groups(features)
    names = list(numeric)
    names += [f"{c}_{sorted(FEATURE_SPEC[c])[1]}" for c in binary]  # drop="if_binary" keeps 2nd
    names += [f"{c}_{v}" for c in nominal for v in sorted(FEATURE_SPEC[c])]
    return names


# Direction of the effect on P(pass) that the RF is allowed to learn (-1 = non-increasing).
MONOTONIC_DECREASING = ["absences"]


def monotonic_constraints(features: list[str]) -> list[int]:
    """`monotonic_cst` vector for RandomForestClassifier, aligned with the preprocessor output.

    For binary classification sklearn applies the constraint to P(classes_[1]) = P(pass).
    """
    names = output_feature_names(features)
    return [-1 if n in MONOTONIC_DECREASING else 0 for n in names]
