"""Feature sets and ColumnTransformer builders (spec §6)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import GRADE_COLS, SENSITIVE
from src.data import EXPECTED_COLUMNS

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


def build_preprocessor(features: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    """One-hot encode categoricals; pass numeric/ordinal through (or scale them for linear models)."""
    binary = [c for c in BINARY_COLS if c in features]
    nominal = [c for c in NOMINAL_COLS if c in features]
    numeric = [c for c in NUMERIC_COLS if c in features]
    unknown = set(features) - set(binary) - set(nominal) - set(numeric)
    if unknown:
        raise ValueError(f"No preprocessing rule for columns: {sorted(unknown)}")
    return ColumnTransformer(
        transformers=[
            ("binary", OneHotEncoder(drop="if_binary", handle_unknown="error"), binary),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal),
            ("numeric", StandardScaler() if scale_numeric else "passthrough", numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
