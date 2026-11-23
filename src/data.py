"""Loading, target creation and schema validation for the UCI Student Performance data."""
from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR, GRADE_COLS, PASS_THRESHOLD, SUBJECTS

TARGET_COL = "pass"
GRADE_MIN, GRADE_MAX = 0, 20

# The 33 columns of student-mat.csv / student-por.csv, in file order (spec §2.3).
EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3",
]


class SchemaError(ValueError):
    """Raised when a DataFrame does not match the expected dataset schema."""


def load_subject(subject: str) -> pd.DataFrame:
    """Load one subject's raw CSV ("por" or "mat"). The files use ';' as separator."""
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject {subject!r}; expected one of {sorted(SUBJECTS)}")
    path = DATA_DIR / SUBJECTS[subject]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, sep=";")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``pass`` = 1 if G3 >= PASS_THRESHOLD else 0."""
    if "G3" not in df.columns:
        raise SchemaError("Cannot create target: column 'G3' is missing")
    out = df.copy()
    out[TARGET_COL] = (out["G3"] >= PASS_THRESHOLD).astype(int)
    return out


def validate_schema(df: pd.DataFrame) -> None:
    """Check the 33 expected columns (in order), no missing values, grades in 0–20.

    A ``pass`` column added by :func:`add_target` is tolerated. Raises
    :class:`SchemaError` describing every problem found.
    """
    problems: list[str] = []

    cols = [c for c in df.columns if c != TARGET_COL]
    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in EXPECTED_COLUMNS]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if not missing and not extra and cols != EXPECTED_COLUMNS:
        problems.append("columns are not in the expected order")

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        per_col = df.isna().sum()
        problems.append(f"{n_missing} missing values: {per_col[per_col > 0].to_dict()}")

    for col in (c for c in GRADE_COLS if c in df.columns):
        if not pd.api.types.is_integer_dtype(df[col]):
            problems.append(f"{col} is not integer (dtype {df[col].dtype})")
            continue
        lo, hi = df[col].min(), df[col].max()
        if lo < GRADE_MIN or hi > GRADE_MAX:
            problems.append(f"{col} out of range [{GRADE_MIN}, {GRADE_MAX}]: min={lo}, max={hi}")

    if problems:
        raise SchemaError("Schema validation failed: " + "; ".join(problems))
