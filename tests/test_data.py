"""Tests for src/data.py (spec §10: test_data.py)."""
import pandas as pd
import pytest

from src.config import DATA_DIR, PASS_THRESHOLD, SUBJECTS
from src.data import (
    EXPECTED_COLUMNS,
    SchemaError,
    add_target,
    load_subject,
    validate_schema,
)

EXPECTED_SHAPES = {"mat": (395, 33), "por": (649, 33)}


@pytest.fixture(scope="module", params=sorted(SUBJECTS))
def subject_df(request):
    return request.param, load_subject(request.param)


def test_csvs_use_semicolon_separator():
    for filename in SUBJECTS.values():
        header = (DATA_DIR / filename).read_text(encoding="utf-8").splitlines()[0]
        assert ";" in header and "," not in header


def test_shape(subject_df):
    subject, df = subject_df
    assert df.shape == EXPECTED_SHAPES[subject]


def test_column_names(subject_df):
    _, df = subject_df
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(EXPECTED_COLUMNS) == 33


def test_no_missing_values(subject_df):
    _, df = subject_df
    assert not df.isna().any().any()


def test_grade_ranges(subject_df):
    _, df = subject_df
    for col in ["G1", "G2", "G3"]:
        assert pd.api.types.is_integer_dtype(df[col])
        assert df[col].between(0, 20).all(), col


def test_validate_schema_passes_on_real_data(subject_df):
    _, df = subject_df
    validate_schema(df)
    validate_schema(add_target(df))  # the target column is tolerated


def test_add_target(subject_df):
    _, df = subject_df
    out = add_target(df)
    assert "pass" not in df.columns  # input not mutated
    assert set(out["pass"].unique()) <= {0, 1}
    assert (out["pass"] == (df["G3"] >= PASS_THRESHOLD).astype(int)).all()


def test_add_target_threshold_boundary():
    df = pd.DataFrame({"G3": [0, 9, 10, 11, 20]})
    assert add_target(df)["pass"].tolist() == [0, 0, 1, 1, 1]


def test_load_subject_unknown():
    with pytest.raises(ValueError):
        load_subject("bio")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns=["absences"]),
        lambda d: d.assign(extra_col=1),
        lambda d: d.assign(G3=d["G3"].where(d.index != 0, 21)),
        lambda d: d.assign(G1=d["G1"].where(d.index != 0, -1)),
        lambda d: d.assign(age=d["age"].astype("float").where(d.index != 0, None)),
    ],
    ids=["missing-col", "extra-col", "grade-too-high", "grade-negative", "nan-value"],
)
def test_validate_schema_rejects_bad_data(mutate):
    df = load_subject("mat")
    with pytest.raises(SchemaError):
        validate_schema(mutate(df))
