"""Shared Streamlit helpers: page setup, sidebar, cached loaders, the shared student, styling.

Every number shown in the app comes from reports/metrics.json, the saved models, or live
computation on the data; nothing numeric is hard-coded in the pages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.data import FEATURE_SPEC, TARGET_COL

ROOT_DIR = Path(__file__).resolve().parent.parent
SUBJECT_LABELS = {"por": "Portuguese", "mat": "Mathematics"}
SCENARIO_LABELS = {"early": "Early (start of year)", "midterm": "Mid-term (after period 2)"}
RETRAIN_HINT = "Run `python scripts/train_all.py`"
PAGES = {
    "home": "app.py",
    "predict": "pages/1_Performance_Prediction.py",
    "cluster": "pages/2_Student_Clustering.py",
    "plan": "pages/3_Study_Planner.py",
    "eval": "pages/4_Model_Evaluation.py",
    "about": "pages/5_About_Project.py",
}

# Reference categorical palette (fixed order), status colours reserved for pass / at risk.
NAVY = "#1E3A8A"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
GRAY, TEXT2 = "#8a8984", "#52514e"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
PASS_COLOR, RISK_COLOR = BLUE, ORANGE
ACTION_COLORS = [BLUE, AQUA, YELLOW]  # quizzes, videos, review notes

# --------------------------------------------------------------------------- plain-English labels
_LEVEL5 = {1: "1 – very low", 2: "2 – low", 3: "3 – medium", 4: "4 – high", 5: "5 – very high"}
_QUALITY5 = {1: "1 – very bad", 2: "2 – bad", 3: "3 – fair", 4: "4 – good", 5: "5 – very good"}
_EDU = {0: "None", 1: "Primary (4th grade)", 2: "5th–9th grade", 3: "Secondary", 4: "Higher education"}
_JOB = {"teacher": "Teacher", "health": "Health care", "services": "Civil services", "at_home": "At home",
        "other": "Other"}
FIELD_INFO: dict[str, dict[str, Any]] = {
    "school": {"label": "School", "options": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}},
    "age": {"label": "Age"},
    "address": {"label": "Home address", "options": {"U": "Urban", "R": "Rural"}},
    "famsize": {"label": "Family size", "options": {"LE3": "3 or fewer", "GT3": "More than 3"}},
    "Pstatus": {"label": "Parents", "options": {"T": "Living together", "A": "Living apart"}},
    "Medu": {"label": "Mother's education", "options": _EDU},
    "Fedu": {"label": "Father's education", "options": _EDU},
    "Mjob": {"label": "Mother's job", "options": _JOB},
    "Fjob": {"label": "Father's job", "options": _JOB},
    "reason": {"label": "Reason for choosing this school",
               "options": {"home": "Close to home", "reputation": "School reputation",
                           "course": "Course preference", "other": "Other"}},
    "guardian": {"label": "Guardian", "options": {"mother": "Mother", "father": "Father", "other": "Other"}},
    "traveltime": {"label": "Travel time to school",
                   "options": {1: "< 15 min", 2: "15–30 min", 3: "30–60 min", 4: "> 1 hour"}},
    "studytime": {"label": "Weekly study time", "options": {1: "< 2 h", 2: "2–5 h", 3: "5–10 h", 4: "> 10 h"}},
    "failures": {"label": "Past class failures", "options": {0: "0", 1: "1", 2: "2", 3: "3 or more"}},
    "schoolsup": {"label": "Extra support from school"},
    "famsup": {"label": "Educational support from family"},
    "paid": {"label": "Extra paid classes"},
    "activities": {"label": "Extra-curricular activities"},
    "nursery": {"label": "Attended nursery school"},
    "higher": {"label": "Wants higher education"},
    "internet": {"label": "Internet at home"},
    "romantic": {"label": "In a romantic relationship"},
    "famrel": {"label": "Family relationships", "options": {1: "1 – very bad", 2: "2 – bad", 3: "3 – fair",
                                                              4: "4 – good", 5: "5 – excellent"}},
    "freetime": {"label": "Free time after school", "options": _LEVEL5},
    "goout": {"label": "Going out with friends", "options": _LEVEL5},
    "Dalc": {"label": "Workday alcohol use", "options": _LEVEL5},
    "Walc": {"label": "Weekend alcohol use", "options": _LEVEL5},
    "health": {"label": "Current health", "options": _QUALITY5},
    "absences": {"label": "School absences"},
    "G1": {"label": "Period 1 grade (0–20)"},
    "G2": {"label": "Period 2 grade (0–20)"},
}
FORM_GROUPS = {
    "Background": ["school", "age", "address", "famsize", "Pstatus", "guardian", "Medu", "Fedu", "Mjob", "Fjob",
                   "reason", "traveltime", "nursery", "internet"],
    "Study & support": ["studytime", "failures", "absences", "schoolsup", "famsup", "paid", "higher"],
    "Lifestyle": ["activities", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health"],
}
STUDENT_FEATURES = [f for f in FEATURE_SPEC if f not in ("sex", "G3")]  # what the shared student holds


def label(feature: str) -> str:
    """Plain-English label of a feature."""
    return FIELD_INFO.get(feature, {}).get("label", feature)


def fmt_value(feature: str, value: Any) -> str:
    """Plain-English text for a feature value (e.g. studytime 2 -> 2–5 h)."""
    opts = FIELD_INFO.get(feature, {}).get("options")
    if opts and value in opts:
        return opts[value]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


# --------------------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def metrics() -> dict:
    """reports/metrics.json, cached for the Streamlit session."""
    from src.metrics_io import load_metrics

    return load_metrics()


@st.cache_data(show_spinner=False)
def subject_data(subject: str) -> pd.DataFrame:
    """One subject's validated data with the pass target, cached."""
    from src.supervised import prepare_subject

    return prepare_subject(subject)


@st.cache_resource(show_spinner=False)
def classifier(subject: str, scenario: str):
    """Saved classifier for (subject, scenario), shared across sessions."""
    from src.supervised import load_model

    return load_model(subject, scenario)


@st.cache_resource(show_spinner=False)
def kmeans(subject: str):
    """Saved K-Means pipeline for a subject, shared across sessions."""
    from src.unsupervised import load_kmeans

    return load_kmeans(subject)


@st.cache_resource(show_spinner=False)
def q_tables():
    """Saved Q-tables, shared across sessions."""
    from src.rl_agent import load_q_tables

    return load_q_tables()


def require(models: list[str] = (), sections: list[str] = ()) -> dict:
    """Stop the page with a clear message if a model file or a metrics section is missing."""
    missing = [m for m in models if not (ROOT_DIR / "models" / m).is_file()]
    if missing:
        st.error(f"Model file(s) missing: {', '.join('models/' + m for m in missing)}. {RETRAIN_HINT}.")
        st.stop()
    if not (ROOT_DIR / "reports" / "metrics.json").is_file():
        st.error(f"reports/metrics.json is missing. {RETRAIN_HINT}.")
        st.stop()
    m = metrics()
    absent = [s for s in sections if s not in m]
    if absent:
        st.error(f"reports/metrics.json has no {', '.join(absent)} results. {RETRAIN_HINT}.")
        st.stop()
    return m


def classifier_files(subject: str) -> list[str]:
    """File names of a subject's two classifiers (for require())."""
    return [f"clf_{subject}_{sc}.joblib" for sc in ("early", "midterm")]


# --------------------------------------------------------------------------- shared student
def _init_state() -> None:
    st.session_state.setdefault("subject", C.DEFAULT_SUBJECT)
    st.session_state.setdefault("scenario", "early")
    st.session_state.setdefault("student", None)        # dict of STUDENT_FEATURES
    st.session_state.setdefault("student_label", None)  # where the student came from
    st.session_state.setdefault("student_actual", None)  # {"subject", "G3", "pass"} for real records


def subject() -> str:
    """Currently selected subject (por or mat)."""
    return st.session_state["subject"]


def student() -> dict | None:
    """The shared student dict, or None."""
    return st.session_state.get("student")


def default_student(subj: str) -> dict:
    """Training median (numeric) / most common value (categorical) of the Mid-term model."""
    ref = metrics()["supervised"]["subjects"][subj]["midterm"]["train_reference"]
    return {f: (int(round(v)) if isinstance(v, float) else v) for f, v in ref.items()}


def set_student(values: dict, source: str, actual: dict | None = None) -> None:
    """Store the student and sync the prediction-form widgets (call from callbacks only)."""
    s = {f: values[f] for f in STUDENT_FEATURES if f in values}
    st.session_state["student"] = s
    st.session_state["student_label"] = source
    st.session_state["student_actual"] = actual
    for f, v in s.items():
        st.session_state[f"f_{f}"] = v


def clear_student() -> None:
    """Forget the shared student."""
    for k in ("student", "student_label", "student_actual"):
        st.session_state[k] = None


EXAMPLES = {
    "pass": "Clear pass",
    "at_risk": "Clear At Risk",
    "dropout": "Dropout case (final grade 0)",
}


@st.cache_data(show_spinner=False)
def example_students(subj: str) -> dict[str, dict]:
    """Three real, anonymised records from the seed-42 TEST split (never seen in training).

    clear pass = passing student with the highest Early P(pass); clear At Risk = failing student
    (G3 > 0) with the lowest; dropout = the G3 = 0 student with the lowest.
    """
    from src.supervised import proba_pass, split_indices
    from src.preprocessing import feature_columns

    df = subject_data(subj)
    _, test_idx = split_indices(df)
    test = df.iloc[test_idx].copy()
    test["p_early"] = proba_pass(classifier(subj, "early"), test[feature_columns("early")])
    picks = {
        "pass": test[test[TARGET_COL] == 1]["p_early"].idxmax(),
        "at_risk": test[(test[TARGET_COL] == 0) & (test["G3"] > 0)]["p_early"].idxmin(),
        "dropout": test[test["G3"] == 0]["p_early"].idxmin(),
    }
    out = {}
    for key, idx in picks.items():
        row = df.loc[idx]
        out[key] = {
            "record": {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in STUDENT_FEATURES},
            "row": int(idx) + 2,  # line number in the CSV (header = line 1)
            "G3": int(row["G3"]),
            "pass": int(row[TARGET_COL]),
        }
    return out


def load_example(key: str) -> None:
    """Load one of the three real example students of the current subject (callback)."""
    subj = subject()
    ex = example_students(subj)[key]
    set_student(ex["record"], f"{EXAMPLES[key]} · real record, {SUBJECT_LABELS[subj]} line {ex['row']}",
                actual={"subject": subj, "G3": ex["G3"], "pass": ex["pass"]})


def example_buttons(prefix: str = "ex") -> None:
    """The three load-example-student buttons."""
    st.caption("Load a **real, anonymised record from the UCI dataset** (taken from the held-out test set):")
    cols = st.columns(len(EXAMPLES))
    for col, (key, name) in zip(cols, EXAMPLES.items()):
        col.button(name, key=f"{prefix}_{key}", on_click=load_example, args=(key,), width="stretch")


def _query_example() -> None:
    """Handle ?example=pass|at_risk|dropout and ?subject=por|mat; unknown values get a clear warning."""
    subj = st.query_params.get("subject")
    if subj is not None and subj not in SUBJECT_LABELS:
        st.warning(f"Unknown subject '{subj}' in the link. Use one of: {', '.join(SUBJECT_LABELS)}. "
                   f"Showing {SUBJECT_LABELS[subject()]}.")
        subj = None
    key = st.query_params.get("example")
    if key is not None and key not in EXAMPLES:
        st.warning(f"Unknown example '{key}' in the link. Use one of: {', '.join(EXAMPLES)}.")
        return
    if key is not None and st.session_state.get("_query_example_done") != (key, subj):
        if subj is not None:
            st.session_state["subject"] = subj
        load_example(key)
        st.session_state["_query_example_done"] = (key, subj)


# --------------------------------------------------------------------------- page chrome
CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.ssa-card {background: var(--secondary-background-color, #E0EDFF); border-radius: 14px; padding: 1.0rem 1.15rem;
           height: 100%; border: 1px solid rgba(30,58,138,0.12);}
.ssa-card h3 {margin: 0 0 .35rem 0; font-size: 1.12rem; color: #1E3A8A;}
.ssa-card p {margin: 0; font-size: .95rem; line-height: 1.45;}
.ssa-kpi {background: #FFFFFF; border: 1px solid rgba(30,58,138,0.18); border-radius: 14px; padding: .85rem 1rem;
          height: 100%;}
.ssa-kpi .v {font-size: 1.9rem; font-weight: 700; color: #1E3A8A; line-height: 1.15;}
.ssa-kpi .l {font-size: .85rem; color: #52514e; margin-top: .1rem;}
.ssa-kpi .s {font-size: .8rem; color: #52514e; margin-top: .25rem;}
.ssa-result {border-radius: 14px; padding: 1.1rem 1.3rem; color: #0F172A; border: 2px solid;}
.ssa-result.pass {background: #eaf2fd; border-color: #2a78d6;}
.ssa-result.risk {background: #fdeee7; border-color: #eb6834;}
.ssa-result .t {font-size: 1.55rem; font-weight: 700;}
.ssa-note {background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: .65rem .9rem; border-radius: 6px;
           font-size: .92rem;}
@media (max-width: 640px) {
  .ssa-kpi .v {font-size: 1.5rem;}
  .ssa-result .t {font-size: 1.25rem;}
  .block-container {padding-left: 1rem; padding-right: 1rem;}
}
</style>
"""


def _validate_student() -> None:
    """Drop a stored student that is incomplete or invalid (e.g. a stale session after an app update)."""
    from src.supervised import InputValidationError, validate_input

    s = st.session_state.get("student")
    if s is None:
        return
    try:
        if not isinstance(s, dict):
            raise InputValidationError("not a dict")
        validate_input({k: v for k, v in s.items() if k in STUDENT_FEATURES}, STUDENT_FEATURES)
    except InputValidationError:
        clear_student()
        st.warning("The stored student was incomplete or invalid and has been cleared. "
                   "Please enter the student again or load an example.")


def setup_page(title: str, icon: str) -> None:
    """Page config, styling, shared state, ?example= handling and the sidebar (call first on every page)."""
    st.set_page_config(page_title=f"{title} · Smart Student AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()
    _validate_student()
    _query_example()
    _sidebar()


def _on_subject_change() -> None:
    st.session_state["subject"] = st.session_state["_subject_widget"]


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### Smart Student AI")
        options = list(SUBJECT_LABELS)
        st.selectbox("Subject", options, index=options.index(subject()), format_func=SUBJECT_LABELS.get,
                     key="_subject_widget", on_change=_on_subject_change,
                     help="Each subject has its own models (the two files are never mixed).")
        st.divider()
        st.markdown("**Current student**")
        if student() is None:
            st.caption("None yet. Enter one on the Prediction page or load an example.")
        else:
            st.caption(st.session_state.get("student_label") or "Entered by hand")
            st.button("Clear student", on_click=clear_student, key="_clear_student")
        st.divider()
        st.caption("Data: UCI Student Performance (Cortez, 2008), CC BY 4.0. "
                   "A teaching project; not for real decisions about students.")


def card(title: str, body: str) -> str:
    """HTML for a feature card (title + text)."""
    return f'<div class="ssa-card"><h3>{title}</h3><p>{body}</p></div>'


def kpi(value: str, label_: str, sub: str = "") -> str:
    """HTML for a KPI card (big value, label, optional sub-text)."""
    return f'<div class="ssa-kpi"><div class="v">{value}</div><div class="l">{label_}</div>' + \
           (f'<div class="s">{sub}</div>' if sub else "") + "</div>"


def note(text: str) -> None:
    """A highlighted note box."""
    st.markdown(f'<div class="ssa-note">{text}</div>', unsafe_allow_html=True)


def html_row(items: list[str], spec=None) -> None:
    """Render HTML snippets side by side in columns."""
    cols = st.columns(spec or len(items))
    for c, item in zip(cols, items):
        c.markdown(item, unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380, legend_top: bool = True) -> go.Figure:
    """Apply the app's Plotly look (white template, fonts, legend below the plot)."""
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#0F172A"),
        title_font=dict(size=15),
        # legend below the plot: never collides with the title, and works on narrow screens
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0) if legend_top else None,
        hoverlabel=dict(font_size=12),
    )
    return fig


def pm(d: dict, pct: bool = False, digits: int = 2) -> str:
    """'mean ± std' from a {'mean', 'std'} dict."""
    if pct:
        return f"{d['mean']:.0%} ± {d['std'] * 100:.0f} pts"
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def cluster_features(s: dict) -> dict:
    """The six clustering features of a student."""
    return {f: s[f] for f in C.CLUSTER_FEATURES}


def no_student_box(prefix: str) -> None:
    """Message plus example buttons when no student is loaded yet."""
    st.info("No student selected yet. Load a real example below, or enter a student on the "
            "**Performance Prediction** page.")
    example_buttons(prefix)

