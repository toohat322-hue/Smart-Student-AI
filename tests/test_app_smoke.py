"""Smoke tests for the Streamlit app (spec §10: test_app_smoke.py) with streamlit.testing.v1.AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.ui as ui

PAGES = ["app.py", "pages/1_Performance_Prediction.py", "pages/2_Student_Clustering.py", "pages/3_Study_Planner.py",
         "pages/4_Model_Evaluation.py", "pages/5_About_Project.py"]
TIMEOUT = 120
ROOT = Path(__file__).resolve().parent.parent


def app(page: str) -> AppTest:
    return AppTest.from_file(str(ROOT / page), default_timeout=TIMEOUT)


def run(page: str, **state) -> AppTest:
    at = app(page)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def texts(at: AppTest) -> str:
    return " ".join(str(x.value) for x in [*at.markdown, *at.caption, *at.success, *at.info, *at.subheader, *at.title])


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("subject", ["por", "mat"])
def test_every_page_loads_without_a_student(page, subject):
    at = run(page, subject=subject)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_with_an_example_student(page):
    at = app(page)
    at.query_params["example"] = "dropout"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["student"] is not None


@pytest.mark.parametrize("example", list(ui.EXAMPLES))
def test_load_example_student_and_predict(example):
    at = run("pages/1_Performance_Prediction.py")
    at.button(key=f"pred_{example}").click().run()
    assert not at.exception
    s = at.session_state["student"]
    assert s is not None and set(s) == set(ui.STUDENT_FEATURES)
    assert "real record" in at.session_state["student_label"]
    actual = at.session_state["student_actual"]
    if example == "dropout":
        assert actual["G3"] == 0
    body = texts(at)
    assert ("At Risk – recommend follow-up" in body) or ("Likely to Pass" in body)
    assert "screening tool, not a verdict" in body


def test_prediction_runs_from_the_form():
    at = run("pages/1_Performance_Prediction.py")
    assert at.session_state["student"] is None
    at.select_slider(key="f_failures").set_value(3)
    next(b for b in at.button if b.label == "Predict").click().run()
    assert not at.exception
    assert at.session_state["student"]["failures"] == 3
    assert at.session_state["student_label"] == "Entered by hand"
    body = texts(at)
    assert ("At Risk – recommend follow-up" in body) or ("Likely to Pass" in body)


def test_midterm_scenario_shows_grades_and_predicts():
    at = run("pages/1_Performance_Prediction.py", scenario="midterm")
    at.button(key="pred_pass").click().run()
    assert not at.exception
    assert at.slider(key="f_G1") is not None and at.slider(key="f_G2") is not None
    assert "Mid-term" in texts(at)


def test_student_flows_to_clustering_and_planner():
    at = app("pages/2_Student_Clustering.py")
    at.query_params["example"] = "at_risk"
    at.run()
    assert any("Your cluster" in str(h.value) for h in at.subheader)

    at = app("pages/3_Study_Planner.py")
    at.query_params["example"] = "at_risk"
    at.run()
    assert "this student" in str(at.selectbox[0].format_func(at.selectbox[0].value))


def test_planner_simulation_runs():
    at = run("pages/3_Study_Planner.py")
    at.toggle(key="sim_animate").set_value(False)
    at.button(key="sim_run").click().run()
    assert not at.exception
    assert any(("Mastered in" in str(s.value)) or ("Not mastered" in str(s.value)) for s in at.success)


def test_missing_models_show_retrain_message(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "ROOT_DIR", tmp_path)
    at = run("pages/1_Performance_Prediction.py")
    assert at.error and "python scripts/train_all.py" in at.error[0].value
