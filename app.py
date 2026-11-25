"""Smart Student AI — entry point: navigation + Home dashboard.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from src import config as C  # noqa: E402
from src import ui  # noqa: E402


def home() -> None:
    ui.setup_page("Home", "🎓")
    m = ui.require(sections=["eda", "supervised", "supervised_repeated_splits", "clustering", "rl"])
    subj = ui.subject()
    eda = m["eda"]["subjects"]
    rep = m["supervised_repeated_splits"]["subjects"][subj]["early"]
    n_splits = m["supervised_repeated_splits"]["subjects"][subj]["early"]["n_splits"]

    st.title("Smart Student AI")
    st.markdown("Spot students who may need help **early**, see which **study patterns** they share with others, "
                "and explore a **simulated study planner**, all from one student profile.")

    ui.html_row([
        ui.kpi(f"{eda['por']['rows'] + eda['mat']['rows']:,}", "student records",
               f"{eda['por']['rows']} Portuguese · {eda['mat']['rows']} Mathematics"),
        ui.kpi(f"{eda[subj]['pass_rate']:.0%}", f"pass rate · {ui.SUBJECT_LABELS[subj]}",
               f"{eda[subj]['n_at_risk']} of {eda[subj]['rows']} students below 10/20"),
        ui.kpi(ui.pm(rep["rf_tuned"]["at_risk_recall"], pct=True), "At-Risk students caught (Early model)",
               f"mean ± std over {n_splits} random test splits, before any grade is known"),
    ])
    st.write("")

    st.subheader("Three ways of learning from the same data")
    ui.html_row([
        ui.card("1 · Predict (supervised)",
                "A Random Forest trained on past students flags who is <b>At Risk</b> of failing (final grade < 10), "
                "either at the start of the year or after the second period."),
        ui.card("2 · Group (unsupervised)",
                "K-Means finds groups of students with similar study and lifestyle habits, <b>without</b> looking "
                "at their grades."),
        ui.card("3 · Plan (reinforcement learning)",
                "A Q-learning agent learns which study activity to do next in a <b>simulated</b> study session, "
                "for each student group."),
    ])
    c1, c2, c3 = st.columns(3)
    c1.page_link(ui.PAGES["predict"], label="Go to prediction", icon="🎯")
    c2.page_link(ui.PAGES["cluster"], label="Go to clustering", icon="🧩")
    c3.page_link(ui.PAGES["plan"], label="Go to study planner", icon="🗺️")

    st.divider()
    st.subheader("Start with a student")
    st.markdown("Enter a student once: the prediction, their group and their study plan all follow them through "
                "the app.")
    if st.button("Start with a student →", type="primary"):
        st.switch_page(ui.PAGES["predict"])
    ui.example_buttons("home")
    if ui.student() is not None:
        st.success(f"Loaded: {st.session_state['student_label']}. Open any page; it uses this student.")

    st.caption(f"Subject: {ui.SUBJECT_LABELS[subj]} (change it in the sidebar). Model quality and limitations: "
               "see **Model Evaluation** and **About Project**.")
    st.markdown(f"🌐 Live app: [{C.APP_URL}]({C.APP_URL}) · 💻 Source code: [GitHub]({C.REPO_URL})")


PAGES = [
    st.Page(home, title="Home", icon="🎓", default=True),
    st.Page(ui.PAGES["predict"], url_path="prediction", title="Performance Prediction", icon="🎯"),
    st.Page(ui.PAGES["cluster"], url_path="clustering", title="Student Clustering", icon="🧩"),
    st.Page(ui.PAGES["plan"], url_path="planner", title="Study Planner", icon="🗺️"),
    st.Page(ui.PAGES["eval"], url_path="evaluation", title="Model Evaluation", icon="📊"),
    st.Page(ui.PAGES["about"], url_path="about", title="About Project", icon="ℹ️"),
]
st.navigation(PAGES).run()
