"""About Project — data, methods, limitations, future work."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import config as C  # noqa: E402
from src import ui  # noqa: E402
from src.supervised import TARGET_AT_RISK_RECALL  # noqa: E402

MERGE_KEYS = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason",
              "nursery", "internet"]  # the 13 attributes the UCI merge script (student-merge.R) joins on


@st.cache_data(show_spinner=False)
def students_in_both() -> int:
    return len(ui.subject_data("mat").merge(ui.subject_data("por"), on=MERGE_KEYS))

ui.setup_page("About Project", "ℹ️")
m = ui.require(sections=["eda", "supervised", "supervised_repeated_splits", "clustering", "rl"])
eda = m["eda"]["subjects"]
cl = m["clustering"]["subjects"]
rl = m["rl"]
rep = m["supervised_repeated_splits"]["subjects"]

st.title("About this project")

st.subheader("Data")
st.markdown(
    f"**UCI Student Performance** (dataset #320): questionnaire answers and grades of secondary-school students from "
    f"two Portuguese schools, 2005–2006. Two separate files: **Portuguese** ({eda['por']['rows']} students, "
    f"{eda['por']['pass_rate']:.0%} pass) and **Mathematics** ({eda['mat']['rows']} students, "
    f"{eda['mat']['pass_rate']:.0%} pass), {eda['por']['columns']} attributes each. {students_in_both()} students "
    "appear in both "
    "files, so the subjects are always modelled separately, never mixed in one training set.\n\n"
    "- Target: **pass** if the final grade G3 ≥ 10 out of 20, otherwise **At Risk**.\n"
    "- **License:** CC BY 4.0, used with attribution.\n"
    "- **Citation:** Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5TG7T\n"
    "- **Original study:** Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student "
    "Performance.* Proc. 5th FUBUTEC, pp. 5–12.")

st.subheader("Methods")
st.markdown(
    "**1 · Supervised (prediction).** Random Forest with balanced class weights, tuned by 5-fold grid search, compared "
    "with a majority-class baseline and logistic regression. Two scenarios: **Early** (no grades, no sex) and "
    "**Mid-term** (adds the period 1 and 2 grades). Absences may only *lower* the predicted pass chance (monotonic "
    "constraint). The alert threshold is tuned on training data to catch at least "
    f"{TARGET_AT_RISK_RECALL:.0%} of At-Risk students, never below 0.5. Quality is reported over "
    f"{rep['por']['early']['n_splits']} random train/test splits.\n\n"
    f"**2 · Unsupervised (grouping).** K-Means on six habits (absences log-transformed, all standardised). "
    f"k chosen by silhouette with a preference for 3 when close: Portuguese k = {cl['por']['k']}, "
    f"Mathematics k = {cl['mat']['k']}. Group names are generated from each group's most distinctive habits.\n\n"
    f"**3 · Reinforcement learning (planning).** Tabular Q-learning (learning rate {rl['config']['alpha_schedule']}, "
    f"γ = {rl['config']['gamma']}, {rl['config']['episodes']:,} episodes) in a **simulated** study session with "
    "states (knowledge 0–4, energy low/mid/high) and three activities. One environment per student cluster; its "
    "parameters come from a documented formula on the cluster's average habits.")

st.subheader("Limitations")
g3z = sum(eda[s]["g3_zero_count"] for s in eda)
st.markdown(
    "- **Old, local data:** two Portuguese schools in 2005–2006; results may not generalise to other schools or "
    "years.\n"
    f"- **Small samples** ({eda['mat']['rows']} / {eda['por']['rows']} students), so numbers vary between splits; "
    "we report mean ± std over many splits, and one split (the saved model's) happens to be unusually hard.\n"
    "- **Self-reported categories:** study time, free time, going out etc. are questionnaire categories, not "
    "measurements.\n"
    "- **Mid-term looks better than it is useful:** its accuracy comes mainly from the period 2 grade, which is "
    "almost the final grade. The Early model is the honest one.\n"
    f"- **Dropouts:** {g3z} students have a final grade of 0 and 0 recorded absences; they most likely left or did "
    "not sit the exam. They are kept as At Risk, and the absences effect is constrained so they cannot teach the "
    "model that perfect attendance is risky.\n"
    "- **Screening, not a verdict:** the At-Risk alert trades missed students against false alerts; many flagged "
    "students go on to pass.\n"
    "- **Fairness:** recall differs between girls and boys in some models (see Model Evaluation); groups are small "
    "and the estimate is uncertain.\n"
    f"- **Weak clusters:** silhouette scores are low ({min(cl[s]['silhouette'] for s in cl):.2f}–"
    f"{max(cl[s]['silhouette'] for s in cl):.2f}); clusters partition a continuum of habits, they are not natural "
    "types and never a judgement of a student.\n"
    "- **Simulated RL:** the data has no record of study actions and their effects. The study environments are "
    "assumptions; the planner illustrates the method and is not validated teaching advice. "
    + ("The best strategy turned out to be the same for every cluster; clusters only change the expected outcome."
       if rl["policy_differences"]["n_distinct_optimal_policies"] == 1 else ""))

st.subheader("Future work")
st.markdown(
    "- **Cluster-specific action effectiveness** (e.g. videos helping weaker students more) would give different "
    "plans per cluster, but needs data on which study activities students used and how they progressed.\n"
    "- Validate the Early model on newer data from other schools before any real use.\n"
    "- Collect attendance records that distinguish 'no absences' from 'not recorded'.\n"
    "- Calibrate the alert threshold with teachers, based on how many follow-ups they can actually do.")

st.subheader("Project")
st.markdown("Built with Python, scikit-learn, Streamlit and Plotly. All numbers in this app are read from "
            "`reports/metrics.json` or computed live from the saved models; `python scripts/train_all.py` "
            "regenerates everything. Design decisions are logged in `DECISIONS.md`.")
st.markdown("\n".join([
    f"- **Team:** {C.PROJECT_TEAM}",
    f"- **Course:** {C.PROJECT_COURSE}",
    f"- **Instructor:** {C.PROJECT_INSTRUCTOR}",
]))
