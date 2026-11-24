"""Student Clustering — study / lifestyle groups found by K-Means (unsupervised)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src import config as C  # noqa: E402
from src import ui  # noqa: E402
from src.data import FEATURE_SPEC  # noqa: E402
from src.unsupervised import FEATURE_LABELS, assign_cluster  # noqa: E402

ui.setup_page("Student Clustering", "🧩")
subj = ui.subject()
m = ui.require(models=[f"kmeans_{subj}.joblib"], sections=["clustering"])
cl = m["clustering"]["subjects"][subj]
profiles = cl["profile"]["clusters"]

st.title("Student Clustering")
st.markdown(f"K-Means groups **{ui.SUBJECT_LABELS[subj]}** students by six study and lifestyle habits "
            f"({', '.join(FEATURE_LABELS[f] for f in C.CLUSTER_FEATURES)}). Grades are **not** used to form "
            "the groups; pass rates are added afterwards only to describe them.")


@st.cache_resource(show_spinner=False)
def pca_space(subject: str):
    pipe = ui.kmeans(subject)
    X = ui.subject_data(subject)[C.CLUSTER_FEATURES]
    Z = pipe[:-1].transform(X)
    pca = PCA(n_components=2, random_state=C.RANDOM_STATE).fit(Z)
    return pipe, pca, pca.transform(Z), pipe[-1].labels_


# ---------------------------------------------------------------- the student
s = ui.student()
if s is None:
    ui.no_student_box("clu")
else:
    def _update_habits() -> None:
        new = dict(st.session_state["student"])
        for f in C.CLUSTER_FEATURES:
            new[f] = st.session_state[f"c_{f}"]
        if new != st.session_state["student"]:
            ui.set_student(new, "Edited by hand")

    for f in C.CLUSTER_FEATURES:
        st.session_state[f"c_{f}"] = s[f]
    with st.expander("Edit this student's habits"):
        with st.form("habits"):
            cols = st.columns(3)
            for i, f in enumerate(C.CLUSTER_FEATURES):
                lo, hi = FEATURE_SPEC[f]
                if f == "absences":
                    cols[i % 3].number_input(f"{ui.label(f)} ({lo}–{hi})", min_value=lo, max_value=hi, step=1,
                                             key=f"c_{f}")
                else:
                    cols[i % 3].select_slider(ui.label(f), options=list(range(lo, hi + 1)), key=f"c_{f}",
                                              format_func=lambda v, f=f: ui.fmt_value(f, v))
            st.form_submit_button("Update student", on_click=_update_habits)

    a = assign_cluster(subj, ui.cluster_features(s))
    prof = a["profile"]
    overall = cl["profile"]
    traits = sorted(prof["feature_z"].items(), key=lambda kv: -abs(kv[1]))
    trait_txt = "; ".join(
        f"{'more' if z > 0 else 'less'} {FEATURE_LABELS[f]} than average ({z:+.1f} SD)" for f, z in traits[:3])
    st.subheader(f"Your cluster: {a['name']}")
    ui.html_row([
        ui.kpi(f"{prof['size']}", "students in this cluster",
               f"you are similar to {prof['size']} of {cl['n_students']} {ui.SUBJECT_LABELS[subj]} students"),
        ui.kpi(f"{prof['pass_rate']:.0%}", "pass rate in this cluster",
               f"all students: {overall['overall_pass_rate']:.0%}"),
        ui.kpi(f"{prof['mean_G3']:.1f}", "average final grade (0–20)",
               f"all students: {overall['overall_mean_G3']:.1f}"),
    ])
    st.caption(f"What sets this group apart: {trait_txt}. Pass rate and grade describe the group; they were not "
               "used to form it.")

st.divider()

# ---------------------------------------------------------------- charts
pipe, pca, coords, labels = pca_space(subj)
left, right = st.columns(2)
with left:
    rows_log, rows_raw = cl["k_selection"]["log1p"], cl["k_selection"]["raw"]
    best = max(r["silhouette"] for r in rows_log)
    fig = go.Figure()
    fig.add_hrect(y0=best - cl["choice"]["tolerance"], y1=best, fillcolor=ui.BLUE, opacity=0.08, line_width=0,
                  annotation_text=f"within {cl['choice']['tolerance']} of best", annotation_position="top left")
    fig.add_scatter(x=[r["k"] for r in rows_log], y=[r["silhouette"] for r in rows_log], mode="lines+markers",
                    name="log1p(absences), used", line=dict(color=ui.BLUE, width=3))
    fig.add_scatter(x=[r["k"] for r in rows_raw], y=[r["silhouette"] for r in rows_raw], mode="lines+markers",
                    name="raw absences", line=dict(color=ui.GRAY, dash="dash"))
    fig.add_vline(x=cl["k"], line_dash="dot", line_color=ui.TEXT2, annotation_text=f"chosen k = {cl['k']}")
    fig.update_layout(title="Silhouette score by number of clusters", xaxis_title="k", yaxis_title="silhouette")
    st.plotly_chart(ui.style(fig, 360), width="stretch")
    st.caption(f"Why k = {cl['k']}: {cl['choice']['reason']}")

with right:
    fig = go.Figure()
    for p in profiles:
        mask = labels == p["cluster"]
        fig.add_scatter(x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=f"{p['cluster']}: {p['name']}",
                        marker=dict(size=6, color=ui.SERIES[p["cluster"]], opacity=0.55,
                                    line=dict(width=0.5, color="white")),
                        hovertemplate=f"{p['name']}<extra></extra>")
    if s is not None:
        sp = pca.transform(pipe[:-1].transform(pd.DataFrame([ui.cluster_features(s)])))[0]
        fig.add_scatter(x=[sp[0]], y=[sp[1]], mode="markers+text", name="this student", text=["this student"],
                        textposition="top center", marker=dict(size=18, symbol="star", color="#0F172A",
                                                               line=dict(width=2, color="white")))
    ev = pca.explained_variance_ratio_
    fig.update_layout(title=f"Students in 2-D (PCA keeps {ev.sum():.0%} of the variation)",
                      xaxis_title=f"PC1 ({ev[0]:.0%})", yaxis_title=f"PC2 ({ev[1]:.0%})",
                      legend=dict(orientation="h", y=-0.2, yanchor="top", x=0))
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

ui.note(f"<b>Read the groups with care.</b> The silhouette score at the chosen k is only "
        f"{cl['silhouette']:.2f} (scale −1 to 1). The six habits do not split students into clearly separate types; "
        "the clusters are a useful <i>partition</i> of a continuum, not natural student categories, and never a "
        "judgement about a student.")

st.subheader("Cluster profiles")
table = pd.DataFrame([{
    "Cluster": f"{p['cluster']}: {p['name']}",
    "Students": p["size"],
    **{FEATURE_LABELS[f]: round(p["feature_means"][f], 2) for f in C.CLUSTER_FEATURES},
    "Average G3": round(p["mean_G3"], 1),
    "Pass rate": f"{p['pass_rate']:.0%}",
    "Final grade 0": p["n_g3_zero"],
} for p in profiles] + [{
    "Cluster": "All students", "Students": cl["n_students"],
    **{FEATURE_LABELS[f]: round(cl["profile"]["overall_means"][f], 2) for f in C.CLUSTER_FEATURES},
    "Average G3": round(cl["profile"]["overall_mean_G3"], 1), "Pass rate": f"{cl['profile']['overall_pass_rate']:.0%}",
    "Final grade 0": cl["g3_zero"]["n_g3_zero"],
}])
st.dataframe(table, hide_index=True, width="stretch")
st.caption("Habit columns are averages on the questionnaire scales (study time 1–4, others 1–5, absences = count). "
           "Names come from a rule: the habits furthest (≥ 0.5 SD) from the average student.")
