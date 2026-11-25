"""Performance Prediction — screening a student for follow-up (supervised)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui  # noqa: E402
from src.data import FEATURE_SPEC  # noqa: E402
from src.preprocessing import feature_columns  # noqa: E402
from src.supervised import TARGET_AT_RISK_RECALL, InputValidationError, predict  # noqa: E402

ui.setup_page("Performance Prediction", "🎯")
subj = ui.subject()
m = ui.require(models=ui.classifier_files(subj), sections=["supervised", "supervised_repeated_splits"])

st.title("Performance Prediction")
st.markdown(f"Screen a **{ui.SUBJECT_LABELS[subj]}** student for follow-up. The model flags students whose "
            "final grade is likely to fall below 10/20.")

ui.example_buttons("pred")

# ---------------------------------------------------------------- scenario
def _on_scenario() -> None:
    st.session_state["scenario"] = st.session_state["_scenario_widget"]


st.radio("When are you predicting?", list(ui.SCENARIO_LABELS), format_func=ui.SCENARIO_LABELS.get,
         index=list(ui.SCENARIO_LABELS).index(st.session_state["scenario"]), horizontal=True,
         key="_scenario_widget", on_change=_on_scenario,
         help="Early uses no grades at all. Mid-term also uses the period 1 and 2 grades (G1, G2), which makes it "
              "much more accurate but only available later in the year.")
scenario = st.session_state["scenario"]
features = feature_columns(scenario)

# ---------------------------------------------------------------- form
base = ui.student() or ui.default_student(subj)
for f in ui.STUDENT_FEATURES:  # initialise widget state once (before the widgets exist)
    st.session_state.setdefault(f"f_{f}", base[f])


def widget(f: str, container) -> None:
    spec, info, key = FEATURE_SPEC[f], ui.FIELD_INFO[f], f"f_{f}"
    if isinstance(spec, list) and set(spec) == {"yes", "no"}:
        container.radio(info["label"], ["yes", "no"], key=key, horizontal=True, format_func=str.capitalize)
    elif isinstance(spec, list):
        container.selectbox(info["label"], spec, key=key, format_func=lambda v: ui.fmt_value(f, v))
    elif "options" in info:
        lo, hi = spec
        container.select_slider(info["label"], options=list(range(lo, hi + 1)), key=key,
                                format_func=lambda v: ui.fmt_value(f, v))
    elif f == "absences":
        lo, hi = spec
        container.number_input(f"{info['label']} ({lo}–{hi})", min_value=lo, max_value=hi, step=1, key=key)
    else:
        lo, hi = spec
        container.slider(info["label"], min_value=lo, max_value=hi, step=1, key=key)


def _submit() -> None:
    # Callbacks run before the script: widgets not shown last run (e.g. G1/G2 in Early) may have no state.
    fallback = st.session_state.get("student") or ui.default_student(ui.subject())
    values = {f: st.session_state.get(f"f_{f}", fallback[f]) for f in ui.STUDENT_FEATURES}
    if values != st.session_state.get("student"):
        st.session_state["student_label"] = "Entered by hand" if st.session_state.get("student") is None \
            else "Edited by hand"
        st.session_state["student_actual"] = None
    st.session_state["student"] = values


key = ui.key_fields(scenario)
order = (["G1", "G2"] if scenario == "midterm" else []) + [f for grp in ui.FORM_GROUPS.values() for f in grp]
key = sorted(key, key=order.index)
rest = [f for f in features if f not in key]

with st.form("student_form"):
    st.markdown(f"**Key questions** ({len(key)} of {len(features)})")
    cols = st.columns(3)
    for i, f in enumerate(key):
        widget(f, cols[i % 3])
    with st.expander(f"More details (optional, {len(rest)} questions): typical values are used if you skip them"):
        for name, fields in ui.FORM_GROUPS.items():
            shown = [f for f in fields if f in rest]
            if not shown:
                continue
            st.markdown(f"###### {name}")
            cols = st.columns(3)
            for i, f in enumerate(shown):
                widget(f, cols[i % 3])
    st.caption("Key questions = the inputs that matter most to the models (permutation importance, both subjects) "
               "plus the six habits used for grouping and the study planner. Sex is deliberately not used.")
    st.form_submit_button("Predict", type="primary", on_click=_submit)

# ---------------------------------------------------------------- result
s = ui.student()
if s is None:
    st.info("Fill in the form and press **Predict**, or load an example student above.")
    st.stop()

try:
    out = predict(subj, scenario, s)
except InputValidationError as exc:
    st.error(str(exc))
    st.stop()

model = ui.classifier(subj, scenario)
thr = float(model.threshold)
p = out["proba_pass"]
at_risk = out["label"] == "At Risk"

st.divider()
st.subheader(f"Result · {ui.SCENARIO_LABELS[scenario]}")
left, right = st.columns([1, 1])
with left:
    title = "At Risk – recommend follow-up" if at_risk else "Likely to Pass"
    text = (f"Estimated chance of passing: <b>{p:.0%}</b>, below the alert threshold of {thr:.0%}."
            if at_risk else f"Estimated chance of passing: <b>{p:.0%}</b>, at or above the alert threshold "
            f"of {thr:.0%}.")
    st.markdown(f'<div class="ssa-result {"risk" if at_risk else "pass"}"><div class="t">{title}</div>'
                f'<div>{text}</div><div style="font-size:.85rem;margin-top:.4rem">Student: '
                f'{st.session_state.get("student_label") or "Entered by hand"}</div></div>', unsafe_allow_html=True)
    actual = st.session_state.get("student_actual")
    if actual and actual["subject"] == subj:
        outcome = "passed" if actual["pass"] else ("dropped out / did not sit the final exam" if actual["G3"] == 0
                                                     else "failed")
        st.caption(f"Real outcome in the data (not used by the model): final grade G3 = {actual['G3']}/20, "
                   f"the student {outcome}.")

    st.markdown("**What drives this prediction most**")
    for fac in out["top_factors"]:
        f = fac["feature"]
        ref = ui.fmt_value(f, fac["reference"])
        ref_word = "typical student" if fac["reference_type"] == "most common" else "median student"
        delta = fac["delta_proba_pass"] * 100
        if fac["effect"] == "neutral":
            eff = "about the same as for the " + ref_word
        else:
            pts = round(abs(delta))
            eff = (f"{'raises' if delta > 0 else 'lowers'} the pass chance by {pts} point{'s' if pts != 1 else ''} "
                   f"vs the {ref_word}")
        icon = {"better": "🟦", "worse": "🟧", "neutral": "⬜"}[fac["effect"]]
        st.markdown(f"{icon} **{ui.label(f)}: {ui.fmt_value(f, fac['value'])}** (typical: {ref}) – {eff}.")
    st.caption("Factors = the 3 most important inputs overall (permutation importance). The effect is measured "
               "by changing only that input to the typical value and re-running the model.")

with right:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Chance of passing"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": ui.RISK_COLOR if at_risk else ui.PASS_COLOR, "thickness": 0.3},
            "steps": [{"range": [0, thr * 100], "color": "#fbe0d4"}, {"range": [thr * 100, 100], "color": "#dbe9fb"}],
            "threshold": {"line": {"color": "#0F172A", "width": 4}, "thickness": 0.85, "value": thr * 100},
        },
    ))
    st.plotly_chart(ui.style(fig, height=300, legend_top=False), width="stretch")
    st.caption(f"Black line = alert threshold ({thr:.1%}), tuned so the model catches at least "
               f"{TARGET_AT_RISK_RECALL:.0%} of At-Risk students in cross-validation. Below it, the student is flagged.")

# ---------------------------------------------------------------- screening note
rep = m["supervised_repeated_splits"]["subjects"][subj][scenario]
split = m["supervised"]["subjects"][subj]["split"]
n_pass_test = split["n_test"] - split["test_at_risk"]
fa_rate = rep["rf_tuned"]["false_alerts"]["mean"] / n_pass_test
prec = rep["rf_tuned"]["at_risk_precision"]["mean"]
rec = rep["rf_tuned"]["at_risk_recall"]["mean"]
ui.note(
    f"<b>This is a screening tool, not a verdict.</b> It flags students for a conversation, never for a decision. "
    f"Over {rep['n_splits']} random test splits, this {ui.SCENARIO_LABELS[scenario].split(' (')[0]} model caught "
    f"<b>{rec:.0%}</b> of At-Risk students, but also raised a <b>false alert for about {fa_rate:.0%} of students "
    f"who went on to pass</b>; roughly {prec * 10:.0f} in 10 flagged students actually ended below 10/20."
)
