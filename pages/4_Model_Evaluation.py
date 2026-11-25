"""Model Evaluation — honest numbers for every model (headline = 20 random splits)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui  # noqa: E402
from src.data import TARGET_COL  # noqa: E402

ui.setup_page("Model Evaluation", "📊")
subj = ui.subject()
m = ui.require(sections=["eda", "supervised", "supervised_repeated_splits", "clustering", "rl"])
sup = m["supervised"]["subjects"][subj]
rep = m["supervised_repeated_splits"]["subjects"][subj]
SC = list(ui.SCENARIO_LABELS)
n_splits = rep["early"]["n_splits"]

st.title("Model Evaluation")
st.markdown(f"Subject: **{ui.SUBJECT_LABELS[subj]}**. Headline numbers are the **mean ± std over {n_splits} random "
            "train/test splits**, each with the full procedure (tuning, threshold, single test evaluation). The "
            "saved model was trained on one of those splits (seed 42); its numbers are shown as secondary.")

tabs = st.tabs(["Prediction quality", "Saved model (seed 42)", "Early vs Mid-term & threshold", "Fairness",
                "Data issues & fixes", "Study planner (RL)"])

MODEL_ROWS = [("dummy", "Dummy (always 'pass')"), ("logreg", "Logistic Regression · threshold 0.5"),
              ("logreg_tuned", "Logistic Regression · final threshold"),
              ("rf_default", "Random Forest · threshold 0.5"), ("rf_tuned", "Random Forest · final threshold (saved)")]

# ---------------------------------------------------------------- 1. headline
with tabs[0]:
    for sc in SC:
        st.markdown(f"#### {ui.SCENARIO_LABELS[sc]}")
        r = rep[sc]
        split = sup["split"]
        st.dataframe(pd.DataFrame([{
            "Model": name,
            "Macro F1": ui.pm(r[k]["f1_macro"]),
            "At-Risk recall": ui.pm(r[k]["at_risk_recall"]),
            "At-Risk precision": ui.pm(r[k]["at_risk_precision"]),
            "ROC-AUC": ui.pm(r[k]["roc_auc"]),
            "False alerts per test set": f"{r[k]['false_alerts']['mean']:.1f} of {split['n_test'] - split['test_at_risk']}",
        } for k, name in MODEL_ROWS]), hide_index=True, width="stretch")
        d = r["rf_minus_logreg_tuned"]
        st.caption(
            "Same splits, both models at their own tuned threshold, Random Forest minus Logistic Regression: "
            + " · ".join(f"{lbl} {d[k]['mean']:+.3f} (SE {d[k]['se']:.3f})" for k, lbl in (
                ("f1_macro", "macro F1"), ("at_risk_recall", "At-Risk recall"),
                ("at_risk_precision", "At-Risk precision"))) + f" · false alerts {d['false_alerts']['mean']:+.1f}")
    st.caption("At-Risk recall = share of students who really ended below 10/20 that the model flagged (the headline: "
               "missing a struggling student is the costly error). Precision = share of flagged students who really "
               "ended below 10/20. False alerts = passing students flagged.")
    with st.expander("Same table for the single seed-42 split (this saved model's split)"):
        for sc in SC:
            mm = sup[sc]["models"]
            rows = [("Dummy", mm["dummy"]["test_default"]),
                    ("Logistic Regression · threshold 0.5", mm["logreg"]["test_default"]),
                    ("Logistic Regression · final threshold", mm["logreg"]["test_tuned"]),
                    ("Random Forest · threshold 0.5", mm["rf"]["test_default"]),
                    ("Random Forest · final threshold (saved)", mm["rf"]["test_tuned"])]
            st.markdown(f"**{ui.SCENARIO_LABELS[sc]}**")
            st.dataframe(pd.DataFrame([{
                "Model": n, "Accuracy": round(t["accuracy"], 3), "Macro F1": round(t["f1_macro"], 3),
                "At-Risk recall": round(t["at_risk_recall"], 3), "At-Risk precision": round(t["at_risk_precision"], 3),
                "ROC-AUC": round(t["roc_auc"], 3), "False alerts": f"{t['false_alerts']} of {t['n'] - t['n_at_risk']}",
            } for n, t in rows]), hide_index=True, width="stretch")
        z = rep["early"]["seed42"]
        st.caption(f"Seed 42 is a hard split for {ui.SUBJECT_LABELS[subj]} Early: macro F1 {z['rf_default_f1_macro']:.3f} "
                   f"vs {rep['early']['rf_default']['f1_macro']['mean']:.3f} on average "
                   f"({z['share_of_seeds_with_lower_f1_macro']:.0%} of the {n_splits} splits score lower).")

# ---------------------------------------------------------------- 2. seed-42 details
with tabs[1]:
    sc = st.radio("Scenario", SC, format_func=ui.SCENARIO_LABELS.get, horizontal=True, key="eval_sc")
    e = sup[sc]
    rf = e["models"]["rf"]
    st.caption(f"This saved model's split: {sup['split']['n_train']} training / {sup['split']['n_test']} test students "
               f"({sup['split']['test_at_risk']} At Risk). Final threshold {e['threshold']:.3f}; "
               f"best parameters {rf['best_params']}.")
    c1, c2 = st.columns(2)
    with c1:
        t = rf["test_tuned"]
        cm = np.array(t["confusion_matrix"]["matrix"])
        labels = t["confusion_matrix"]["labels"]
        fig = go.Figure(go.Heatmap(z=cm, x=[f"predicted {x}" for x in labels], y=[f"actual {x}" for x in labels],
                                   text=cm, texttemplate="%{text}", textfont=dict(size=22), colorscale="Blues",
                                   showscale=False))
        fig.update_layout(title=f"Confusion matrix (final threshold) · recall {t['at_risk_recall']:.0%}",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(ui.style(fig, 340, legend_top=False), width="stretch")
    with c2:
        fig = go.Figure()
        for key, name, col, dash in (("rf", "Random Forest", ui.BLUE, None), ("logreg", "Logistic Regression", ui.ORANGE, None),
                                     ("dummy", "Dummy", ui.GRAY, "dash")):
            t = e["models"][key]["test_default"]
            fig.add_scatter(x=t["roc_curve"]["fpr"], y=t["roc_curve"]["tpr"], mode="lines",
                            name=f"{name} (AUC {t['roc_auc']:.2f})", line=dict(color=col, dash=dash, width=2.5))
        fig.update_layout(title="ROC curve", xaxis_title="false positive rate", yaxis_title="true positive rate")
        st.plotly_chart(ui.style(fig, 340), width="stretch")
    imp = e["permutation_importance"][::-1]
    fig = go.Figure(go.Bar(x=[d["mean"] for d in imp], y=[ui.label(d["feature"]) for d in imp], orientation="h",
                           marker_color=ui.BLUE, error_x=dict(type="data", array=[d["std"] for d in imp],
                                                              color=ui.TEXT2)))
    fig.update_layout(title="Permutation importance (drop in test macro F1 when the input is shuffled)",
                      xaxis_title="importance")
    st.plotly_chart(ui.style(fig, 380, legend_top=False), width="stretch")

# ---------------------------------------------------------------- 3. early vs midterm, threshold
with tabs[2]:
    keys = [("f1_macro", "Macro F1"), ("at_risk_recall", "At-Risk recall"), ("at_risk_precision", "At-Risk precision"),
            ("roc_auc", "ROC-AUC")]
    fig = go.Figure()
    for sc, col in (("early", ui.BLUE), ("midterm", ui.AQUA)):
        fig.add_bar(x=[n for _, n in keys], y=[rep[sc]["rf_tuned"][k]["mean"] for k, _ in keys], name=ui.SCENARIO_LABELS[sc],
                    marker_color=col, error_y=dict(type="data", array=[rep[sc]["rf_tuned"][k]["std"] for k, _ in keys]),
                    text=[f"{rep[sc]['rf_tuned'][k]['mean']:.2f}" for k, _ in keys], textposition="inside")
    fig.update_layout(title=f"Random Forest, Early vs Mid-term ({n_splits} splits, mean ± std)", yaxis=dict(range=[0, 1.05]),
                      barmode="group")
    st.plotly_chart(ui.style(fig, 380), width="stretch")
    st.caption("Mid-term is far more accurate mainly because the period 2 grade (G2) is almost the final grade. The Early "
               "model is the honest, useful one: it works before any grade exists.")

    rows = []
    for sc in SC:
        tt = sup[sc]["models"]["rf"]["threshold_tuning"]
        rows.append({
            "Scenario": ui.SCENARIO_LABELS[sc], "Final threshold (saved)": f"{tt['threshold']:.3f}",
            "At-Risk recall @0.5": ui.pm(rep[sc]["rf_default"]["at_risk_recall"]),
            "At-Risk recall @final": ui.pm(rep[sc]["rf_tuned"]["at_risk_recall"]),
            "False alerts @0.5": f"{rep[sc]['rf_default']['false_alerts']['mean']:.1f}",
            "False alerts @final": f"{rep[sc]['rf_tuned']['false_alerts']['mean']:.1f}",
        })
    st.markdown("#### Effect of the alert threshold")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(m["supervised"]["config"]["threshold_rule"] + ". The threshold is chosen on training data only. "
               "The floor at 0.5 stops it from lowering recall when the default already catches enough students.")

# ---------------------------------------------------------------- 4. fairness
with tabs[3]:
    st.markdown("Sex is **not** a model input, but we check whether At-Risk students of each sex are caught equally "
                f"often. Counts are pooled over the {n_splits} test sets (final threshold).")
    rows = []
    for s_ in ("por", "mat"):
        df = ui.subject_data(s_)
        real = df[df[TARGET_COL] == 0]["sex"].value_counts()
        for sc in SC:
            f = m["supervised_repeated_splits"]["subjects"][s_][sc]["fairness_pooled_final"]
            rows.append({
                "Subject": ui.SUBJECT_LABELS[s_], "Scenario": ui.SCENARIO_LABELS[sc].split(" (")[0],
                "Girls caught": f"{f['F']['at_risk_recall']:.0%} ({f['F']['n_caught']}/{f['F']['n_at_risk']})",
                "Boys caught": f"{f['M']['at_risk_recall']:.0%} ({f['M']['n_caught']}/{f['M']['n_at_risk']})",
                "Gap (points)": f"{(f['F']['at_risk_recall'] - f['M']['at_risk_recall']) * 100:+.0f}",
                "Real At-Risk students (F / M)": f"{real.get('F', 0)} / {real.get('M', 0)}",
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    times = n_splits * m["supervised"]["config"]["test_size"]
    ui.note(f"<b>Caveat:</b> the test sets overlap, so each student is counted about {times:.0f} times. The pooled counts show a "
            "direction, not a statistically tested difference. Fixing a gap would mean using sex at prediction time, "
            "which this project deliberately avoids.")

# ---------------------------------------------------------------- 5. data issues
with tabs[4]:
    e_ = m["eda"]["subjects"][subj]
    z = e_["g3_zero_check"]
    st.markdown("#### Students with a final grade of 0")
    st.markdown(
        f"- **{e_['g3_zero_count']}** {ui.SUBJECT_LABELS[subj]} students ({e_['g3_zero_share']:.1%}) have a final "
        f"grade of 0, yet their period 1 grade averages **{z['g3_zero']['G1_mean']:.1f}**/20 (other At-Risk students: "
        f"{z['other_at_risk']['G1_mean']:.1f}).\n"
        f"- **{z['g3_zero']['share_absences_eq_0']:.0%}** of them have 0 recorded absences "
        f"(other At-Risk students: {z['other_at_risk']['share_absences_eq_0']:.0%}); "
        f"{z['g3_zero']['share_G2_eq_0']:.0%} already had a period 2 grade of 0.\n"
        "- They look like students who **dropped out or did not sit the final exam**, and whose absences stopped being "
        "recorded. They stay in the data labelled At Risk: catching dropouts is the point of an early warning.")
    st.markdown("#### Fix: absences may only lower the pass chance")
    sc = st.radio("Scenario ", SC, format_func=ui.SCENARIO_LABELS.get, horizontal=True, key="eval_pd_sc")
    pdd = sup[sc]["absences_partial_dependence"]
    fig = go.Figure()
    for key, name, col, dash in (("rf_unconstrained", "Unconstrained model (before)", ui.GRAY, "dash"),
                                 ("rf_main", "Monotonic model (saved)", ui.BLUE, None)):
        curve = pdd[key]["curve_0_30"]
        fig.add_scatter(x=[int(k) for k in curve], y=list(curve.values()), mode="lines+markers", name=name,
                        line=dict(color=col, dash=dash, width=2.5), marker=dict(size=5))
    fig.update_layout(title="Average predicted pass chance if every student had this many absences",
                      xaxis_title="absences", yaxis_title="mean P(pass)", yaxis_tickformat=".0%")
    st.plotly_chart(ui.style(fig, 360), width="stretch")
    st.caption("Because the dropouts all show 0 absences, the unconstrained model could learn that '0 absences = "
               "risk'. The saved model is constrained so its pass probability never rises as absences rise.")

# ---------------------------------------------------------------- 6. RL
with tabs[5]:
    rl = m["rl"]
    keys = [(s_, c) for s_ in ("mat", "por") for c in sorted(rl["environments"][s_], key=int)]
    xs = [f"{ui.SUBJECT_LABELS[s_][:4]}. {c}" for s_, c in keys]
    c1, c2 = st.columns(2)
    for col, metric, title in ((c1, "match", "States where the learned action is optimal (of 12)"),
                               (c2, "gap", "Reward left on the table vs the exact optimum")):
        fig = go.Figure()
        for name, color, get in (("decaying α (saved)", ui.BLUE, lambda e: e),
                                 (f"constant α = {rl['config']['comparison_constant_alpha']}", ui.GRAY,
                                  lambda e: e["comparison_constant_alpha"])):
            ys = [get(rl["environments"][s_][c])["decision_states_matching_optimal"] if metric == "match"
                  else get(rl["environments"][s_][c])["optimal_minus_learned"]["mean"] for s_, c in keys]
            fig.add_bar(x=xs, y=ys, name=name, marker_color=color)
        fig.update_layout(title=title, barmode="group")
        col.plotly_chart(ui.style(fig, 340), width="stretch")
    st.caption(f"Learning rate: {rl['config']['alpha_schedule']} (saved) vs the constant α from the original plan. "
               "The optimum is computed exactly from the simulator, as a reference; the agent never sees it.")
    learned = [rl["environments"][s_][c]["learned_minus_best_fixed"]["mean"] for s_, c in keys]
    st.markdown(f"In every simulated environment the learned strategy beats the best single activity by "
                f"**{min(learned):+.1f} to {max(learned):+.1f}** reward. "
                f"{'The exact optimal strategy is identical for all clusters.' if rl['policy_differences']['n_distinct_optimal_policies'] == 1 else ''}")
