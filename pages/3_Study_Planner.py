"""Study Planner — Q-learning in a SIMULATED study-session environment (reinforcement learning)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config as C  # noqa: E402
from src import ui  # noqa: E402
from src.rl_agent import recommend, simulate_episode  # noqa: E402
from src.unsupervised import assign_cluster  # noqa: E402

ui.setup_page("Study Planner", "🗺️")
subj = ui.subject()
m = ui.require(models=["q_tables.joblib", f"kmeans_{subj}.joblib"], sections=["rl", "clustering"])
rl = m["rl"]
envs = rl["environments"][subj]
clusters = sorted(int(c) for c in envs)

st.title("Study Planner")
ui.note("<b>Simulation.</b> " + rl["honesty_note"].split(". ", 1)[1] +
        " The planner shows how reinforcement learning works; it is not validated study advice.")
st.write("")

# ---------------------------------------------------------------- which environment
s = ui.student()
student_cluster = None
if s is None:
    ui.no_student_box("plan")
else:
    student_cluster = assign_cluster(subj, ui.cluster_features(s))["cluster"]

names = {c: envs[str(c)]["cluster_name"] for c in clusters}
default = student_cluster if student_cluster is not None else clusters[0]
c = st.selectbox("Study environment (one per student cluster)", clusters, index=clusters.index(default),
                 format_func=lambda k: f"Cluster {k}: {names[k]}" + ("  ← this student" if k == student_cluster
                                                                     else ""),
                 key=f"plan_cluster_{subj}_{default}")
e = envs[str(c)]
ev = e["evaluation"]

# ---------------------------------------------------------------- strategy
st.subheader("The learned strategy")
same = rl["policy_differences"]["n_distinct_optimal_policies"] == 1
left, right = st.columns([1.15, 1])
with left:
    grid = e["policy"][:4]  # knowledge 4 = mastered (terminal)
    short = ["Quiz", "Video", "Review"]
    z = np.array(grid, dtype=float)
    text = [[short[int(v)] + ("" if e["optimal_policy"][k][j] == v else " *") for j, v in enumerate(row)]
            for k, row in enumerate(grid)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["low", "mid", "high"], y=[f"knowledge {k}" for k in range(4)], text=text,
        texttemplate="%{text}", textfont=dict(size=15, color="white"), showscale=False, zmin=-0.5, zmax=2.5,
        colorscale=[[0, ui.ACTION_COLORS[0]], [1 / 3, ui.ACTION_COLORS[0]], [1 / 3, ui.ACTION_COLORS[1]],
                    [2 / 3, ui.ACTION_COLORS[1]], [2 / 3, ui.ACTION_COLORS[2]], [1, ui.ACTION_COLORS[2]]],
        hovertemplate="%{y}, energy %{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(title="Best next activity by current state", xaxis_title="energy",
                      yaxis_title="", xaxis_side="bottom")
    st.plotly_chart(ui.style(fig, 340, legend_top=False), width="stretch")
    st.caption("Quiz = Practice Quizzes · Video = Watch Videos · Review = Review Notes. "
               "* = differs from the exact optimal action (computed from the simulator, for reference).")
with right:
    n_match, n_dec = e["decision_states_matching_optimal"], e["n_decision_states"]
    st.markdown(
        f"- **What decides the action is the student's current state** (knowledge, energy), not their cluster.\n"
        f"- The exact optimal strategy is {'**identical for all 8 clusters**' if same else 'different per cluster'}; "
        f"this cluster's learned strategy matches it in **{n_match} of {n_dec}** states.\n"
        "- In words: beginners learn from **videos**, more advanced students from **quizzes** when fresh; "
        "**review notes** whenever energy is mid or low (they restore energy).")
    st.markdown("**Ask the planner**")
    k_in = st.select_slider("Current knowledge", options=list(range(C.RL_N_KNOWLEDGE)), value=0, key="ask_k")
    e_in = st.select_slider("Current energy", options=[0, 1, 2], value=2, key="ask_e",
                            format_func=lambda v: ["low", "mid", "high"][v])
    r = recommend(subj, c, int(k_in), int(e_in))
    if r["action"] is None:
        st.success(r["action_name"])
    else:
        best = max(r["q_values"])
        st.success(f"Next: **{r['action_name']}**")
        st.caption("Expected long-run value of each option (Q-values): " + ", ".join(
            f"{a} {q:.1f}" + (" ✓" if q == best else "") for a, q in zip(C.RL_ACTIONS, r["q_values"])))

# ---------------------------------------------------------------- outcomes per cluster
st.subheader("What this cluster can expect")
ui.html_row([
    ui.kpi(f"{ev['learned']['mastery_rate']:.0%}", "reach full knowledge (mastery)",
           f"in {ev['learned']['n_episodes']:,} simulated sessions of ≤ {C.RL_MAX_STEPS} steps"),
    ui.kpi(f"{ev['learned']['mean_steps_to_mastery']:.1f}", "steps to mastery (when reached)",
           f"best single activity: {ev['best_fixed']['mean_steps_to_mastery']:.1f}"),
    ui.kpi(f"{ev['learned']['mean_reward']:.1f}", "average reward with the learned strategy",
           f"exact optimum {ev['optimal_reference']['mean_reward']:.1f} · random {ev['random']['mean_reward']:.1f}"),
])

left, right = st.columns(2)
with left:
    xs = [f"{k}: {names[k]}" for k in clusters]
    mastery = [envs[str(k)]["evaluation"]["learned"]["mastery_rate"] for k in clusters]
    colors = [ui.NAVY if k == c else "#9db4e0" for k in clusters]
    fig = go.Figure(go.Bar(x=mastery, y=xs, orientation="h", marker_color=colors,
                           text=[f"{v:.0%}" for v in mastery], textposition="outside"))
    fig.update_layout(title="Mastery rate by cluster (same strategy, different students)",
                      xaxis=dict(range=[0, 1.3], tickformat=".0%"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(ui.style(fig, 330, legend_top=False), width="stretch")
with right:
    series = [("learned", "Learned", ui.BLUE),
              ("best_fixed", "Always " + e["best_fixed_action"].split()[-1].lower(), ui.AQUA),
              ("random", "Random", ui.GRAY), ("always_quizzes", "Always quizzes", ui.YELLOW)]
    fig = go.Figure()
    for key, name, col in series:
        fig.add_bar(x=[name], y=[ev[key]["mean_reward"]], marker_color=col, name=name,
                    error_y=dict(type="data", array=[ev[key]["std_reward"]], color=ui.TEXT2, thickness=1),
                    text=[f"{ev[key]['mean_reward']:.1f}"], textposition="outside", cliponaxis=False)
    fig.add_hline(y=ev["optimal_reference"]["mean_reward"], line_dash="dot", line_color="#0F172A",
                  annotation_text=f"exact optimum {ev['optimal_reference']['mean_reward']:.1f}",
                  annotation_position="top right")
    d = e["learned_minus_best_fixed"]
    fig.update_layout(title="Average reward per strategy (± std)", showlegend=False, yaxis_title="reward",
                      xaxis_tickangle=0)
    st.plotly_chart(ui.style(fig, 330, legend_top=False), width="stretch")
    st.caption(f"Same {ev['learned']['n_episodes']:,} test sessions for every strategy. Learned beats the best single "
               f"activity by {d['mean']:+.1f} reward (standard error {d['se']:.1f}).")

# ---------------------------------------------------------------- replay
st.subheader("Simulate a study session")
cc1, cc2, cc3 = st.columns([1, 1, 2])
seed = cc1.number_input("Session seed", min_value=0, max_value=99_999, value=7, step=1, key="sim_seed")
animate = cc2.toggle("Animate", value=True, key="sim_animate")
run = cc3.button("▶ Simulate a study session", type="primary", key="sim_run")
if run:
    steps = simulate_episode(subj, c, seed=int(seed))
    chart, table_box, msg = st.empty(), st.empty(), st.empty()

    def draw(upto: int) -> None:
        shown = steps[:upto]
        fig = go.Figure()
        fig.add_scatter(x=[0] + [x["step"] for x in shown], y=[shown[0]["knowledge"]] + [x["next_knowledge"] for x in shown],
                        mode="lines+markers", name="knowledge (0–4)", line=dict(color=ui.BLUE, width=3, shape="hv"))
        fig.add_scatter(x=[0] + [x["step"] for x in shown], y=[shown[0]["energy"]] + [x["next_energy"] for x in shown],
                        mode="lines+markers", name="energy (0–2)", line=dict(color=ui.ORANGE, width=2, shape="hv"))
        fig.update_layout(title=f"Step {shown[-1]['step']}: {shown[-1]['action_name']} · reward so far "
                                f"{shown[-1]['cumulative_reward']:.0f}",
                          xaxis=dict(range=[0, C.RL_MAX_STEPS + 0.5], title="step"), yaxis=dict(range=[-0.3, 4.3]))
        chart.plotly_chart(ui.style(fig, 300), width="stretch", key=f"sim_chart_{upto}")
        table_box.dataframe(pd.DataFrame([{
            "Step": x["step"], "Knowledge": x["knowledge"], "Energy": ["low", "mid", "high"][x["energy"]],
            "Action": x["action_name"],
            "What happened": "skipped (absent)" if x["skipped"] else ("learned +1" if x["gained"] else "no gain")
                             + (" · energy hit 0" if x["hit_zero_energy"] else "") + (" · MASTERED" if x["mastered"] else ""),
            "Reward": x["reward"], "Total": x["cumulative_reward"],
        } for x in shown]), hide_index=True, width="stretch")

    for i in range(1, len(steps) + 1) if animate else [len(steps)]:
        draw(i)
        if animate:
            time.sleep(0.35)
    last = steps[-1]
    msg.success(f"Mastered in {last['step']} steps, total reward {last['cumulative_reward']:.0f}." if last["mastered"]
                else f"Not mastered within {C.RL_MAX_STEPS} steps (knowledge {last['next_knowledge']}/4), total reward "
                     f"{last['cumulative_reward']:.0f}.")

with st.expander("Environment parameters for this cluster (simulation assumptions)"):
    p = e["params"]
    cfg = rl["config"]
    st.markdown(
        f"- Learning multiplier **×{p['gain_multiplier']:.2f}** = 1 + {cfg['gain_coef']['studytime']}·z(study time) "
        f"{cfg['gain_coef']['failures']:+}·z(failures)\n"
        f"- Energy drain after a quiz **{p['p_drain'][0]:.0%}**, after a video **{p['p_drain'][1]:.0%}**; review notes "
        f"restore energy with **{p['p_restore']:.0%}** (shift {p['drain_shift']:+.3f} from going out, free time, health)\n"
        f"- Session skipped (absences) **{p['p_skip']:.0%}**\n"
        f"- Rewards: {cfg['rewards']['per_level']:+.0f} per knowledge level, {cfg['rewards']['per_step']:+.0f} per step, "
        f"{cfg['rewards']['energy_hits_zero']:+.0f} when energy drops to 0, {cfg['rewards']['mastery_bonus']:+.0f} at mastery")
    st.caption("These numbers come from a documented formula on the cluster's average habits (DECISIONS.md D12), "
               "not from observed students.")
