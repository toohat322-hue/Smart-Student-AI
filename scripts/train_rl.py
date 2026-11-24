"""Phase 6 — Reinforcement-learning study planner runner (spec §8).

One SIMULATED environment per K-Means cluster (5 Mathematics + 3 Portuguese). For each:
Q-learning (alpha = 1/visits^0.6, gamma 0.9, eps 1.0 -> 0.05, 5,000 episodes, seed 42), plus the
spec's constant alpha = 0.1 as a stored comparison; evaluation of the
learned policy vs random / always-quizzes / best single fixed action on 1,000 fixed test seeds;
exact optimal policy by value iteration (analysis reference); convergence diagnostics.
Saves models/q_tables.joblib, reports/metrics.json["rl"] and figures 17–20.

Usage:
    python scripts/train_rl.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import config as C  # noqa: E402
from src.metrics_io import load_metrics, update_metrics  # noqa: E402
from src.rl_agent import (  # noqa: E402
    CHECKPOINT_EVERY, DECISION_STATES, Q_TABLES_PATH, cluster_centroids, env_params, epsilon_schedule,
    evaluate_policy, fixed_policy, greedy, optimal_q, random_policy, train,
)
from src.rl_env import state_tuple  # noqa: E402

SUBJECT_ORDER = ["mat", "por"]
SUBJECT_NAMES = {"mat": "Mathematics", "por": "Portuguese"}
ACTION_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]  # quizzes, videos, review notes (fixed order)
C_GRAY, C_TEXT2, C_ORANGE = "#8a8984", "#52514e", "#eb6834"
NEAR_TIE = 0.5          # Q* gap below this = the two best actions are almost equally good

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name: str) -> str:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / name)
    plt.close(fig)
    return f"reports/figures/{name}"


def summary(ev: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items() if k != "rewards"}


def paired(a: dict, b: dict) -> dict:
    """Mean reward difference a - b over the same test seeds, with its standard error."""
    d = a["rewards"] - b["rewards"]
    return {"mean": round(float(d.mean()), 3), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 3),
            "share_episodes_better": round(float((d > 0).mean()), 4)}


def policy_grid(policy: np.ndarray) -> list[list[int | None]]:
    """Rows knowledge 0..4, columns energy 0..2; None for terminal (knowledge 4)."""
    grid = policy.reshape(C.RL_N_KNOWLEDGE, C.RL_N_ENERGY).tolist()
    grid[-1] = [None] * C.RL_N_ENERGY
    return grid


def run() -> tuple[dict, dict]:
    clustering = load_metrics()["clustering"]["subjects"]
    q_tables: dict[tuple[str, int], np.ndarray] = {}
    res: dict = {
        "honesty_note": ("SIMULATION. The UCI data has no sequences of study actions and rewards. Each environment's "
                         "probabilities come from a documented linear formula on a K-Means cluster centroid; they are "
                         "assumptions, not learned from data. Policies illustrate the method, not validated pedagogy."),
        "config": {
            "actions": C.RL_ACTIONS, "start_state": list(C.RL_START_STATE), "max_steps": C.RL_MAX_STEPS,
            "rewards": {"per_level": C.RL_REWARD_LEVEL, "per_step": C.RL_REWARD_STEP,
                        "energy_hits_zero": C.RL_REWARD_ENERGY_ZERO, "mastery_bonus": C.RL_REWARD_MASTERY},
            "base_gain": {C.RL_ACTIONS[a]: v for a, v in C.RL_BASE_GAIN.items()},
            "video_low_knowledge_bonus": C.RL_VIDEO_LOW_KNOWLEDGE_BONUS,
            "video_high_knowledge_penalty": C.RL_VIDEO_HIGH_KNOWLEDGE_PENALTY,
            "base_drain": {C.RL_ACTIONS[a]: v for a, v in C.RL_BASE_DRAIN.items()},
            "base_restore": C.RL_BASE_RESTORE, "base_skip": C.RL_BASE_SKIP, "prob_clip": list(C.RL_PROB_CLIP),
            "gain_coef": C.RL_GAIN_COEF, "gain_mult_clip": list(C.RL_GAIN_MULT_CLIP),
            "drain_coef": C.RL_DRAIN_COEF, "skip_coef": C.RL_SKIP_COEF,
            "alpha_schedule": f"max({C.RL_ALPHA_MIN}, 1 / visits**{C.RL_ALPHA_POWER}) per (state, action)",
            "alpha_power": C.RL_ALPHA_POWER, "comparison_constant_alpha": C.RL_ALPHA, "gamma": C.RL_GAMMA, "eps_start": C.RL_EPS_START, "eps_end": C.RL_EPS_END,
            "eps_decay_fraction": C.RL_EPS_DECAY_FRACTION, "episodes": C.RL_EPISODES, "seed": C.RANDOM_STATE,
            "eval_episodes": C.RL_EVAL_EPISODES, "eval_seeds": [C.RL_EVAL_SEED_OFFSET,
                                                                  C.RL_EVAL_SEED_OFFSET + C.RL_EVAL_EPISODES - 1],
            "checkpoint_every": CHECKPOINT_EVERY, "near_tie_q_gap": NEAR_TIE,
        },
        "environments": {},
    }
    for s in SUBJECT_ORDER:
        res["environments"][s] = {}
        centroids = cluster_centroids(s)
        names = {c["cluster"]: c["name"] for c in clustering[s]["profile"]["clusters"]}
        for c, z in centroids.items():
            t0 = time.time()
            p = env_params(s, c)
            agent, hist = train(p)
            q_tables[(s, c)] = agent.q.copy()

            evals = {"learned": evaluate_policy(p, greedy(agent.q)), "random": evaluate_policy(p, random_policy())}
            fixed = {a: evaluate_policy(p, fixed_policy(a)) for a in range(len(C.RL_ACTIONS))}
            best_a = max(fixed, key=lambda a: fixed[a]["mean_reward"])
            evals["always_quizzes"] = fixed[0]
            evals["best_fixed"] = fixed[best_a]

            q_star = optimal_q(p)
            pi_star, pi = q_star.argmax(1), agent.greedy_policy()
            evals["optimal_reference"] = evaluate_policy(p, greedy(q_star))
            sorted_q = np.sort(q_star, axis=1)
            mismatches = []
            for st in DECISION_STATES:
                if pi[st] != pi_star[st]:
                    k, e = state_tuple(st)
                    mismatches.append({
                        "knowledge": k, "energy": e, "learned": C.RL_ACTIONS[pi[st]], "optimal": C.RL_ACTIONS[pi_star[st]],
                        "q_star_loss": round(float(q_star[st, pi_star[st]] - q_star[st, pi[st]]), 3),
                        "visits": int(agent.visits[st].sum()),
                    })

            # Comparison: the spec's constant alpha = 0.1, same seed / episodes / evaluation (not saved).
            ag_const, hist_const = train(p, alpha_power=None)
            ev_const = evaluate_policy(p, greedy(ag_const.q))
            cps = hist["checkpoints"]
            last5 = cps[-5:]
            res["environments"][s][str(c)] = {
                "cluster_name": names[c],
                "centroid_z": {f: round(v, 4) for f, v in z.items()},
                "params": p.to_dict(),
                "evaluation": {k: summary(v) for k, v in evals.items()},
                "best_fixed_action": C.RL_ACTIONS[best_a],
                "fixed_actions": {C.RL_ACTIONS[a]: summary(v) for a, v in fixed.items()},
                "learned_minus_best_fixed": paired(evals["learned"], evals["best_fixed"]),
                "learned_minus_random": paired(evals["learned"], evals["random"]),
                "optimal_minus_learned": paired(evals["optimal_reference"], evals["learned"]),
                "policy": policy_grid(pi),
                "optimal_policy": policy_grid(pi_star),
                "q_table": np.round(agent.q, 4).tolist(),
                "q_star": np.round(q_star, 4).tolist(),
                "visits": agent.visits.tolist(),
                "decision_states_matching_optimal": int((pi[DECISION_STATES] == pi_star[DECISION_STATES]).sum()),
                "n_decision_states": len(DECISION_STATES),
                "mismatches_vs_optimal": mismatches,
                "near_tie_states_q_star": int(((sorted_q[:, -1] - sorted_q[:, -2])[DECISION_STATES] < NEAR_TIE).sum()),
                "learning_curve_100": hist["learning_curve_100"],
                "checkpoints": cps,
                "convergence": {
                    "last5_checkpoint_greedy_reward_range": round(max(x["greedy_mean_reward"] for x in last5)
                                                                  - min(x["greedy_mean_reward"] for x in last5), 3),
                    "last5_checkpoint_states_changed": [x["decision_states_changed"] for x in last5],
                    "behaviour_reward_last_1000_first_half_vs_second_half": [
                        round(float(np.mean(hist["episode_rewards"][-1000:-500])), 3),
                        round(float(np.mean(hist["episode_rewards"][-500:])), 3)],
                },
                "comparison_constant_alpha": {
                    "alpha": C.RL_ALPHA,
                    "decision_states_matching_optimal":
                        int((ag_const.greedy_policy()[DECISION_STATES] == pi_star[DECISION_STATES]).sum()),
                    "evaluation": summary(ev_const),
                    "optimal_minus_learned": paired(evals["optimal_reference"], ev_const),
                    "policy": policy_grid(ag_const.greedy_policy()),
                    "learning_curve_100": hist_const["learning_curve_100"],
                    "checkpoints": hist_const["checkpoints"],
                    "last5_checkpoint_greedy_reward_range": round(
                        max(x["greedy_mean_reward"] for x in hist_const["checkpoints"][-5:])
                        - min(x["greedy_mean_reward"] for x in hist_const["checkpoints"][-5:]), 3),
                    "last5_checkpoint_states_changed": [x["decision_states_changed"]
                                                        for x in hist_const["checkpoints"][-5:]],
                },
            }
            print(f"  {s}/{c}: {time.time() - t0:.0f}s", flush=True)

    # ---- do policies differ between clusters?
    keys = [(s, c) for s in SUBJECT_ORDER for c in res["environments"][s]]
    diffs = []
    for st in DECISION_STATES:
        k, e = state_tuple(st)
        learned = {f"{s}/{c}": res["environments"][s][c]["policy"][k][e] for s, c in keys}
        optimal = {f"{s}/{c}": res["environments"][s][c]["optimal_policy"][k][e] for s, c in keys}
        if len(set(learned.values())) > 1 or len(set(optimal.values())) > 1:
            diffs.append({"knowledge": k, "energy": e,
                          "learned": {kk: C.RL_ACTIONS[v] for kk, v in learned.items()},
                          "optimal_actions": sorted({C.RL_ACTIONS[v] for v in optimal.values()})})
    optimal_grids = {str(res["environments"][s][c]["optimal_policy"]) for s, c in keys}
    learned_grids = {str(res["environments"][s][c]["policy"]) for s, c in keys}
    res["policy_differences"] = {
        "n_distinct_learned_policies": len(learned_grids),
        "n_distinct_optimal_policies": len(optimal_grids),
        "states_where_policies_differ": diffs,
    }
    return res, q_tables


# --------------------------------------------------------------------------- figures
def env_keys(res):
    return [(s, c) for s in SUBJECT_ORDER for c in res["environments"][s]]


def fig_learning(res: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for i, s in enumerate(SUBJECT_ORDER):
        for c, e in res["environments"][s].items():
            col = colors[int(c)]
            lc = e["learning_curve_100"]
            axes[i, 0].plot(np.arange(len(lc)) * 100 + 100, lc, color=col, linewidth=1.6,
                            label=f"{c}: {e['cluster_name']}")
            cps = e["checkpoints"]
            axes[i, 1].plot([x["episode"] for x in cps], [x["greedy_mean_reward"] for x in cps], color=col,
                            linewidth=1.6, marker="o", markersize=2.5, label=f"{c}: {e['cluster_name']}")
            axes[i, 1].axhline(e["evaluation"]["optimal_reference"]["mean_reward"], color=col, linestyle=":",
                               linewidth=1)
        eps = [epsilon_schedule(ep) for ep in range(C.RL_EPISODES)]
        ax2 = axes[i, 0].twinx()
        ax2.plot(range(C.RL_EPISODES), eps, color=C_GRAY, linestyle="--", linewidth=1)
        ax2.set_ylabel("epsilon (dashed)", color=C_GRAY)
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)
        axes[i, 0].set_title(f"{SUBJECT_NAMES[s]}: training reward (ε-greedy, mean per 100 episodes)")
        axes[i, 1].set_title(f"{SUBJECT_NAMES[s]}: greedy policy every {CHECKPOINT_EVERY} episodes "
                             "(dotted = exact optimum)")
        for ax in axes[i]:
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean episode reward")
        axes[i, 1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Q-learning convergence per cluster environment (simulation)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "17_rl_learning_curves.png")


def draw_policy(ax, grid, title, ref=None):
    arr = np.array([[np.nan if v is None else v for v in row] for row in grid], dtype=float)
    ax.imshow(arr, cmap=ListedColormap(ACTION_COLORS), vmin=-0.5, vmax=2.5, origin="lower", aspect="auto")
    short = ["Quiz", "Video", "Review"]
    for k in range(arr.shape[0]):
        for e in range(arr.shape[1]):
            if np.isnan(arr[k, e]):
                ax.text(e, k, "mastered", ha="center", va="center", fontsize=7.5, color=C_TEXT2)
                continue
            differs = ref is not None and ref[k][e] != grid[k][e]
            ax.text(e, k, short[int(arr[k, e])] + (" *" if differs else ""), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold" if differs else "normal")
    ax.set_xticks(range(3), labels=["low", "mid", "high"], fontsize=8)
    ax.set_yticks(range(5), labels=range(5), fontsize=8)
    ax.set_xlabel("energy", fontsize=8)
    ax.set_ylabel("knowledge", fontsize=8)
    ax.grid(False)
    ax.set_title(title, fontsize=8.5)


def fig_policies(res: dict) -> str:
    keys = env_keys(res)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
    flat = axes.ravel()
    ref = res["environments"][keys[0][0]][keys[0][1]]["optimal_policy"]
    for ax, (s, c) in zip(flat, keys):
        e = res["environments"][s][c]
        draw_policy(ax, e["policy"], f"{SUBJECT_NAMES[s]} {c}: {e['cluster_name']}\n"
                    f"{e['decision_states_matching_optimal']}/12 states = optimum", ref=e["optimal_policy"])
    same = res["policy_differences"]["n_distinct_optimal_policies"] == 1
    draw_policy(flat[len(keys)], ref, "Exact optimal policy (value iteration)\n"
                + ("identical for all 8 environments" if same else "cluster 0 of Mathematics"))
    for ax in flat[len(keys) + 1:]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in ACTION_COLORS]
    fig.legend(handles, C.RL_ACTIONS, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Learned policy per cluster: best action per (knowledge, energy).  * = differs from the exact optimum",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    return save(fig, "18_rl_policy_heatmaps.png")


def fig_comparison(res: dict) -> str:
    keys = env_keys(res)
    series = [("learned", "Learned (Q-learning)", "#2a78d6"), ("best_fixed", "Best single fixed action", "#1baf7a"),
              ("random", "Random", C_GRAY), ("always_quizzes", "Always quizzes", "#eda100")]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(keys))
    w = 0.2
    for i, (k, name, col) in enumerate(series):
        m = [res["environments"][s][c]["evaluation"][k]["mean_reward"] for s, c in keys]
        sd = [res["environments"][s][c]["evaluation"][k]["std_reward"] for s, c in keys]
        ax.bar(x + (i - 1.5) * (w + 0.01), m, width=w, color=col, label=name, yerr=sd,
               error_kw={"ecolor": C_TEXT2, "elinewidth": 0.8, "capsize": 2})
    opt = [res["environments"][s][c]["evaluation"]["optimal_reference"]["mean_reward"] for s, c in keys]
    ax.scatter(x - 1.5 * (w + 0.01), opt, marker="_", s=300, color="#0b0b0b", zorder=5,
               label="Exact optimum (reference)")
    ax.axhline(0, color=C_TEXT2, linewidth=0.8)
    ax.set_xticks(x, labels=[f"{SUBJECT_NAMES[s][:4]}. {c}\n{res['environments'][s][c]['cluster_name']}".replace(" · ", "\n")
                             for s, c in keys], fontsize=7.5)
    ax.set_ylabel("Mean episode reward (± std), 1,000 test episodes")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8.5)
    ax.set_title("Policy comparison per cluster environment (simulation, same 1,000 test seeds)", pad=34)
    return save(fig, "19_rl_policy_comparison.png")


def fig_alpha_comparison(res: dict) -> str:
    """Decaying alpha (saved) vs the spec's constant alpha = 0.1."""
    keys = env_keys(res)
    labels = [f"{s} {c}" for s, c in keys]
    x = np.arange(len(keys))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    series = (("decaying α = 1/visits^0.6 (saved)", "#2a78d6", lambda e: e),
              ("constant α = 0.1 (spec)", C_GRAY, lambda e: e["comparison_constant_alpha"]))
    for i, (name, col, get) in enumerate(series):
        match = [get(res["environments"][s][c])["decision_states_matching_optimal"] for s, c in keys]
        gap = [get(res["environments"][s][c])["optimal_minus_learned"]["mean"] for s, c in keys]
        xs = x + (i - 0.5) * (w + 0.02)
        axes[0].bar(xs, match, width=w, color=col, label=name)
        axes[1].bar(xs, gap, width=w, color=col, label=name)
        # mean gap of the greedy policy to the optimum, across the 8 environments, per checkpoint
        cps = [get(res["environments"][s][c])["checkpoints"] for s, c in keys]
        opt = [res["environments"][s][c]["evaluation"]["optimal_reference"]["mean_reward"] for s, c in keys]
        eps_axis = [cp["episode"] for cp in cps[0]]
        mean_gap = [np.mean([o - cp[j]["greedy_mean_reward"] for o, cp in zip(opt, cps)]) for j in range(len(eps_axis))]
        axes[2].plot(eps_axis, mean_gap, color=col, marker="o", markersize=3, linewidth=1.8, label=name)
    axes[0].set_ylim(0, 13)
    axes[0].axhline(12, color=C_TEXT2, linestyle=":", linewidth=1)
    axes[0].set_ylabel("Decision states with the optimal action (of 12)")
    axes[1].set_ylabel("Reward gap to the exact optimum (1,000 test episodes)")
    axes[2].set_ylabel("Mean reward gap to optimum (greedy policy)")
    axes[2].set_xlabel("Training episode")
    axes[2].axhline(0, color=C_TEXT2, linewidth=0.8)
    for ax in axes[:2]:
        ax.set_xticks(x, labels=labels, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("Agreement with the exact optimal policy")
    axes[1].set_title("How much reward the learned policy leaves")
    axes[2].set_title("Convergence, averaged over the 8 environments")
    handles, lbls = axes[2].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Learning-rate comparison: decaying vs constant α (same seed, 5,000 episodes)", fontweight="bold",
                 y=1.07)
    fig.tight_layout()
    return save(fig, "20_rl_alpha_comparison.png")


# --------------------------------------------------------------------------- summary
def print_summary(res: dict) -> None:
    print("\nENVIRONMENT PARAMETERS")
    for s, c in env_keys(res):
        e = res["environments"][s][c]
        p = e["params"]
        print(f"  {s}/{c} {e['cluster_name']:38s} gain x{p['gain_multiplier']:.2f}  drain shift {p['drain_shift']:+.3f}  "
              f"P(drain) quiz {p['p_drain'][0]:.2f} video {p['p_drain'][1]:.2f}  P(restore) {p['p_restore']:.2f}  "
              f"P(skip) {p['p_skip']:.2f}")
    print("\nEVALUATION (1,000 test episodes): mean ± std reward | mastery | steps to mastery")
    for s, c in env_keys(res):
        e = res["environments"][s][c]
        print(f"  {s}/{c} {e['cluster_name']}  (best fixed = {e['best_fixed_action']})")
        for k in ("learned", "best_fixed", "random", "always_quizzes", "optimal_reference"):
            v = e["evaluation"][k]
            st = f"{v['mean_steps_to_mastery']:.1f}" if v["mean_steps_to_mastery"] else "-"
            print(f"     {k:18s} {v['mean_reward']:6.1f} ± {v['std_reward']:5.1f} | {v['mastery_rate']:6.1%} | {st}")
        d1, d2, d3 = e["learned_minus_best_fixed"], e["learned_minus_random"], e["optimal_minus_learned"]
        cv = e["convergence"]
        print(f"     learned - best fixed {d1['mean']:+.2f} (SE {d1['se']:.2f});  learned - random {d2['mean']:+.2f} "
              f"(SE {d2['se']:.2f});  optimum - learned {d3['mean']:+.2f} (SE {d3['se']:.2f})")
        print(f"     matches optimum in {e['decision_states_matching_optimal']}/12 states; mismatches: "
              + ("; ".join(f"k{m['knowledge']}e{m['energy']} {m['learned']} vs {m['optimal']} (loss {m['q_star_loss']}, "
                           f"visits {m['visits']})" for m in e["mismatches_vs_optimal"]) or "none"))
        print(f"     convergence: last 5 checkpoints greedy range {cv['last5_checkpoint_greedy_reward_range']}, states "
              f"changed {cv['last5_checkpoint_states_changed']}")
        ca = e["comparison_constant_alpha"]
        print(f"     constant alpha 0.1: {ca['evaluation']['mean_reward']:.1f} reward, matches optimum "
              f"{ca['decision_states_matching_optimal']}/12, gap {ca['optimal_minus_learned']['mean']:+.2f}, "
              f"last-5 range {ca['last5_checkpoint_greedy_reward_range']}")
    pdiff = res["policy_differences"]
    print(f"\nDistinct learned policies: {pdiff['n_distinct_learned_policies']} of 8;"
          f" distinct OPTIMAL policies: {pdiff['n_distinct_optimal_policies']}")
    for d in pdiff["states_where_policies_differ"]:
        print(f"  k{d['knowledge']} e{d['energy']}: optimal {d['optimal_actions']}; learned "
              + ", ".join(f"{k}={v.split()[0]}" for k, v in d["learned"].items()))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    res, q_tables = run()
    joblib.dump(q_tables, Q_TABLES_PATH)
    res["figures"] = [fig_learning(res), fig_policies(res), fig_comparison(res), fig_alpha_comparison(res)]
    update_metrics("rl", res)
    print_summary(res)
    print(f"\nSaved {Q_TABLES_PATH.name} ({len(q_tables)} Q-tables), metrics.json['rl'], "
          f"{', '.join(res['figures'])}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
