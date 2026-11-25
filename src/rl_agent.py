"""Tabular Q-learning study planner (spec §8.2, decaying learning rate since Phase 6b) + evaluation helpers.

One Q-table (15 states x 3 actions) per (subject, cluster) environment. The environments are
SIMULATIONS whose parameters come from the K-Means cluster centroids (src/rl_env.py); the
learned policies illustrate the method and are not validated pedagogy.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import joblib
import numpy as np

from src import config as C
from src.config import MODELS_DIR, SUBJECTS
from src.rl_env import N_ACTIONS, N_STATES, EnvParams, StudyEnv, params_from_centroid, state_index, state_tuple

Q_TABLES_PATH = MODELS_DIR / "q_tables.joblib"
CHECKPOINT_EVERY = 250
CHECKPOINT_EPISODES = 300
CHECKPOINT_SEED_OFFSET = 200_000
TERMINAL_STATES = [state_index(C.RL_N_KNOWLEDGE - 1, e) for e in range(C.RL_N_ENERGY)]
DECISION_STATES = [s for s in range(N_STATES) if s not in TERMINAL_STATES]

Policy = Callable[[int, np.random.Generator], int]


def epsilon_schedule(episode: int, n_episodes: int = C.RL_EPISODES) -> float:
    """Exponential decay from EPS_START to EPS_END over the first EPS_DECAY_FRACTION of episodes."""
    horizon = max(1, int(C.RL_EPS_DECAY_FRACTION * n_episodes))
    rate = (C.RL_EPS_END / C.RL_EPS_START) ** (1 / horizon)
    return max(C.RL_EPS_END, C.RL_EPS_START * rate ** episode)


class QLearningAgent:
    """Tabular Q-learning.

    alpha_power=x (default RL_ALPHA_POWER = 0.6) -> per-(state, action) learning rate
    max(RL_ALPHA_MIN, 1 / visits**x); used for the saved Q-tables (DECISIONS.md D16).
    alpha_power=None -> constant alpha = 0.1 (the spec's setting), kept as a comparison.
    """

    def __init__(self, n_states: int = N_STATES, n_actions: int = N_ACTIONS, alpha: float = C.RL_ALPHA,
                 gamma: float = C.RL_GAMMA, seed: int = C.RANDOM_STATE,
                 alpha_power: float | None = C.RL_ALPHA_POWER):
        self.q = np.zeros((n_states, n_actions))
        self.visits = np.zeros((n_states, n_actions), dtype=int)
        self.alpha, self.gamma = alpha, gamma
        self.alpha_power = alpha_power
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, terminal: bool) -> None:
        # Mastery is a true terminal state. Hitting the 20-step limit is a time-out (truncation),
        # so the target still bootstraps from the next state.
        self.visits[s, a] += 1
        alpha = self.alpha if self.alpha_power is None else max(C.RL_ALPHA_MIN, 1.0 / self.visits[s, a] ** self.alpha_power)
        target = r if terminal else r + self.gamma * self.q[s_next].max()
        self.q[s, a] += alpha * (target - self.q[s, a])

    def greedy_policy(self) -> np.ndarray:
        return self.q.argmax(axis=1)


def train(params: EnvParams, n_episodes: int = C.RL_EPISODES, seed: int = C.RANDOM_STATE,
          checkpoints: bool = True,
          alpha_power: float | None = C.RL_ALPHA_POWER) -> tuple[QLearningAgent, dict[str, Any]]:
    """Q-learning on one environment. Deterministic for a given seed."""
    env = StudyEnv(params)
    agent = QLearningAgent(seed=seed, alpha_power=alpha_power)
    rewards = np.zeros(n_episodes)
    history: dict[str, Any] = {"checkpoints": []}
    prev_policy = None
    for ep in range(n_episodes):
        eps = epsilon_schedule(ep, n_episodes)
        s, _ = env.reset(seed=seed if ep == 0 else None)
        done, total = False, 0.0
        while not done:
            a = agent.act(s, eps)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, terminal=info["mastered"])
            s, total = s2, total + r
        rewards[ep] = total
        if checkpoints and (ep + 1) % CHECKPOINT_EVERY == 0:
            pol = agent.greedy_policy()
            ev = evaluate_policy(params, greedy(agent.q), CHECKPOINT_EPISODES, CHECKPOINT_SEED_OFFSET)
            changed = None if prev_policy is None else int((pol[DECISION_STATES] != prev_policy[DECISION_STATES]).sum())
            history["checkpoints"].append({"episode": ep + 1, "epsilon": round(eps, 4),
                                           "greedy_mean_reward": round(ev["mean_reward"], 3),
                                           "decision_states_changed": changed})
            prev_policy = pol
    history["episode_rewards"] = rewards
    history["learning_curve_100"] = [round(float(rewards[i:i + 100].mean()), 3) for i in range(0, n_episodes, 100)]
    return agent, history


# --------------------------------------------------------------------------- policies & evaluation
def greedy(q: np.ndarray) -> Policy:
    """Policy that always takes the action with the highest Q-value."""
    return lambda s, rng: int(np.argmax(q[s]))


def random_policy() -> Policy:
    """Policy that picks one of the 3 actions uniformly at random."""
    return lambda s, rng: int(rng.integers(N_ACTIONS))


def fixed_policy(action: int) -> Policy:
    """Policy that always takes the same action."""
    return lambda s, rng: action


def evaluate_policy(params: EnvParams, policy: Policy, n_episodes: int = C.RL_EVAL_EPISODES,
                    seed_offset: int = C.RL_EVAL_SEED_OFFSET) -> dict[str, Any]:
    """Run n episodes with seeds seed_offset..seed_offset+n-1 (same seeds for every policy).

    The environment draws the same random numbers each step whatever the action, so all policies
    face the same skips / gains / energy events (common random numbers).
    """
    env = StudyEnv(params)
    rewards, mastered, steps = np.zeros(n_episodes), np.zeros(n_episodes, bool), np.zeros(n_episodes)
    for i in range(n_episodes):
        seed = seed_offset + i
        policy_rng = np.random.default_rng(seed + 7_919)  # separate stream for random policies
        s, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            s, r, done, info = env.step(policy(s, policy_rng))
            total += r
        rewards[i], mastered[i], steps[i] = total, info["mastered"], info["t"]
    return {
        "n_episodes": n_episodes,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=1)),
        "mastery_rate": float(mastered.mean()),
        "mean_steps_to_mastery": float(steps[mastered].mean()) if mastered.any() else None,
        "rewards": rewards,
    }


# --------------------------------------------------------------------------- cluster environments
def cluster_centroids(subject: str) -> dict[int, dict[str, float]]:
    """Standardized centroids (log1p(absences) -> StandardScaler space) of the saved K-Means model."""
    from src.unsupervised import load_kmeans

    pipe = load_kmeans(subject)
    names = list(pipe[:-1].get_feature_names_out())
    return {c: {f: float(v) for f, v in zip(names, center)} for c, center in enumerate(pipe[-1].cluster_centers_)}


def env_params(subject: str, cluster: int) -> EnvParams:
    """Environment parameters of one (subject, cluster), from the saved K-Means centroid."""
    centroids = cluster_centroids(subject)
    if cluster not in centroids:
        raise ValueError(f"cluster must be one of {sorted(centroids)} for subject {subject!r}, got {cluster!r}")
    return params_from_centroid(centroids[cluster])


@lru_cache(maxsize=None)
def load_q_tables() -> dict[tuple[str, int], np.ndarray]:
    """Load the saved Q-tables, keyed by (subject, cluster) (cached per process)."""
    if not Q_TABLES_PATH.is_file():
        raise FileNotFoundError(f"{Q_TABLES_PATH} not found. Run `python scripts/train_rl.py` first.")
    return joblib.load(Q_TABLES_PATH)


def _q_for(subject: str, cluster: int) -> np.ndarray:
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject {subject!r}; expected one of {sorted(SUBJECTS)}")
    tables = load_q_tables()
    if (subject, cluster) not in tables:
        valid = sorted(c for s, c in tables if s == subject)
        raise ValueError(f"cluster must be one of {valid} for subject {subject!r}, got {cluster!r}")
    return tables[(subject, cluster)]


def recommend(subject: str, cluster: int, knowledge: int, energy: int) -> dict[str, Any]:
    """Best next study action for a state, from the learned Q-table of that cluster's environment.

    At knowledge 4 (mastery) the episode is over: action is None.
    """
    q = _q_for(subject, cluster)
    for name, v, hi in (("knowledge", knowledge, C.RL_N_KNOWLEDGE - 1), ("energy", energy, C.RL_N_ENERGY - 1)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= hi:
            raise ValueError(f"{name} must be an integer in [0, {hi}], got {v!r}")
    s = state_index(int(knowledge), int(energy))
    qv = [round(float(x), 4) for x in q[s]]
    if knowledge == C.RL_N_KNOWLEDGE - 1:
        return {"action": None, "action_name": "Mastered (no further study needed)", "q_values": qv}
    a = int(np.argmax(q[s]))
    return {"action": a, "action_name": C.RL_ACTIONS[a], "q_values": qv}


def simulate_episode(subject: str, cluster: int, seed: int = 0) -> list[dict[str, Any]]:
    """One greedy episode in the cluster's (simulated) environment, step by step, for the app replay."""
    q = _q_for(subject, cluster)
    env = StudyEnv(env_params(subject, cluster))
    s, _ = env.reset(seed=seed)
    steps, total, done = [], 0.0, False
    while not done:
        k, e = state_tuple(s)
        a = int(np.argmax(q[s]))
        s, r, done, info = env.step(a)
        total += r
        steps.append({
            "step": info["t"], "knowledge": k, "energy": e, "action": a, "action_name": C.RL_ACTIONS[a],
            "skipped": info["skipped"], "gained": info["gained"], "hit_zero_energy": info["hit_zero_energy"],
            "reward": r, "cumulative_reward": total, "next_knowledge": info["knowledge"],
            "next_energy": info["energy"], "mastered": info["mastered"], "done": done,
        })
    return steps


# --------------------------------------------------------------------------- exact reference (analysis only)
def optimal_q(params: EnvParams, gamma: float = C.RL_GAMMA, tol: float = 1e-10) -> np.ndarray:
    """Q* of the simulated MDP by value iteration, using the KNOWN transition probabilities.

    Analysis tool only: the Q-learning agent never sees the model. Like the agent, it treats
    mastery as terminal and ignores the 20-step limit (the state has no clock), so Q* is exactly
    what Q-learning is trying to estimate.
    """
    from src.rl_env import REVIEW

    K, E = C.RL_N_KNOWLEDGE, C.RL_N_ENERGY
    q = np.zeros((N_STATES, N_ACTIONS))
    while True:
        v = q.max(axis=1)
        v[TERMINAL_STATES] = 0.0
        q_new = np.zeros_like(q)
        for s in DECISION_STATES:
            k, e = state_tuple(s)
            for a in range(N_ACTIONS):
                # skipped session: nothing changes
                total = params.p_skip * (C.RL_REWARD_STEP + gamma * v[s])
                g = params.p_gain[k, a, e]
                if a == REVIEW:
                    e_moves = [(min(E - 1, e + 1), params.p_restore), (e, 1 - params.p_restore)]
                else:
                    e_moves = [(max(0, e - 1), params.p_drain[a]), (e, 1 - params.p_drain[a])]
                for gained, pg in ((1, g), (0, 1 - g)):
                    k2 = k + gained
                    for e2, pe in e_moves:
                        r = C.RL_REWARD_STEP + C.RL_REWARD_LEVEL * gained
                        r += C.RL_REWARD_ENERGY_ZERO * (e2 == 0 and e > 0)
                        terminal = k2 == K - 1
                        r += C.RL_REWARD_MASTERY * terminal
                        nxt = 0.0 if terminal else v[state_index(k2, e2)]
                        total += (1 - params.p_skip) * pg * pe * (r + gamma * nxt)
                q_new[s, a] = total
        if np.abs(q_new - q).max() < tol:
            return q_new
        q = q_new
