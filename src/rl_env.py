"""StudyEnv — a SIMULATED study-session environment (spec §8.1), Gymnasium-style API.

Honesty note: the UCI data contains no sequences of study actions and rewards. Every
probability below is a simulation assumption (see src/config.py and DECISIONS.md), shaped per
K-Means cluster by a simple linear formula on the cluster's standardized centroid. The agent
learns from this synthetic environment, not from real student trajectories.

State   (knowledge 0..4, energy 0..2) -> index knowledge * 3 + energy (15 states)
Actions 0 Practice Quizzes, 1 Watch Videos, 2 Review Notes
Step    reward -1; the session may be skipped (absences); otherwise knowledge +1 with
        p_gain[knowledge, action, energy] (+10), then energy changes (quizzes/videos may drain,
        review notes may restore); -5 when energy drops to 0; +20 and terminal at knowledge 4;
        at most 20 steps per episode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src import config as C

N_STATES = C.RL_N_KNOWLEDGE * C.RL_N_ENERGY
N_ACTIONS = len(C.RL_ACTIONS)
QUIZ, VIDEO, REVIEW = 0, 1, 2
CENTROID_FEATURES = ["studytime", "absences", "failures", "freetime", "health", "goout"]


def state_index(knowledge: int, energy: int) -> int:
    return knowledge * C.RL_N_ENERGY + energy


def state_tuple(index: int) -> tuple[int, int]:
    return divmod(index, C.RL_N_ENERGY)


def _clip(p):
    return np.clip(p, *C.RL_PROB_CLIP)


@dataclass(frozen=True)
class EnvParams:
    """All transition probabilities of one environment (already clipped to RL_PROB_CLIP)."""
    gain_multiplier: float
    drain_shift: float
    p_gain: np.ndarray        # shape (5 knowledge, 3 actions, 3 energy)
    p_drain: np.ndarray       # shape (3 actions,); review notes = 0 (it never drains)
    p_restore: float          # review notes: P(energy +1)
    p_skip: float             # P(the session does not count)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["p_gain"] = np.round(self.p_gain, 4).tolist()
        d["p_drain"] = np.round(self.p_drain, 4).tolist()
        for k in ("gain_multiplier", "drain_shift", "p_restore", "p_skip"):
            d[k] = round(float(d[k]), 4)
        return d


def params_from_centroid(z: dict[str, float]) -> EnvParams:
    """Linear, documented mapping from a cluster centroid (z units) to environment parameters.

    gain multiplier = clip(1 + 0.15*z_studytime - 0.15*z_failures, 0.5, 1.5)
    drain shift     = 0.08*z_goout + 0.04*z_freetime - 0.08*z_health
    p_drain[a]      = clip(base_drain[a] + drain shift)          (quizzes, videos)
    p_restore       = clip(base_restore - drain shift)           (review notes)
    p_skip          = clip(base_skip + 0.05*z_absences)
    p_gain[k,a,e]   = clip(base_gain[a][e] (+ video knowledge adjustment) * gain multiplier)
    Probabilities are clipped to RL_PROB_CLIP (0.05–0.95). Coefficients live in src/config.py.
    """
    missing = [f for f in CENTROID_FEATURES if f not in z]
    if missing:
        raise ValueError(f"centroid is missing {missing}")
    mult = float(np.clip(1 + sum(c * z[f] for f, c in C.RL_GAIN_COEF.items()), *C.RL_GAIN_MULT_CLIP))
    shift = float(sum(c * z[f] for f, c in C.RL_DRAIN_COEF.items()))

    p_gain = np.zeros((C.RL_N_KNOWLEDGE, N_ACTIONS, C.RL_N_ENERGY))
    for k in range(C.RL_N_KNOWLEDGE):
        for a in range(N_ACTIONS):
            base = np.array(C.RL_BASE_GAIN[a], dtype=float)
            if a == VIDEO and k <= 1:
                base = base + C.RL_VIDEO_LOW_KNOWLEDGE_BONUS
            if a == VIDEO and k >= 3:
                base = base - C.RL_VIDEO_HIGH_KNOWLEDGE_PENALTY
            p_gain[k, a] = _clip(base * mult)
    p_drain = np.array([_clip(C.RL_BASE_DRAIN[QUIZ] + shift), _clip(C.RL_BASE_DRAIN[VIDEO] + shift), 0.0])
    return EnvParams(
        gain_multiplier=mult,
        drain_shift=shift,
        p_gain=p_gain,
        p_drain=p_drain,
        p_restore=float(_clip(C.RL_BASE_RESTORE - shift)),
        p_skip=float(_clip(C.RL_BASE_SKIP + sum(c * z[f] for f, c in C.RL_SKIP_COEF.items()))),
    )


AVERAGE_PARAMS = params_from_centroid({f: 0.0 for f in CENTROID_FEATURES})


class StudyEnv:
    """Gymnasium-style: reset(seed) -> (state, info); step(action) -> (state, reward, done, info)."""

    n_states = N_STATES
    n_actions = N_ACTIONS

    def __init__(self, params: EnvParams = AVERAGE_PARAMS):
        self.params = params
        self.rng = np.random.default_rng()
        self.knowledge, self.energy = C.RL_START_STATE
        self.t = 0
        self.done = True

    @property
    def state(self) -> int:
        return state_index(self.knowledge, self.energy)

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        if seed is not None:  # like Gymnasium: a seed re-seeds; None continues the current stream
            self.rng = np.random.default_rng(seed)
        self.knowledge, self.energy = C.RL_START_STATE
        self.t = 0
        self.done = False
        return self.state, {"knowledge": self.knowledge, "energy": self.energy}

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is over; call reset() first.")
        if action not in range(N_ACTIONS):
            raise ValueError(f"action must be 0..{N_ACTIONS - 1}, got {action!r}")
        p = self.params
        k0, e0 = self.knowledge, self.energy
        reward = C.RL_REWARD_STEP
        # Draw all random numbers every step so episodes stay aligned across policies.
        u_skip, u_gain, u_energy = self.rng.random(3)

        skipped = bool(u_skip < p.p_skip)
        gained = False
        if not skipped:
            gained = bool(u_gain < p.p_gain[k0, action, e0])
            if gained:
                self.knowledge = k0 + 1
                reward += C.RL_REWARD_LEVEL
            if action == REVIEW:
                if u_energy < p.p_restore:
                    self.energy = min(C.RL_N_ENERGY - 1, e0 + 1)
            elif u_energy < p.p_drain[action]:
                self.energy = max(0, e0 - 1)

        hit_zero = self.energy == 0 and e0 > 0
        if hit_zero:
            reward += C.RL_REWARD_ENERGY_ZERO
        mastered = self.knowledge == C.RL_N_KNOWLEDGE - 1
        if mastered:
            reward += C.RL_REWARD_MASTERY
        self.t += 1
        self.done = mastered or self.t >= C.RL_MAX_STEPS
        info = {"knowledge": self.knowledge, "energy": self.energy, "skipped": skipped, "gained": gained,
                "hit_zero_energy": hit_zero, "mastered": mastered, "t": self.t}
        return self.state, float(reward), self.done, info
