"""Tests for src/rl_env.py and src/rl_agent.py (spec §10: test_rl.py)."""
import numpy as np
import pytest

from src import config as C
from src.metrics_io import load_metrics
from src.rl_agent import (
    DECISION_STATES,
    cluster_centroids,
    env_params,
    epsilon_schedule,
    evaluate_policy,
    fixed_policy,
    greedy,
    load_q_tables,
    optimal_q,
    random_policy,
    recommend,
    simulate_episode,
    train,
)
from src.rl_env import AVERAGE_PARAMS, CENTROID_FEATURES, N_STATES, StudyEnv, params_from_centroid, state_index

ENVS = [("mat", c) for c in range(5)] + [("por", c) for c in range(3)]


# --------------------------------------------------------------------------- environment
def test_reset_returns_valid_start_state():
    env = StudyEnv()
    s, info = env.reset(seed=0)
    assert s == state_index(*C.RL_START_STATE) and 0 <= s < N_STATES
    assert (info["knowledge"], info["energy"]) == C.RL_START_STATE


def test_step_contract_and_episode_length():
    env = StudyEnv()
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    n, done = 0, False
    while not done:
        out = env.step(int(rng.integers(3)))
        assert isinstance(out, tuple) and len(out) == 4
        s, r, done, info = out
        assert isinstance(s, int) and 0 <= s < N_STATES and isinstance(r, float) and isinstance(done, bool)
        assert isinstance(info, dict)
        n += 1
    assert n <= C.RL_MAX_STEPS
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize("seed", range(30))
def test_rewards_follow_spec(seed):
    env = StudyEnv(env_params("mat", seed % 5))
    env.reset(seed=seed)
    done, steps = False, 0
    while not done:
        _, r, done, info = env.step(seed % 3)
        expected = (C.RL_REWARD_STEP + C.RL_REWARD_LEVEL * info["gained"]
                    + C.RL_REWARD_ENERGY_ZERO * info["hit_zero_energy"] + C.RL_REWARD_MASTERY * info["mastered"])
        assert r == expected
        steps += 1
        assert done == (info["mastered"] or steps == C.RL_MAX_STEPS)
    assert steps <= C.RL_MAX_STEPS


def test_reset_seed_is_deterministic():
    def run(seed):
        env = StudyEnv()
        env.reset(seed=seed)
        return [env.step(i % 3)[:3] for i in range(10) if not env.done]
    assert run(5) == run(5)


def test_invalid_action():
    env = StudyEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(3)


@pytest.mark.parametrize("subject,cluster", ENVS)
def test_probabilities_valid_for_every_cluster(subject, cluster):
    p = env_params(subject, cluster)
    lo, hi = C.RL_PROB_CLIP
    assert p.p_gain.shape == (5, 3, 3)
    assert ((p.p_gain >= lo) & (p.p_gain <= hi)).all()
    assert all(0 <= x <= 1 for x in (*p.p_drain, p.p_restore, p.p_skip))
    assert lo <= p.p_drain[0] <= hi and lo <= p.p_drain[1] <= hi and p.p_drain[2] == 0
    assert lo <= p.p_restore <= hi and lo <= p.p_skip <= hi
    assert C.RL_GAIN_MULT_CLIP[0] <= p.gain_multiplier <= C.RL_GAIN_MULT_CLIP[1]


def test_formula_directions():
    base = {f: 0.0 for f in CENTROID_FEATURES}
    p0 = params_from_centroid(base)
    assert params_from_centroid({**base, "studytime": 1}).gain_multiplier > p0.gain_multiplier
    assert params_from_centroid({**base, "failures": 1}).gain_multiplier < p0.gain_multiplier
    assert params_from_centroid({**base, "goout": 1}).p_drain[0] > p0.p_drain[0]
    assert params_from_centroid({**base, "freetime": 1}).p_drain[0] > p0.p_drain[0]
    assert params_from_centroid({**base, "health": 1}).p_drain[0] < p0.p_drain[0]
    assert params_from_centroid({**base, "health": 1}).p_restore > p0.p_restore
    assert params_from_centroid({**base, "absences": 1}).p_skip > p0.p_skip
    assert params_from_centroid({**base, "failures": 50}).gain_multiplier == C.RL_GAIN_MULT_CLIP[0]
    with pytest.raises(ValueError):
        params_from_centroid({"studytime": 0})


def test_action_characteristics_from_spec():
    p = AVERAGE_PARAMS
    q, v, r = 0, 1, 2
    assert p.p_gain[2, q, 2] == p.p_gain[2].max()            # quizzes: highest gain at high energy
    assert p.p_drain[q] > p.p_drain[v]                       # quizzes cost more energy than videos
    assert p.p_gain[0, v, 0] > p.p_gain[3, v, 0]             # videos better when knowledge is low
    assert p.p_gain[2, r].max() < p.p_gain[2, q, 2] and p.p_restore > 0  # review: small gain, restores


# --------------------------------------------------------------------------- agent
def test_epsilon_schedule():
    eps = [epsilon_schedule(e) for e in range(C.RL_EPISODES)]
    assert eps[0] == C.RL_EPS_START and eps[-1] == pytest.approx(C.RL_EPS_END)
    assert all(a >= b for a, b in zip(eps, eps[1:]))


def test_q_tables_saved_for_every_cluster():
    tables = load_q_tables()
    assert sorted(tables) == sorted(ENVS)
    clustering = load_metrics()["clustering"]["subjects"]
    assert {c for s, c in tables if s == "mat"} == set(range(clustering["mat"]["k"]))
    assert {c for s, c in tables if s == "por"} == set(range(clustering["por"]["k"]))
    for q in tables.values():
        assert q.shape == (15, 3) and np.isfinite(q).all()


@pytest.mark.parametrize("subject,cluster", [("mat", 2), ("por", 1)])
def test_training_is_deterministic_and_matches_saved(subject, cluster):
    p = env_params(subject, cluster)
    a1, _ = train(p, checkpoints=False)
    a2, _ = train(p, checkpoints=False)
    assert np.array_equal(a1.q, a2.q)
    assert np.allclose(a1.q, load_q_tables()[(subject, cluster)])


@pytest.mark.parametrize("subject,cluster", ENVS)
def test_learned_policy_beats_random(subject, cluster):
    p = env_params(subject, cluster)
    q = load_q_tables()[(subject, cluster)]
    learned = evaluate_policy(p, greedy(q), n_episodes=500)
    rand = evaluate_policy(p, random_policy(), n_episodes=500)
    assert learned["mean_reward"] > rand["mean_reward"]


@pytest.mark.parametrize("subject,cluster", ENVS)
def test_exact_optimum_is_an_upper_reference(subject, cluster):
    p = env_params(subject, cluster)
    q_star = optimal_q(p)
    assert (q_star[[s for s in range(15) if s not in DECISION_STATES]] == 0).all()
    opt = evaluate_policy(p, greedy(q_star), n_episodes=500)["mean_reward"]
    for a in range(3):
        assert opt >= evaluate_policy(p, fixed_policy(a), n_episodes=500)["mean_reward"]


def test_metrics_report_every_environment():
    rl = load_metrics()["rl"]
    assert "SIMULATION" in rl["honesty_note"]
    for s, c in ENVS:
        e = rl["environments"][s][str(c)]
        for k in ("learned", "random", "always_quizzes", "best_fixed", "optimal_reference"):
            assert e["evaluation"][k]["n_episodes"] == C.RL_EVAL_EPISODES
        assert len(e["learning_curve_100"]) == C.RL_EPISODES // 100
        assert len(e["policy"]) == 5 and e["policy"][4] == [None, None, None]


# --------------------------------------------------------------------------- app helpers
@pytest.mark.parametrize("subject,cluster", ENVS)
def test_recommend_matches_q_table(subject, cluster):
    q = load_q_tables()[(subject, cluster)]
    for k in range(4):
        for e in range(3):
            out = recommend(subject, cluster, k, e)
            assert out["action"] == int(np.argmax(q[state_index(k, e)]))
            assert out["action_name"] == C.RL_ACTIONS[out["action"]] and len(out["q_values"]) == 3
    assert recommend(subject, cluster, 4, 1)["action"] is None


@pytest.mark.parametrize("args", [("mat", 5, 0, 0), ("por", 3, 0, 0), ("bio", 0, 0, 0), ("mat", 0, 5, 0),
                                  ("mat", 0, 0, 3), ("mat", 0, -1, 0), ("mat", 0, 1.5, 0), ("mat", 0, True, 0)])
def test_recommend_rejects_bad_input(args):
    with pytest.raises(ValueError):
        recommend(*args)


@pytest.mark.parametrize("subject,cluster", ENVS)
def test_simulate_episode(subject, cluster):
    steps = simulate_episode(subject, cluster, seed=3)
    assert steps == simulate_episode(subject, cluster, seed=3)
    assert 1 <= len(steps) <= C.RL_MAX_STEPS and steps[-1]["done"]
    assert np.isclose(sum(x["reward"] for x in steps), steps[-1]["cumulative_reward"])
    for x in steps:
        assert x["action"] == recommend(subject, cluster, x["knowledge"], x["energy"])["action"]
    for a, b in zip(steps, steps[1:]):
        assert (b["knowledge"], b["energy"]) == (a["next_knowledge"], a["next_energy"])


def test_centroids_cover_all_features():
    for s in ("mat", "por"):
        for z in cluster_centroids(s).values():
            assert set(z) == set(CENTROID_FEATURES)
