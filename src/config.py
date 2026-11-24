"""Central configuration: paths, seeds, feature lists, thresholds."""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --- Reproducibility & target ------------------------------------------------
RANDOM_STATE = 42
PASS_THRESHOLD = 10              # pass = 1 if G3 >= 10 else 0

# --- Subjects ----------------------------------------------------------------
SUBJECTS = {"por": "student-por.csv", "mat": "student-mat.csv"}
DEFAULT_SUBJECT = "por"          # larger file

# --- Features ----------------------------------------------------------------
SENSITIVE = ["sex"]              # excluded from prediction by default
GRADE_COLS = ["G1", "G2", "G3"]

# Early scenario: everything except grades and sensitive attributes
# Mid-term scenario: Early features + G1, G2
# G3 is NEVER a feature.

CLUSTER_FEATURES = ["studytime", "absences", "failures", "freetime", "health", "goout"]
K_RANGE = range(2, 7)

# --- Reinforcement learning (spec §8; Phase 6) --------------------------------
# SIMULATION ASSUMPTIONS, not learned from data: the UCI data has no action/reward sequences.
# One environment per K-Means cluster; its parameters come from the cluster's standardized
# centroid z (log1p(absences) -> StandardScaler space) through the linear formulas below.
RL_ACTIONS = ["Practice Quizzes", "Watch Videos", "Review Notes"]
RL_N_KNOWLEDGE = 5               # knowledge 0..4 (4 = mastery, terminal)
RL_N_ENERGY = 3                  # energy 0 low, 1 mid, 2 high
RL_START_STATE = (0, 2)          # (knowledge, energy) at reset
RL_MAX_STEPS = 20
RL_REWARD_LEVEL = 10.0           # per knowledge level gained
RL_REWARD_STEP = -1.0            # every step
RL_REWARD_ENERGY_ZERO = -5.0     # when energy drops to 0
RL_REWARD_MASTERY = 20.0         # bonus when knowledge reaches 4
RL_PROB_CLIP = (0.05, 0.95)

# Base dynamics for an average student (z = 0). Rows: action; columns: energy low / mid / high.
RL_BASE_GAIN = {
    0: [0.10, 0.35, 0.60],       # quizzes: best gain at high energy, poor when tired
    1: [0.20, 0.30, 0.35],       # videos: moderate, less sensitive to energy
    2: [0.10, 0.15, 0.20],       # review notes: small gain
}
RL_VIDEO_LOW_KNOWLEDGE_BONUS = 0.15   # videos: + when knowledge <= 1 (good for beginners)
RL_VIDEO_HIGH_KNOWLEDGE_PENALTY = 0.10  # videos: - when knowledge >= 3
RL_BASE_DRAIN = {0: 0.70, 1: 0.25}    # P(energy -1) after quizzes / videos (high / low cost)
RL_BASE_RESTORE = 0.50                # P(energy +1) after review notes
RL_BASE_SKIP = 0.10                   # P(session does not count) for an average student

# Cluster centroid (z units) -> environment parameters.
RL_GAIN_COEF = {"studytime": 0.15, "failures": -0.15}          # gain multiplier = 1 + sum(coef * z)
RL_GAIN_MULT_CLIP = (0.5, 1.5)
RL_DRAIN_COEF = {"goout": 0.08, "freetime": 0.04, "health": -0.08}  # added to drain, subtracted from restore
RL_SKIP_COEF = {"absences": 0.05}                              # added to skip probability

# Q-learning (spec §8.2)
RL_ALPHA = 0.1
RL_GAMMA = 0.9
RL_EPS_START = 1.0
RL_EPS_END = 0.05
RL_EPS_DECAY_FRACTION = 0.8      # epsilon reaches EPS_END after 80% of episodes, then stays there
RL_EPISODES = 5000
RL_EVAL_EPISODES = 1000
RL_EVAL_SEED_OFFSET = 100_000    # test episodes use seeds 100000..100999 (never used in training)
