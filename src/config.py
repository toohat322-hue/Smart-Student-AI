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
