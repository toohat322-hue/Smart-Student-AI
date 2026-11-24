# Smart Student AI

Student performance prediction (supervised), student clustering (unsupervised) and a simulated
study planner (reinforcement learning) on the UCI Student Performance dataset.

> The full README (screenshots, deployed link, citation, limitations) is written in Phase 9.
> Design decisions that differ from `PROJECT_SPEC.md` are logged in [DECISIONS.md](DECISIONS.md).

## How to retrain

Everything in `models/` and `reports/` is produced by one script, with fixed seeds (42). The
committed files are exactly what it produces, so you only need to retrain after changing code,
data or `src/config.py`.

```bash
# one-time setup (Python 3.11)
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt

# regenerate models/ and reports/ (~7–10 min on 4 cores)
python scripts/train_all.py

# also rerun the 20-split repeated evaluation (~30 min extra)
python scripts/train_all.py --full

# run the tests afterwards
pytest
```

What `train_all.py` runs, in order. It stops at the first failure.

| Step | Script | Produces | Time* |
|---|---|---|---|
| 1. data | `scripts/download_data.py` + schema validation | `data/student-{mat,por}.csv` (downloaded only if missing) | seconds |
| 2. eda | `scripts/eda.py` | figures 01–06, `metrics.json["eda"]` | seconds |
| 3. supervised | `scripts/train_supervised.py` | `models/clf_{mat,por}_{early,midterm}.joblib`, figures 07–12, `metrics.json["supervised"]` | ~8 min |
| 4. clustering | `scripts/train_clustering.py` | `models/kmeans_{mat,por}.joblib`, figures 14–16, `metrics.json["clustering"]` | seconds |
| 5. rl | `scripts/train_rl.py` | `models/q_tables.joblib`, figures 17–20, `metrics.json["rl"]` | ~2 min |
| 6. repeated (`--full` only) | `scripts/repeated_splits.py` | figure 13, `metrics.json["supervised_repeated_splits"]` | ~30 min |

\*On a 4-core laptop. Without `--full`, step 6 is skipped and the existing
`metrics.json["supervised_repeated_splits"]` is left as it is. Rerun with `--full` after any change that
affects the supervised models, so the repeated-split numbers stay in line with them.

**Checking reproducibility:** `scripts/fingerprint.py` hashes `metrics.json` and the predictions of every
saved model.

```bash
python scripts/fingerprint.py before.json
python scripts/train_all.py
python scripts/fingerprint.py after.json --compare before.json   # prints IDENTICAL or the differences
```

Each step can also be run on its own (e.g. `python scripts/train_rl.py`) if only that part changed. The
RL step reads the saved K-Means models, so rerun clustering before RL whenever clustering changes.

## Data

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5TG7T — licensed CC BY 4.0.
