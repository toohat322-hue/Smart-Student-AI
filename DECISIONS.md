# Decision log

Every decision that differs from, or adds to, `PROJECT_SPEC.md`. Newest phase last.
All numbers come from `reports/metrics.json` at the time of the decision.

---

## Phase 2 — Data

### D1. `failures` is 0–3, not 0–4
Spec §2.3 copies the UCI text ("n if 1 ≤ n < 3, else 4"), but both CSVs only contain 0, 1, 2, 3, where
3 means "3 or more". Input validation (`FEATURE_SPEC` in `src/data.py`) therefore accepts 0–3. Allowing 4
would let the app send a value the models have never seen.

---

## Phase 3 — EDA

### D2. G3 = 0 students: kept, labelled At Risk, treated as "did not sit the final exam"
There are 38 such students in Mathematics (9.6% of all, 29% of At Risk) and 15 in Portuguese (2.3%, 15%).
They look like dropouts, not like the weakest students:
- **Every one of them has 0 absences** (100% in both subjects; 13–22% among other At-Risk students),
  which points to absences no longer being recorded rather than perfect attendance.
- **The drop is all-or-nothing.** No Mathematics student has a G3 between 1 and 3, yet 11 students with
  G2 ≥ 8 end on 0. Everyone with G2 = 0 also has G3 = 0 (13 in Mathematics, 7 in Portuguese).
- **They started out weak**: mean G1 is about 7, and they have more past failures (53–60% vs 35–47%).

We keep them labelled At Risk, because dropping out is exactly what an early-warning tool should catch.
The side effect is that `absences = 0` became a false risk signal (see D3). Removing them was tested as a
sensitivity check in Phase 4 and rejected: Mathematics Early At-Risk recall fell from 0.42 to 0.32.

---

## Phase 4 — Supervised

### D3. Monotonic constraint on `absences`
The unconstrained Mathematics Early RF learned "0 absences = higher risk": mean predicted P(pass) was 0.517
at 0 absences vs 0.564 at 1–5 (partial dependence). That is an artefact of D2. All 4 RFs now use
`monotonic_cst = -1` on absences, so P(pass) can only stay the same or fall as absences rise. The
constrained Mathematics Early curve is 0.597 / 0.594 / 0.564 / 0.466 for 0 / 1–5 / 6–10 / >10. To make the
column index fixed before fitting, the preprocessor puts numeric columns first and takes its one-hot
categories from the data dictionary (absences = output column 12 in every model). The cost is small: test
macro-F1 −0.009 (Mathematics Early), −0.012 (Mathematics Mid-term), 0 for Portuguese. A test checks that
P(pass) is non-increasing for every test student as absences goes from 0 to 93.

### D4. Alert cut-off: tuned on training data, with a floor at 0.5
At the default 0.5 cut-off the Early models caught too few At-Risk students (seed-42 test: Mathematics 0.38,
Portuguese 0.35). The rule is:
- `threshold_raw` = the smallest cut-off whose **out-of-fold** At-Risk recall on the **training set** is
  ≥ 70% (5-fold CV with the best RF parameters). The test set is never used.
- `threshold = max(0.5, threshold_raw)`.

The floor was added in Phase 4c. Without it, models whose recall at 0.5 was already above 70% got a cut-off
below 0.5, which lowered recall. Over 20 splits, Mid-term recall fell from 0.93 to 0.69 (Mathematics) and
from 0.81 to 0.70 (Portuguese). Final cut-offs: Mathematics Early 0.575; Mathematics Mid-term, Portuguese
Early and Portuguese Mid-term 0.5 (raw 0.295 / 0.475 / 0.38). The cut-off lives inside the saved model
(`FixedThresholdClassifier`), so `predict()` always uses it. Applying the floor did not retrain anything:
predictions from the saved forests are identical before and after.

### D5. Repeated splits are the headline numbers
The single seed-42 split is unusually hard for Portuguese Early. Its test macro-F1 is 0.543, while the mean
over 20 other splits (seeds 0–19, same full procedure) is 0.697 ± 0.042. Seed 42 is 3.7 SD below that mean
and lower than all 20 splits. Its ROC-AUC is 0.65 vs a typical 0.85 ± 0.03. The cross-validation score is
genuine (GridSearchCV `mean_test_score` 0.773; training score 0.848), and the CV − test gap averages
+0.005 ± 0.049 across splits, so nothing is leaking. The app and slides show **mean ± std over the 20
splits** as the main result, and the seed-42 numbers only as "the split of the saved model".

Final-cut-off results over 20 splits (mean ± std):
| Model | Cut-off | Macro F1 | At-Risk recall | At-Risk precision | False alerts per test set |
|---|---|---|---|---|---|
| Mathematics Early | 0.583 ± 0.029 | 0.562 ± 0.042 | 0.696 ± 0.088 | 0.410 ± 0.035 | 26.2 of 53 |
| Mathematics Mid-term | 0.5 (floored in all 20) | 0.905 ± 0.031 | 0.929 ± 0.038 | 0.833 ± 0.058 | 5.0 of 53 |
| Portuguese Early | 0.539 ± 0.028 | 0.690 ± 0.033 | 0.698 ± 0.107 | 0.405 ± 0.043 | 20.7 of 110 |
| Portuguese Mid-term | 0.5 (floored in all 20) | 0.860 ± 0.030 | 0.805 ± 0.076 | 0.735 ± 0.067 | 6.0 of 110 |

### D6. Fairness is reported with pooled counts
`sex` is not a model feature, but At-Risk recall is reported per sex. Single-split groups are tiny (e.g. 17
girls and 9 boys At Risk in the Mathematics test set), so we pool caught / total counts over the 20 test
sets instead of averaging percentages (final cut-off):
| Model | Girls (F) | Boys (M) | Gap (F − M) |
|---|---|---|---|
| Mathematics Early | 0.647 (202/312) | 0.769 (160/208) | −0.12 |
| Mathematics Mid-term | 0.923 (288/312) | 0.938 (195/208) | −0.02 |
| Portuguese Early | 0.717 (129/180) | 0.682 (150/220) | +0.04 |
| Portuguese Mid-term | 0.861 (155/180) | 0.759 (167/220) | +0.10 |

**Caveat:** the 20 test sets overlap, so each student is counted about 4 times. The pooled counts are
therefore not independent samples and must not be used for significance tests. The real number of At-Risk
students is mat: 75 F / 55 M; por: 50 F / 50 M. Two gaps persist across splits and are reported in the app:
Mathematics Early catches fewer At-Risk girls (−12 points), and Portuguese Mid-term catches fewer At-Risk
boys (−10 points). Both models are left unchanged. Fixing this would need per-group cut-offs, which would
mean using `sex` at prediction time; we decided against that (spec §6).

### D7. Saved models are about 16 MB, not "< 1 MB"
Grid search picked unlimited-depth trees for the Portuguese models (up to 400 trees), so the 4 classifier
files total about 16 MB. That is well under GitHub's 100 MB per-file limit and loads fine on Streamlit
Cloud. We keep the tuned parameters instead of shrinking the models.

---

## Phase 5 — Clustering

### D8. `log1p(absences)` before scaling
`absences` is very right-skewed (skewness 3.67 in Mathematics, 2.02 in Portuguese; after log1p 0.07 and
0.18). The pipeline is `log1p(absences) → StandardScaler → KMeans`. Raw absences give slightly *higher*
silhouettes (Mathematics k = 6: 0.197 raw vs 0.170 log1p), but only by splitting off small groups of
absence outliers: raw k = 5 in Mathematics produces a 5-student cluster. With log1p, the smallest cluster
at any k is 28 students in Mathematics and 30 in Portuguese. Silhouettes at the chosen k: Mathematics
0.1729 with log1p vs 0.1736 raw; Portuguese 0.1626 vs 0.1685.

### D9. Choosing k: best silhouette, with k = 3 preferred when within 0.02
All silhouettes are low (0.15–0.18 with log1p), so these behaviour features do not form well-separated
groups. The clusters are a useful *partition* of a continuum, not natural types, and the app says so.
Rule: take the k with the best silhouette, but if k = 3 is within 0.02 of it, use k = 3. Three clusters
are the easiest to explain, and they map one-to-one onto the three RL learner profiles (spec §8).
- **Portuguese → k = 3.** Best is k = 6 (0.180); k = 3 scores 0.163 (−0.017, within tolerance).
- **Mathematics → k = 5.** Best is k = 5 (0.173); k = 3 scores 0.153 (−0.020, just outside), so the rule
  keeps k = 5. Its five clusters are distinct and readable, and the "High past failures" cluster has a 21%
  pass rate (overall 67%). Phase 6 maps the 5 clusters onto the 3 learner profiles.

### D10. Cluster names are generated, not hard-coded
Each cluster gets up to 2 features whose cluster mean is at least 0.5 SD from the overall mean (in the
standardized log1p space), strongest first, written as "High …" or "Low …". If no feature reaches 0.5 SD,
the name is "Close to average". Names are recomputed on every training run and tested for uniqueness.
Grades are never used to cluster; mean G3, pass rate and G3 = 0 counts are added afterwards for
description only.

### D11. G3 = 0 students do not pile into one cluster
In Mathematics, the 38 G3 = 0 students are split 8 / 9 / 10 / 3 / 8 across the 5 clusters. The most is 26%,
in "High past failures", which is 3.2× over-represented relative to its size (8% of students) but not a
majority. In Portuguese, the 15 are split 3 / 5 / 7, almost proportional to cluster size. Their recorded
0 absences (D2) is only one of six features, so it does not decide their cluster. No change to spec §14 is
needed, and the students stay in.

---

## Phase 6 — Reinforcement learning (simulation)

### D12. One environment per cluster, derived from the centroid (replaces spec §8's 3 fixed profiles)
There are 8 environments: 5 for Mathematics and 3 for Portuguese. **All parameters are simulation
assumptions, not learned from data**; the UCI files have no action or reward sequences. The base dynamics
for an average student (z = 0) were set once, from the action descriptions in §8.1, before any RL result
was seen, and they were not changed afterwards:
- **Gain** P(knowledge +1) for low / mid / high energy: quizzes 0.10 / 0.35 / 0.60, videos 0.20 / 0.30 /
  0.35, review notes 0.10 / 0.15 / 0.20. Videos get +0.15 at knowledge ≤ 1 and −0.10 at knowledge ≥ 3.
- **Energy**: quizzes lose 1 energy level with P = 0.70 and videos with P = 0.25; review notes restore 1
  level with P = 0.50. A session is skipped (does not count) with P = 0.10.

Each cluster's standardized centroid z (log1p(absences) → StandardScaler space, read from the saved K-Means
model) shifts these through linear formulas, with every coefficient in `src/config.py`:
- gain multiplier = clip(1 + 0.15·z_studytime − 0.15·z_failures, 0.5, 1.5): studying more helps, past
  failures hurt;
- drain shift = 0.08·z_goout + 0.04·z_freetime − 0.08·z_health, added to the quiz and video drain and
  subtracted from the review restore: a busy social life and poor health tire students faster;
- P(skip) = 0.10 + 0.05·z_absences: absent students miss sessions.

All probabilities are clipped to 0.05–0.95. Resulting gain multipliers run from 0.50 (Mathematics "High past
failures") to 1.29 (Mathematics "High study time"); P(quiz drain) from 0.57 to 0.79; P(skip) from 0.08 to
0.13. Other implementation details:
- "−5 if energy hits 0" is applied when energy *drops* to 0, not on every step spent at 0.
- Mastery (knowledge 4) is terminal. The 20-step limit is treated as a time-out, so the Q-update still uses
  the next state's value there, because the state has no clock.
- Every step draws the same 3 random numbers whatever the action, so all policies are compared on the same
  1,000 test seeds (100000–100999).

### D13. The learned policy beats the best fixed action in every environment, not just random
The best single fixed action is "Watch Videos" in all 8 environments. On the same 1,000 test seeds, the
learned policy scores **+9.9 to +16.7 reward** above it (standard error about 0.6), and +12.3 to +19.9 above
random. Mastery rates: learned 34–100% vs best fixed 18–84%. To check how good the learned policy is, the
exact optimal policy was computed by value iteration on the known simulator. This is an analysis reference
only; the agent never sees the model. The learned policy is 0.0–3.9 reward below the optimum. The largest
gaps are in the two "High past failures" environments (Mathematics 3.85, Portuguese 3.92).

### D14. Training does not fully converge with constant α = 0.1. The fix is a decaying learning rate, not more episodes
*(Superseded by D16 in Phase 6b: the decaying learning rate is now the default.)*
The ε-greedy training reward levels off, but the greedy policy keeps changing. In the last 5 checkpoints
(every 250 episodes), 0–4 of the 12 decision states change their best action, and greedy reward swings by
up to 6.7. Learned policies agree with the exact optimum in 7–10 of 12 states. More episodes do not fix this:
with constant α = 0.1, agreement at 5k / 20k / 50k episodes was 7–10 / 9–11 / 7–11 of 12. A per-state
learning rate α = 1/visits^0.6 reaches 10–12 of 12 states at 5,000 episodes, with a mean reward gap to the
optimum of 0.15 (vs 1.8 for constant α). The saved Q-tables keep the spec's α = 0.1; switching is a
one-line change (`alpha_power` in `src/rl_agent.py`) if wanted. The states that disagree are near-ties:
losing at most 1.9 Q-value units, and mostly under 0.9.

### D15. The clusters do not get genuinely different policies
**The exact optimal policy is identical in all 8 environments**:
- knowledge 0–1: videos at low and high energy, review notes at mid energy;
- knowledge 2–3: review notes at low and mid energy, quizzes at high energy.

The cluster parameters change how well a student does (optimal mean reward ranges from 19.7 to 51.6, and
mastery from 41% to 100%), but not which action is best. The 8 learned Q-tables do give 8 slightly different
policies, but every difference is Q-learning noise in near-tie states (D14), not a cluster effect. The app
must not present per-cluster strategies as different advice. It shows the same strategy, plus each
cluster's expected outcome (mastery rate, steps). Making policies differ would mean changing the simulator's
assumptions just to get that result, so we did not.

---

## Phase 6b — Decaying learning rate

### D16. Default learning rate: α = max(0.01, 1/visits^0.6) per (state, action)
All 8 Q-tables were retrained with it (same seed 42, same 5,000 episodes, same ε schedule). The spec's
constant α = 0.1 is still trained and stored as `comparison_constant_alpha` in `metrics.json["rl"]`, and
`reports/figures/20_rl_alpha_comparison.png` shows the two side by side. Results (decaying vs constant):

| Measure | Decaying α (saved) | Constant α = 0.1 |
|---|---|---|
| Decision states with the optimal action (of 12) | 10–12 | 7–10 |
| Reward gap to the exact optimum | 0.00 in 7 of 8 environments; 1.21 in Mathematics 3 | 0.00–3.92 |
| Greedy-reward range over the last 5 checkpoints | 0.0–2.0 | 0.9–6.7 |

The remaining disagreements are six cases across the 8 environments. Five are low-energy states at
knowledge 0–1, which the agent rarely visits (128–4,215 visits, against about 8,000–10,000 for common
states); each costs at most 1.14 in Q-value. The sixth is knowledge 1 / high energy in Mathematics 3,
costing 0.61. The learned policy now beats the best fixed action by
+11.3 to +18.0 reward (standard error about 0.6).

### D17. The best action depends on the student's current state, not their cluster
The exact optimal policy is the same in all 8 environments (D15). The decaying-α policies now come close to
it, so of their 5 distinct versions most differences sit in those rarely visited low-energy states. We
accept this rather than reshape the simulator:
- **The current state decides the action:** knowledge 0–1 → videos when energy is low or high, review notes
  when it is mid; knowledge 2–3 → review notes when energy is low or mid, quizzes when it is high.
- **The cluster changes the expected outcome:** mean reward 19.7 (Mathematics "High past failures") to
  51.6 (Mathematics "High study time"), mastery rate 41% to 100%, and 8.4 to 14.5 steps to mastery.

The app presents it this way: one strategy based on where the student is now, plus what students in their
cluster can expect.

---

## Phase 7 — End-to-end training

### D18. One script regenerates everything; the 20-split evaluation is optional
`python scripts/train_all.py` runs data → EDA → supervised → clustering → RL. Each step is a separate
process, and the script stops at the first failure. It takes about 7–10 minutes on a 4-core laptop
(supervised 350–490 s, RL about 55 s, the rest under 20 s). The 20-split evaluation (~30 min) runs only
with `--full`; otherwise `metrics.json["supervised_repeated_splits"]` is left untouched. The
`apply_threshold_floor.py` migration from Phase 4c is no longer needed, because training applies the floor
itself.

### D19. Reproducibility: two consecutive runs are identical to the byte
`scripts/fingerprint.py` hashes every section of `metrics.json` and the predictions of every saved model on
the full datasets (classifier probabilities and labels, K-Means labels and centroids, Q-tables), plus the
file bytes. The first check found two problems, both now fixed:
- **K-Means centroids varied in the last bits** (differences of about 1e-16) from one fit to the next,
  because multi-threaded K-Means adds numbers in a varying order. Cluster assignments never changed, but
  the RL environments are built from the centroids. K-Means now fits single-threaded
  (`threadpool_limits(1)`; threadpoolctl is already a scikit-learn dependency, so nothing new is added to
  `requirements.txt`). Centroids are now identical to the bit, and a test checks this.
- **A config key was missing from training.** The Phase 4c migration had recorded
  `supervised.config.threshold_floor`, and `train_supervised.py` did not write it. It does now. Apart from
  that key, a full retrain reproduced every supervised number written by the migration, which confirms the
  migration was exact.

After the fixes, two consecutive runs produced the same `metrics.json`, the same predictions from all 7
model files, and byte-identical model and figure files. Compared with the previously committed state,
classifier and Q-table predictions are identical, and the K-Means models differ only in those last-bit
centroid values.

---

## Phase 8 — Dashboard

### D20. Navigation with `st.navigation` and short URLs
`app.py` is still the entry point, but it defines the pages with `st.navigation`, so the first page is
titled "Home" instead of "app". The pages live in `pages/` as in spec §4, with short URL paths
(`/prediction`, `/clustering`, `/planner`, `/evaluation`, `/about`). Each page file also runs on its own,
which is how `tests/test_app_smoke.py` tests it.

### D21. One shared student, with real examples, loadable from the URL
The subject, the scenario and the student are kept in plain `st.session_state` keys, not widget keys, because
Streamlit can reset widget state on page switches. The three example students are real records from the
**seed-42 test split**, chosen by a rule from the Early model's pass probability (not picked by hand):
- clear pass: the passing student with the highest pass probability;
- clear At Risk: the failing student (G3 > 0) with the lowest;
- dropout: the G3 = 0 student with the lowest.

The app labels them as anonymised UCI records and shows their real final grade, noting that the model does
not use it. `?example=…` in the URL loads one on any page, which helps for live demos and for the
screenshot script.

### D22. What the app shows as headline numbers
- **Home and Prediction pages:** the Early model's At-Risk recall, precision and false-alert rate are the
  20-split means at the final threshold. The false-alert rate is the mean number of false alerts divided by
  the number of passing students in a test set.
- **Model Evaluation:** the 20-split tables come first; the seed-42 numbers sit in an expander and a second
  tab labelled "Saved model (seed 42)".
- **Study Planner:** it shows the student's cluster's learned strategy, says that the exact optimum is the
  same for every cluster, and presents clusters as different *expected outcomes* (D17), not different advice.

Numbers in the page text are read from `metrics.json` or computed live. That includes the 382 students who
appear in both files, recomputed with the UCI merge script's 13 join attributes, and the "counted about 4
times" figure in the fairness caveat (number of splits × test share).

### D23. Screenshots without new dependencies
`scripts/take_screenshots.py` drives the installed Google Chrome through Playwright, installed in a separate
virtual environment, so `requirements.txt` is unchanged. Streamlit scrolls inside its own container, so the
script waits until the app has been idle for 2 s, then grows the browser window to the content height before
capturing. It saves 6 desktop shots (1440 px wide) and 2 phone shots (390 px wide) of the Prediction and
Planner pages, all with the "Clear At Risk" example loaded.

---

## Phase 9 — Testing and polish

### D24. Clean-environment test (what Streamlit Cloud will do)
The committed repo was cloned into an empty folder, with a new Python 3.11.9 venv and **only**
`requirements.txt` installed (`pip install --no-cache-dir`, 205 s; no pytest, Playwright or other local
tools). Results:
- **App:** `streamlit run app.py` from the clone loaded all 6 pages with the committed models (no retraining),
  for both subjects, with and without an example student: 24 page loads, 0 exceptions, 0 error boxes.
- **Tests:** after adding `requirements-dev.txt`, all 261 tests passed in the clone (67 s), and running them
  left no committed file modified.
- **Only worked because of the local setup:** `threadpoolctl` is imported directly
  (`src/unsupervised.py`, D19) but was not in `requirements.txt`. It was installed anyway as a scikit-learn
  dependency, which is fragile, so it is now pinned (`threadpoolctl==3.8.0`).
- Nothing else depended on the local machine:
  - every file the code references is committed and matches its exact name and capitalisation (Linux file
    names are case-sensitive);
  - there are no absolute or backslash paths; all paths are built with `pathlib` from the project root;
  - text files are stored with Linux line endings in git.

### D25. Deployment readiness
- **Memory:** with all 7 model files loaded and every page visited for both subjects, the server process
  peaked at **259 MB** (Windows working set). Models and data are cached once per server with
  `st.cache_resource` / `st.cache_data`, so extra visitors add only their small session state. Streamlit
  Community Cloud's free tier has about 1 GB; check the current limit at deploy time. Low risk.
- **Cold start:** the server is ready in about 3 s. The first page took 21 s on the very first launch of a fresh
  install (first-time file access and virus scanning on Windows) and 8 s on the next restart. Later pages take
  about 4–5 s each, including model loading on first use. The free tier puts the app to sleep when unused, so
  open it a few minutes before presenting.
- **Repo size:** 22.7 MB of tracked files. The largest are the classifiers (6.8, 5.6, 2.2 and 1.4 MB), then
  screenshots (up to 0.6 MB) and `metrics.json` (0.6 MB). The git history is about 18 MB compressed, because
  the models changed in several commits. Everything is far below GitHub's 100 MB per-file limit.
- **Ignored by git:** `.venv/`, `__pycache__/`, `.pytest_cache/`, `data/raw/` and `*.zip`. The Playwright
  venv lives outside the repo (in a temporary folder).
- **Error details:** `.streamlit/config.toml` sets `client.showErrorDetails = "type"`, so an unexpected error
  shows only its type to visitors, never a traceback; full details go to the server log.

### D26. Input and error handling
- **Form inputs:** every field is a selectbox, radio button, slider or bounded number box built from
  `FEATURE_SPEC`, so an out-of-range or wrong-type value cannot be entered. Tests confirm that absences 500 or
  −3, and a simulation seed of 10^9, are rejected by the widgets without any error. `predict()`,
  `assign_cluster()` and `recommend()` still validate every value themselves (D1, Phase 4 tests).
- **Link parameters:** an unknown `?example=` or `?subject=` value now shows a warning listing the valid values
  (previously it was silently ignored).
- **Stale session:** an incomplete, wrongly typed or invalid student left in the session (for example after
  an app update) used to crash three pages with a `KeyError`. Every page now checks the stored student against
  the data dictionary, clears it and shows a message.
- **Missing files:** a missing model file or a missing `metrics.json` section shows "… Run
  `python scripts/train_all.py`" on every page that needs it (tested).

### D27. Code tidy-up without behaviour change
- Removed the Phase 4c one-off migration `scripts/apply_threshold_floor.py` (training applies the floor
  itself since Phase 7; the script remains in git history), one unused helper (`ui.as_int_array`) and two
  unused `numpy` imports. pyflakes now reports nothing.
- Added a one-line docstring to each of the 99 public functions that lacked one in `src/` and `scripts/`.
- Moved the About page's Team / Course / Instructor lines into `src/config.py` (`PROJECT_TEAM`,
  `PROJECT_COURSE`, `PROJECT_INSTRUCTOR`), so they are filled in one place.
- **Proof of no behaviour change:** `scripts/fingerprint.py` before the tidy-up and after a full retrain with
  the tidied code is **IDENTICAL**: every `metrics.json` section, the predictions of all 7 model files, and the
  bytes of every model and figure file.

---

## Phase 9b — Fair threshold comparison with Logistic Regression

### D28. Logistic Regression gets the same threshold rule; the Random Forest stays
Until now the Random Forest had a tuned alert threshold and Logistic Regression used 0.5, which was not a fair
comparison. Logistic Regression now gets the same rule: out-of-fold on training data, at least 70% At-Risk
recall, never below 0.5. This applies on the seed-42 split and in all 20 repeated splits (mean LogReg
threshold: Mathematics Early 0.62, Portuguese Early 0.56, Mid-term 0.50 in every split). Random Forest minus
Logistic Regression, paired over the same 20 splits (mean, standard error):

| Model | Macro F1 | At-Risk recall | At-Risk precision | False alerts per test set |
|---|---|---|---|---|
| Portuguese Early | **+0.035** (0.008) | −0.050 (0.027) | **+0.054** (0.010) | **−7.2** of 110 |
| Portuguese Mid-term | **+0.042** (0.007) | −0.035 (0.016) | **+0.125** (0.015) | **−5.0** of 110 |
| Mathematics Early | −0.005 (0.012) | +0.010 (0.024) | −0.003 (0.010) | +0.8 of 53 |
| Mathematics Mid-term | +0.008 (0.005) | +0.025 (0.012) | −0.001 (0.009) | +0.1 of 53 |

**Decision: the saved Random Forest models stay.** Logistic Regression is not better on the 20-split means:
- **Portuguese:** the Random Forest is clearly ahead on macro F1 (4–6 standard errors), precision and false
  alerts. Logistic Regression catches a few more At-Risk students only by flagging many more students overall.
- **Mathematics:** the two models are tied for Early and the Random Forest is slightly ahead for Mid-term.

On the single seed-42 split, Logistic Regression has slightly higher macro F1 in both Mid-term cases
(Portuguese 0.798 vs 0.779, Mathematics 0.851 vs 0.839). This is the single-split noise D5 warns about.
Retraining with `--full` left every model file and figure byte-identical; only the supervised sections of
`metrics.json` gained the new LogReg entries. The README results table and the Model Evaluation page now
compare both models at their own tuned thresholds.

---

## Future work
- **Cluster-specific action effectiveness.** For example, if videos helped weaker students more, the
  clusters would get different policies. Modelling that needs data on which study activities students
  used and how they progressed afterwards, which we do not have.
