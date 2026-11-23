# SMART STUDENT AI — Project Specification (for Claude Code)

> **Intelligent Student Performance Prediction & Personalized Learning System**
> Supervised · Unsupervised · Reinforcement Learning — Python · scikit-learn · Streamlit
>
> **How to use this file:** put it in the root of the repo (as `PROJECT_SPEC.md`), then in Claude Code say:
> *"Read PROJECT_SPEC.md and run Phase 1 only. Stop when its acceptance criteria are met."*
> Then move phase by phase (section 12 has a ready prompt for each phase).

---

## 0. ملخص التحليل (ما الذي عدّلناه على الخطة الأصلية)

الخطة الأصلية ممتازة كفكرة، لكن فيها نقاط لو تركناها سيسألك عنها الدكتور أو ستكسر المشروع عند النشر:

| # | المشكلة في الخطة الأصلية | الحل في هذه الوثيقة |
|---|---|---|
| 1 | ذكرت 649 طالباً فقط | ملف الرياضيات **395** طالباً، والبرتغالي **649**. حوالي **382** طالباً موجودون في الملفين |
| 2 | `studytime` وُصف بأنه "ساعات دراسة" | هو **فئة من 1 إلى 4** (<2h، 2–5h، 5–10h، >10h) وليس عدد ساعات |
| 3 | لم تحدد أي ملف ندرب عليه | نموذج **لكل مادة على حدة** (لا ندمج الملفين حتى لا يظهر نفس الطالب في Train وTest) |
| 4 | سيناريو واحد للتنبؤ | سيناريوهان: **Early** (بدون G1/G2) و**Mid-term** (مع G1/G2)، ونقارن بينهما |
| 5 | لم يُذكر عدم توازن الفئات | الراسبون أقلية (خصوصاً في البرتغالي)، لذلك: `stratify` + `class_weight="balanced"` + التركيز على **Recall لفئة At Risk** |
| 6 | Baseline غير محدد | `DummyClassifier` و`LogisticRegression` كخط أساس |
| 7 | حفظ النموذج فقط | حفظ **Pipeline كامل** (المعالجة + النموذج) حتى لا يختلف الترميز وقت التنبؤ |
| 8 | Q-Learning بدون تعريف بيئة | بيئة محاكاة محددة بالكامل (حالات، أفعال، مكافآت) ومربوطة بنتائج الـ Clustering |
| 9 | `requirements.txt` بدون إصدارات | تثبيت الإصدارات، لأن نموذج محفوظ بإصدار scikit-learn مختلف قد لا يفتح على Cloud |
| 10 | Gymnasium في المراجع وليس في التقنيات | بيئة بسيطة مكتوبة يدوياً بنفس واجهة Gymnasium (`reset`/`step`) بدون اعتماد عليها |
| 11 | لا يوجد تعامل مع الخصائص الحساسة | استبعاد `sex` من نموذج التنبؤ افتراضياً + فحص عدالة بسيط |
| 12 | ملف ZIP من UCI | الملف يحتوي داخله على `student.zip` آخر، والفاصل في CSV هو `;` وليس `,` |
| 13 | الروابط فيها `utm_source=chatgpt.com` وصور خارجية | روابط نظيفة فقط |

---

## 1. Project Overview

Build a Streamlit web app that applies three ML paradigms to the UCI Student Performance data:

1. **Performance Prediction (Supervised)** — classify a student as **Likely to Pass** (G3 ≥ 10) or **At Risk** (G3 < 10).
2. **Student Clustering (Unsupervised)** — discover groups of students with similar study/lifestyle patterns, without labels.
3. **Study Planner (Reinforcement Learning)** — a Q-Learning agent learns which study activity to recommend inside a **simulated** learning environment whose parameters come from the student's cluster.

**Academic goal:** show the practical difference between the three learning types, with honest, reproducible results.

**Integration story (important for the presentation):**
```
Student input ─► [Supervised] risk prediction
              └► [Unsupervised] cluster assignment ─► sets learner profile ─► [RL] recommended study policy
```

---

## 2. Dataset

### 2.1 Source
| Item | Value |
|---|---|
| Name | Student Performance (UCI ML Repository, dataset #320) |
| Page | https://archive.ics.uci.edu/dataset/320/student+performance |
| Direct ZIP | https://archive.ics.uci.edu/static/public/320/student+performance.zip |
| License | CC BY 4.0 — attribution required in README, app "About" page, and slides |
| Citation | Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T |
| Original paper | Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* Proc. 5th FUBUTEC, pp. 5–12. |
| Python alternative | `pip install ucimlrepo` → `fetch_ucirepo(id=320)` (returns one combined table; prefer the ZIP so we keep the two subjects separate) |

### 2.2 Files
| File | Subject | Rows | Columns |
|---|---|---|---|
| `student-mat.csv` | Mathematics | 395 | 33 |
| `student-por.csv` | Portuguese | 649 | 33 |

- **Separator is `;`** → `pd.read_csv(path, sep=";")`
- The outer ZIP contains an inner `student.zip` (plus `student.txt`, `student-merge.R`). The download script must handle the nested ZIP.
- No missing values (verify in EDA anyway).
- ~382 students appear in both files (the R merge script joins on 13 attributes). **Never concatenate the files into one training set.**
- Some students have **G3 = 0** (likely did not sit the final exam). Keep them as "At Risk" but report how many there are.

### 2.3 Full data dictionary (33 columns)

| # | Column | Type | Values |
|---|---|---|---|
| 1 | school | binary | GP / MS |
| 2 | sex | binary | F / M (**sensitive**) |
| 3 | age | numeric | 15–22 |
| 4 | address | binary | U urban / R rural |
| 5 | famsize | binary | LE3 / GT3 |
| 6 | Pstatus | binary | T together / A apart |
| 7 | Medu | ordinal | 0 none … 4 higher education |
| 8 | Fedu | ordinal | 0 none … 4 higher education |
| 9 | Mjob | nominal | teacher, health, services, at_home, other |
| 10 | Fjob | nominal | teacher, health, services, at_home, other |
| 11 | reason | nominal | home, reputation, course, other |
| 12 | guardian | nominal | mother, father, other |
| 13 | traveltime | ordinal | 1 <15min, 2 15–30min, 3 30–60min, 4 >1h |
| 14 | studytime | ordinal | 1 <2h, 2 2–5h, 3 5–10h, 4 >10h (weekly) |
| 15 | failures | ordinal | past class failures: n if 1≤n<3, else 4 |
| 16 | schoolsup | binary | yes / no — extra school support |
| 17 | famsup | binary | yes / no — family support |
| 18 | paid | binary | yes / no — extra paid classes |
| 19 | activities | binary | yes / no |
| 20 | nursery | binary | yes / no |
| 21 | higher | binary | yes / no — wants higher education |
| 22 | internet | binary | yes / no |
| 23 | romantic | binary | yes / no |
| 24 | famrel | ordinal | 1 very bad … 5 excellent |
| 25 | freetime | ordinal | 1 very low … 5 very high |
| 26 | goout | ordinal | 1 very low … 5 very high |
| 27 | Dalc | ordinal | workday alcohol, 1–5 |
| 28 | Walc | ordinal | weekend alcohol, 1–5 |
| 29 | health | ordinal | 1 very bad … 5 very good |
| 30 | absences | numeric | 0–93 |
| 31 | G1 | numeric | 0–20, period 1 grade |
| 32 | G2 | numeric | 0–20, period 2 grade |
| 33 | G3 | numeric | 0–20, final grade (**target source**) |

**Target:** `pass = 1 if G3 >= 10 else 0`. In the app, label 0 = **At Risk**, 1 = **Likely to Pass**.

---

## 3. Tech Stack (pinned)

| Area | Tool |
|---|---|
| Language | Python 3.11 (same version locally and on Streamlit Cloud) |
| Data | pandas, numpy |
| ML | scikit-learn |
| RL | NumPy-based Q-Learning (custom env, Gymnasium-style API) |
| UI | Streamlit (multipage via `pages/`) |
| Charts | Plotly (interactive), Matplotlib (static figures for the report) |
| Persistence | joblib |
| Tests | pytest |
| VCS | Git + GitHub |
| Hosting | Streamlit Community Cloud (alternative: Render) |

`requirements.txt` — **pin exact versions** after Phase 1 with `pip freeze`, keeping only these packages:
```text
pandas==<x>
numpy==<x>
scikit-learn==<x>
streamlit==<x>
plotly==<x>
matplotlib==<x>
joblib==<x>
```
`requirements-dev.txt`:
```text
-r requirements.txt
pytest
```

---

## 4. Project Structure

```text
Smart-Student-AI/
├── PROJECT_SPEC.md            # this file
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore                 # .venv/, __pycache__/, .pytest_cache/, *.zip, data/raw/
├── .streamlit/
│   └── config.toml            # theme (navy / white / light blue)
│
├── app.py                     # Home Dashboard (Streamlit entry point)
├── pages/
│   ├── 1_Performance_Prediction.py
│   ├── 2_Student_Clustering.py
│   ├── 3_Study_Planner.py
│   ├── 4_Model_Evaluation.py
│   └── 5_About_Project.py
│
├── scripts/
│   ├── download_data.py       # downloads + extracts nested ZIP into data/
│   └── train_all.py           # trains everything, writes models/ and reports/
│
├── src/
│   ├── __init__.py
│   ├── config.py              # paths, seeds, feature lists, thresholds
│   ├── data.py                # load_subject(), add_target(), validate_schema()
│   ├── preprocessing.py       # ColumnTransformer builders
│   ├── supervised.py          # train / evaluate / predict
│   ├── unsupervised.py        # k selection, fit, profile clusters, PCA
│   ├── rl_env.py              # StudyEnv (simulation)
│   ├── rl_agent.py            # QLearningAgent
│   └── ui.py                  # shared Streamlit helpers (cards, theme, loaders)
│
├── data/
│   ├── student-mat.csv        # committed (CC BY 4.0, with attribution)
│   └── student-por.csv
│
├── models/                    # committed, produced by scripts/train_all.py
│   ├── clf_por_early.joblib
│   ├── clf_por_midterm.joblib
│   ├── clf_mat_early.joblib
│   ├── clf_mat_midterm.joblib
│   ├── kmeans_por.joblib      # pipeline: scaler + kmeans
│   ├── kmeans_mat.joblib
│   └── q_tables.joblib        # dict: profile -> Q-table
│
├── reports/
│   ├── metrics.json           # every number shown in the app/slides comes from here
│   └── figures/               # PNGs for slides
│
├── notebooks/
│   └── 01_eda.ipynb           # optional exploration
│
└── tests/
    ├── test_data.py
    ├── test_supervised.py
    ├── test_unsupervised.py
    ├── test_rl.py
    └── test_app_smoke.py
```

**Why commit `data/` and `models/`:** Streamlit Cloud should not depend on UCI being reachable, and training on every app start is slow. Files are small (< 1 MB total).

---

## 5. Configuration (`src/config.py`)

```python
RANDOM_STATE = 42
PASS_THRESHOLD = 10
SUBJECTS = {"por": "student-por.csv", "mat": "student-mat.csv"}
DEFAULT_SUBJECT = "por"          # larger file

SENSITIVE = ["sex"]              # excluded from prediction by default
GRADE_COLS = ["G1", "G2", "G3"]

# Early scenario: everything except grades and sensitive attributes
# Mid-term scenario: Early features + G1, G2
# G3 is NEVER a feature.

CLUSTER_FEATURES = ["studytime", "absences", "failures", "freetime", "health", "goout"]
K_RANGE = range(2, 7)
```

---

## 6. Supervised Learning — Random Forest

**Task:** binary classification, `pass` (G3 ≥ 10).

**Scenarios (train both, per subject):**
| Scenario | Features | Use case |
|---|---|---|
| Early | all except `G1, G2, G3, sex` | start of year, before any grade (the useful one) |
| Mid-term | Early + `G1, G2` | after the 2nd period (much higher accuracy, expected) |

**Preprocessing** (`ColumnTransformer` inside a `Pipeline`):
- binary yes/no and 2-level categoricals → `OneHotEncoder(drop="if_binary")`
- nominal (`Mjob, Fjob, reason, guardian`) → `OneHotEncoder(handle_unknown="ignore")`
- ordinal + numeric → passthrough (trees don't need scaling; LogisticRegression baseline gets `StandardScaler`)

**Procedure:**
1. `train_test_split(test_size=0.2, stratify=y, random_state=42)`
2. Baselines: `DummyClassifier(strategy="most_frequent")`, `LogisticRegression(class_weight="balanced", max_iter=1000)`
3. `RandomForestClassifier(class_weight="balanced", random_state=42)` tuned with `GridSearchCV` (5-fold `StratifiedKFold`, scoring=`f1_macro`) over e.g. `n_estimators [200, 400]`, `max_depth [None, 5, 10]`, `min_samples_leaf [1, 3, 5]`
4. Evaluate on the held-out test set only once.

**Metrics** (save to `reports/metrics.json`):
- Accuracy, Precision, Recall, F1 **per class** and macro
- **Recall for At Risk** (headline metric — missing a struggling student is the costly error)
- ROC-AUC
- Confusion matrix
- 5-fold CV mean ± std
- Feature importances (permutation importance on the test set)

**Fairness check (short):** report At-Risk recall separately for `sex = F` and `sex = M` even though `sex` is not a feature.

**Prediction API:** `predict(subject, scenario, input_dict) -> {"label", "proba_pass", "top_factors"}`

---

## 7. Unsupervised Learning — K-Means

**Features:** `CLUSTER_FEATURES` (grades excluded on purpose, so clusters are about behaviour, not outcome).

**Procedure:**
1. `Pipeline([StandardScaler(), KMeans(n_init=10, random_state=42)])`
2. For k in 2..6: compute inertia (elbow) and Silhouette score.
3. Pick k by best Silhouette; if two values are close, prefer the more interpretable one and **write down why**.
4. Profile clusters: mean of each feature per cluster + cluster size + (after the fact) mean G3 and pass rate.
5. Name clusters **from the profile table**, e.g. "High absence", "Low study / high going out" — never assume "excellent/weak" in advance. Names are generated by a rule (feature furthest above/below overall mean), not hard-coded.
6. 2-D PCA scatter (Plotly) coloured by cluster; report PCA explained variance.

**Outputs:** silhouette per k, chosen k, profile table, PCA plot data → `metrics.json`.

**App:** user enters the 6 features → assigned cluster + "you are similar to N students with these traits".

---

## 8. Reinforcement Learning — Q-Learning Study Planner

> **Honesty statement (show in app + slides):** UCI data contains no sequences of actions and rewards. The RL part is a **simulation**. The agent learns from a synthetic environment whose parameters are set by the learner profile, not from real student trajectories.

### 8.1 Environment `StudyEnv` (Gymnasium-style: `reset(seed)`, `step(action)`)

| Element | Definition |
|---|---|
| State | `(knowledge, energy)` → knowledge ∈ {0,1,2,3,4}, energy ∈ {0 low, 1 mid, 2 high} → **15 states** |
| Actions | 0 Practice Quizzes · 1 Watch Videos · 2 Review Notes |
| Episode | 20 steps (one "study session" each) |
| Transition | knowledge goes up with probability `p_gain[profile][action][energy]`; each action has an energy cost; low energy reduces gain |
| Reward | `+10` per knowledge level gained, `−1` per step, `−5` if energy hits 0, `+20` bonus when knowledge reaches 4 (terminal) |

Default dynamics (tunable in `config.py`):
- **Quizzes:** highest gain when energy high, high energy cost
- **Videos:** moderate gain, low cost, good when knowledge is low
- **Review notes:** small gain, slightly restores energy

**Profiles:** 3 learner profiles derived from the K-Means cluster profile (e.g. high-absence cluster → lower base gain + faster energy drop). Map each cluster to a profile with a documented rule in `config.py`.

### 8.2 Agent `QLearningAgent`
- Q-table shape `(15, 3)`, update: `Q[s,a] += α (r + γ max Q[s'] − Q[s,a])`
- α = 0.1, γ = 0.9, ε-greedy with ε: 1.0 → 0.05 (exponential decay), 5,000 episodes, seed 42
- One Q-table per profile → saved in `q_tables.joblib`

### 8.3 Evaluation
- Learning curve: average reward per 100 episodes
- Compare learned policy vs **random policy** vs **always-quizzes** policy over 1,000 test episodes (mean ± std reward, avg steps to mastery)
- Policy heatmap: best action per (knowledge, energy)

**App:** choose profile (or take it from the clustering page), see the policy heatmap, and run an animated step-by-step episode simulation.

---

## 9. Dashboard (Streamlit)

**Theme** `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1E3A8A"          # navy
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#E0EDFF"  # light blue
textColor = "#0F172A"
```

| Page | Content |
|---|---|
| Home Dashboard | project summary, KPI cards (students per subject, pass rate, best model recall), 3 cards linking to the 3 modules; subject selector in sidebar (por / mat) |
| Performance Prediction | form with validated inputs (sliders/selects using real ranges from §2.3), scenario toggle Early / Mid-term, result card + pass probability gauge + top factors |
| Student Clustering | silhouette chart, PCA scatter, profile table, "find my cluster" form |
| Study Planner | profile picker, learning curve, policy heatmap, policy comparison, episode replay |
| Model Evaluation | metrics tables for all models vs baselines, confusion matrices, ROC curves, feature importance, Early vs Mid-term comparison, fairness check |
| About Project | dataset description, license & citation, method summary, limitations, team |

Rules:
- Load models with `@st.cache_resource`, data with `@st.cache_data`.
- **All numbers come from `reports/metrics.json` or live computation — never hard-coded.**
- If a model file is missing, show a clear message: "Run `python scripts/train_all.py`".
- English UI.

---

## 10. Testing (pytest)

| Test file | Checks |
|---|---|
| test_data.py | both CSVs load with `sep=";"`; shapes 395×33 and 649×33; expected column names; no NaNs; grade ranges 0–20; target creation correct |
| test_supervised.py | Early feature set contains no `G1/G2/G3/sex`; Mid-term has no `G3`; pipeline predicts on a single dict input; RF macro-F1 ≥ Dummy macro-F1; probabilities in [0,1] |
| test_unsupervised.py | labels count = chosen k; silhouette in [−1,1]; every cluster non-empty; deterministic with seed |
| test_rl.py | env reset returns valid state; step returns `(state, reward, done, info)`; episode ends ≤ 20 steps; trained policy mean reward > random policy mean reward |
| test_app_smoke.py | each page runs without exceptions using `streamlit.testing.v1.AppTest` |

Plus manual checks: invalid inputs, all pages, fresh clone → `pip install -r requirements.txt` → `streamlit run app.py`.

---

## 11. Deployment

### GitHub
1. Repo `Smart-Student-AI` (public is fine; data is CC BY 4.0).
2. Commit `data/`, `models/`, `reports/`. Don't commit `.venv/` or ZIPs.
3. README: description, screenshots, setup, run, train, test, deployed link, citation, limitations.

### Streamlit Community Cloud
1. https://share.streamlit.io → sign in with GitHub → **Create app**
2. Repo `Smart-Student-AI`, branch `main`, main file `app.py`
3. **Advanced settings → Python 3.11** (must match local)
4. Deploy, open from another browser, check logs if anything fails.
5. Free apps sleep after inactivity — open the link a few minutes before the presentation. Check current limits at deploy time.

Fallback: Render (https://render.com) with start command
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

---

## 12. Execution Phases — Prompts for Claude Code

Run **one phase per session/prompt**. Each phase ends with its acceptance criteria passing.

**Phase 1 — Setup**
> Create the folder structure from §4, `.gitignore`, `requirements*.txt` (unpinned for now), `.streamlit/config.toml`, `src/config.py` from §5, and an empty README. Create a venv, install, then pin versions with the installed ones.
> ✅ `pip install -r requirements-dev.txt` works; `python -c "import sklearn, streamlit"` works.

**Phase 2 — Data download & validation**
> Write `scripts/download_data.py` (download ZIP from §2.1, extract nested `student.zip`, copy the two CSVs to `data/`, skip if present). Write `src/data.py` with `load_subject(subject)`, `add_target(df)`, `validate_schema(df)`. Write `tests/test_data.py`.
> ✅ tests pass; shapes 395×33 and 649×33.
> (If UCI is unreachable from the sandbox, I will place the CSVs in `data/` manually.)

**Phase 3 — EDA**
> Produce `reports/figures/` charts: G3 distribution per subject, pass rate, G3=0 count, correlation of numeric features with G3, studytime/failures/absences vs pass. Save key stats to `reports/metrics.json["eda"]`.
> ✅ figures saved; summary printed.

**Phase 4 — Supervised**
> Implement §6 in `src/preprocessing.py` and `src/supervised.py`, both scenarios × both subjects, baselines, GridSearchCV, metrics, permutation importance, fairness check. Save pipelines to `models/`.
> ✅ `tests/test_supervised.py` passes; metrics in `metrics.json["supervised"]`.

**Phase 5 — Unsupervised**
> Implement §7 in `src/unsupervised.py`.
> ✅ `tests/test_unsupervised.py` passes; chosen k and profiles in `metrics.json["clustering"]`.

**Phase 6 — Reinforcement Learning**
> Implement §8 in `src/rl_env.py` and `src/rl_agent.py`, cluster→profile mapping, training, evaluation vs baselines.
> ✅ `tests/test_rl.py` passes; learned policy beats random; results in `metrics.json["rl"]`.

**Phase 7 — Training script**
> `scripts/train_all.py` runs phases 4–6 end-to-end with fixed seeds and regenerates `models/` and `reports/`.
> ✅ running it twice gives identical metrics.

**Phase 8 — Dashboard**
> Build `app.py`, `pages/*`, `src/ui.py` following §9. No hard-coded numbers.
> ✅ `streamlit run app.py` works; all pages load; `tests/test_app_smoke.py` passes.

**Phase 9 — Testing & polish**
> Full `pytest`, input validation, error messages, README with screenshots.
> ✅ all tests green from a fresh clone and a fresh venv.

**Phase 10 — Deployment**
> Guide me through GitHub push and Streamlit Cloud deploy (§11). Verify the live URL.

**Phase 11 — Presentation**
> Generate a 10-slide deck outline (§13) using real numbers from `metrics.json` and figures from `reports/figures/`.

---

## 13. Presentation (10 slides, English)

| # | Slide | Content |
|---|---|---|
| 1 | Title | name, team, course |
| 2 | Problem | late detection of struggling students |
| 3 | Objectives | predict, group, recommend; compare 3 ML paradigms |
| 4 | Data | UCI dataset, 2 subjects, 33 attributes, target definition, leakage prevention |
| 5 | Supervised | RF vs baselines, Early vs Mid-term, At-Risk recall, top features |
| 6 | Unsupervised | silhouette → k, PCA plot, cluster profiles |
| 7 | Reinforcement | env diagram, learning curve, policy heatmap, vs random |
| 8 | Architecture | data → models → Streamlit → Cloud |
| 9 | Live Demo | deployed app |
| 10 | Conclusion | results, limitations, future work |

Numbers on slides must come from `reports/metrics.json` only.

---

## 14. Limitations (say them before the professor does)

- Data is from two Portuguese schools (2005–2006); results may not generalize.
- Small samples (395 / 649) → wide confidence intervals; we report CV std.
- `studytime`, `freetime` etc. are self-reported categories, not measurements.
- Mid-term accuracy is high mainly because G2 is strongly correlated with G3 — the Early model is the honest, useful one.
- RL environment is synthetic; recommendations illustrate the method, they are not validated pedagogy.
- Clusters describe behaviour patterns; they are not judgments of students.

---

## 15. Definition of Done

- [ ] Original dataset files in `data/` with attribution
- [ ] Supervised code + 4 saved pipelines (2 subjects × 2 scenarios)
- [ ] Unsupervised code + saved K-Means pipelines
- [ ] RL env + agent + saved Q-tables
- [ ] `reports/metrics.json` with real computed results
- [ ] Streamlit app with 6 pages working locally
- [ ] All pytest tests passing from a clean environment
- [ ] GitHub repo with README
- [ ] Live Streamlit Cloud URL
- [ ] 10-slide presentation with real numbers and screenshots
- [ ] Limitations and references in README, app, and slides

---

## 16. References

- Cortez, P. (2008). *Student Performance* [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5TG7T
- Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. FUBUTEC 2008.
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- scikit-learn: https://scikit-learn.org/stable/
- Streamlit: https://docs.streamlit.io/
- Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud
- pandas: https://pandas.pydata.org/docs/
- Gymnasium (API reference for env design): https://gymnasium.farama.org/
