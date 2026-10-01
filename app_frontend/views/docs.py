"""Documentation page. Every statement here describes what the code does, including its limits."""
import streamlit as st

from app_backend import meta_learning
from app_backend.llm_rag_core import DEFAULT_GROQ_MODEL, RETRIEVAL_K
from app_backend.model_matcher import FUZZY_THRESHOLD
from app_backend.model_registry import MODEL_SPECS, TIME_SERIES_MODELS
from app_backend.shap_explainer import DEFAULT_TIME_BUDGET_S
from app_frontend.ui import nav


def render():
    st.title("Docs", anchor=False)
    st.caption("How the app works, what each step does and where its limits are.")
    tabs = st.tabs(["Workflow", "Preprocessing", "Recommendations", "Training & metrics", "Explain & tune",
                    "API", "Limits & troubleshooting"])
    with tabs[0]:
        st.markdown("""
A **workspace** holds one dataset and everything derived from it. The seven steps in the sidebar build on
each other; a step is locked until its prerequisite has run, and marked **stale** when something upstream
changed (for example a new upload or a different target), so you always know which results are current.

| Step | What happens |
|---|---|
| 1. Data | Upload a CSV (encoding detected with `chardet`, max 200 MB), preview it, remove columns. |
| 2. Prepare | Choose target(s), split and options; the preprocessing pipeline is fitted on the training rows. |
| 3. Recommend | Dataset statistics + retrieved rules are sent to an LLM; its model names are matched to trainable models. |
| 4. Train & Compare | Train the selected models; leaderboard on the test split, diagnostics, downloads. |
| 5. Explain | SHAP values for a trained model, with a time budget. |
| 6. Tune | Optuna search on the training split; before/after on the same test split. |
| 7. Deploy | Readiness check against the real `/predict` endpoint, request schema, examples, exports. |
""")
    with tabs[1]:
        st.markdown("""
**Order of operations** (`app_backend/preprocessing_engine/engine.py`):

1. *All rows, structural:* exact duplicate rows and all-empty columns are removed (the number is shown).
2. *All rows, descriptive:* a profile for display; the task is detected from the target
   (classification if the target is text or has ≤ 10 distinct values, otherwise regression).
3. Rows without a target are dropped; classification labels are encoded (the label list is the only thing
   taken from all rows) and decoded again in predictions.
4. **Train/test split** - stratified for single-target classification, chronological for time series.
5. *Fitted on the training rows only, applied to both partitions:* type and boolean detection, date features,
   outlier caps (Q1/Q3 ± 1.5×IQR; skipped for columns where it would produce a constant, e.g. mostly-zero
   columns), log/sqrt transforms for skewed columns, median / most-frequent imputation, encoding
   (≤ 2 levels binary; 3-10 one-hot with a fixed vocabulary; 11-100 target encoding with 5-fold cross-fitting,
   or frequency encoding for multi-class; > 100 frequency), removal of constant and |r| > 0.95 correlated
   features, scaling (Standard or Robust; 0/1 indicators are not scaled).
6. Optional SMOTE on the training rows only.

`transform()` replays step 5 with the fitted values, so serving uses exactly the training representation;
the Prepare page runs a per-feature **train/serve parity check** on held-out rows after every fit.
""")
    with tabs[2]:
        st.markdown(f"""
* **Retrieval:** the {RETRIEVAL_K} most similar rules from `knowledge_base/ml_rules.txt` (FAISS over
  `all-MiniLM-L6-v2` embeddings) for a short description of the conditions that hold (task, size, imbalance…).
  The rules only mention models this app can train.
* **LLM:** Groq, model from the `GROQ_MODEL` environment variable (default `{DEFAULT_GROQ_MODEL}`). The sidebar
  chip shows whether the model is served for your key. The answer must be JSON with `recommendations` and
  `reasoning`; invalid answers are retried once. If the LLM is unavailable or still invalid, the page says
  **Fallback used** and the task's default models are selected.
* **Name matching:** exact name → alias table (scikit-learn/XGBoost class names, abbreviations) → fuzzy match
  (rapidfuzz ratio ≥ {FUZZY_THRESHOLD:.0f}) → otherwise *unsupported* or *unmatched*, shown per name.
  Nothing is trained that the matcher did not map.
* **Meta-learning:** each workspace stores seven profile features (rows, features, categorical share,
  missing rate, minority share, mean |skew|, linearity), each scaled to [0, 1]. Similarity =
  1 − RMS difference. Past workspaces of the same task with a recorded best model and similarity ≥
  {meta_learning.SIMILARITY_THRESHOLD:.2f} are added to the prompt; the page lists them with their similarity.
  If none qualifies, no history is used.
""")
    with tabs[3]:
        cls = sorted(n for n, s in MODEL_SPECS.items() if "Classification" in s)
        reg = sorted(n for n, s in MODEL_SPECS.items() if "Regression" in s)
        st.markdown(f"""
**Models.** Classification: {', '.join(cls)}. Regression: {', '.join(reg)}.
Time series (regression, chronological split): {', '.join(TIME_SERIES_MODELS)} (LSTM if TensorFlow is installed).
An optional **voting ensemble** averages the trained models' probabilities (or predictions); it is not
re-fitted and learns no weights.

Every estimator that accepts a `random_state` gets the workspace's split seed, so results are reproducible.
Models that fail are listed with their error text, separately from the leaderboard.

**Ranking metric.** Regression: RMSE. Classification: accuracy, unless the minority class is under 10% of the
training rows - then F1 of the minority class, shown next to the majority-class baseline.

| Metric | Meaning |
|---|---|
| RMSE / MAE | typical prediction error in target units (lower is better) |
| R² | share of variance explained (1 is perfect, 0 = predicting the mean) |
| Accuracy / Balanced accuracy | share correct / mean recall over classes |
| Precision, Recall, F1 | for the minority (positive) class in binary tasks; weighted otherwise |
| AUC-ROC, PR-AUC | ranking quality from probabilities or decision scores |
| MCC, Cohen's kappa | agreement beyond chance, robust to imbalance |
""")
    with tabs[4]:
        st.markdown(f"""
**SHAP.** Tree models (random forest, extra trees, gradient boosting, hist gradient boosting, decision tree,
XGBoost) use `TreeExplainer`; linear models use `LinearExplainer`; everything else (SVM, KNN, AdaBoost, the
voting ensemble) uses the model-agnostic `KernelExplainer` on a 10-centre k-means summary of the training rows.
Kernel runs are explained row by row with progress; they stop at the time budget (default
{DEFAULT_TIME_BUDGET_S:.0f} s, checked between rows) and show how many rows were finished. The runtime is
estimated before you start. If TreeExplainer's additivity check fails, the values are recomputed without it
and a warning is shown.

**Tuning.** Optuna TPE with a time budget. The range inputs are exactly the search space the tuner uses.
Classification uses stratified K-fold, regression K-fold, time series `TimeSeriesSplit`; scoring follows the
ranking metric above. The tuned model is refitted on the full training split and scored on the unchanged
test split.
""")
    with tabs[5]:
        st.markdown(f"""
Start the API: `uvicorn app_backend.main_api:app --port 8000`, then open
[{nav.API_PUBLIC_URL}/docs]({nav.API_PUBLIC_URL}/docs) for the interactive reference.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness and cache statistics |
| `GET /workspaces` | workspaces and their trained models |
| `GET /workspaces/{{id}}/schema` | the columns a request needs, with kinds and examples |
| `POST /predict` | JSON rows → predictions (class labels), per-class probabilities, warnings |
| `POST /predict/csv/{{id}}/{{model}}` | CSV upload → the rows with prediction columns |

Missing required columns or non-numeric values in numeric columns return **HTTP 422** naming the columns;
extra columns are ignored and unseen categories are handled, both reported in `warnings`. Loaded artefacts
are cached per workspace and model and reloaded when the files change. The Deploy page generates examples
for your workspace.
""")
    with tabs[6]:
        st.markdown("""
* **Pickles.** Models and pipelines are stored as Python pickles with SHA-256 checksums. Loading a pickle can
  run code: only open workspaces and model files you created or trust.
* **LLM.** Needs `GROQ_API_KEY` (in `.env` or the environment). Without it, recommendations fall back to the
  task defaults and reports use a template; both are labelled.
* **SHAP on large data** with SVM/KNN is slow; expect the time budget to stop the run early.
* **Large datasets.** SVM and KNN get slow above ~20k training rows, exact Gradient Boosting above ~100k;
  Hist Gradient Boosting and XGBoost scale better.
* **Old workspaces.** Workspaces prepared before the pipeline rewrite cannot be served; re-run Prepare and Train.
* **"Could not decode the file".** Save the CSV as UTF-8 (or Latin-1) with comma separators.
""")
