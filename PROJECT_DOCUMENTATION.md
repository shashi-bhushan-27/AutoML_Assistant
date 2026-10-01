# AutoML Assistant – Technical documentation

This document describes what the code does. The in-app **Docs** page covers the same ground for users.

## 1. Architecture

```text
Streamlit UI (app_frontend/)                       FastAPI (app_backend/main_api.py)
  main_ui.py: st.navigation + sidebar stepper         /predict, /predict/csv, /workspaces, /schema
  views/*.py: one page per workflow step                        │
  ui/jobs.py: background threads for long tasks                 │
        │                                                        │
        ▼                                                        ▼
app_backend/  preprocessing_engine (AutoPreprocessor) ── the single pipeline used for training, serving and export
              model_registry → model_trainer → leaderboard
              statistical_engine → meta_learning → llm_rag_core (FAISS + Groq) → model_matcher
              shap_explainer · model_tuner (Optuna) · report_generator · code_generator
              workspace_manager (JSON + checksummed pickles under AUTOML_DATA_ROOT)
```

## 2. Preprocessing (`app_backend/preprocessing_engine/`)

`AutoPreprocessor.fit_transform(df)`:

1. **All rows, structural** – drop exact duplicate rows (counted and shown) and all-empty columns.
2. **All rows, descriptive** – profile for display; task detection (`task_types.detect_task_type`:
   classification if the target is text/bool or has ≤ 10 distinct values).
3. Drop rows with a missing target; label-encode classification targets (only the label vocabulary is taken
   from all rows; predictions are decoded back).
4. **Split** – stratified for single-target classification, chronological for time series, random otherwise
   (`test_size`, `random_state` = the workspace seed).
5. **Fitted on the training rows only**, applied to both partitions through one code path:
   `FeatureTransformer` (boolean/numeric-text/date detection, IQR caps – skipped when IQR = 0, the column is
   mostly zero or capping would leave one value –, log1p/sqrt for skewed columns) → `SmartImputer` (median /
   most frequent; a fill value for every column) → `SmartEncoder` (≤ 2 levels binary, 3–10 one-hot with a fixed
   vocabulary, 11–100 scikit-learn `TargetEncoder` with 5-fold cross-fitting on the training rows or frequency
   encoding for multi-class, > 100 frequency) → `FeatureSelector` (constant columns, |r| > 0.95) →
   `SmartScaler` (Standard or Robust on continuous columns; 0/1 indicators unscaled).
6. Optional SMOTE on the training rows; the outcome (applied / skipped / failed and why) is recorded.

`transform(raw_rows)` validates the schema (`SchemaError` for missing required columns or non-numeric values
in numeric columns), ignores extra columns, replays step 5 and returns the training columns in training order.
`parity_check()` compares `transform()` of held-out raw rows with their training-time representation.
Data are excluded from the pickle (`__getstate__`).

## 3. Recommendation

* `statistical_engine.analyze_dataset` – statistics for the prompt plus a 7-feature profile (log rows, feature
  count, categorical share, missing rate, minority share, mean |skew|, linearity). Does not modify its input.
* `meta_learning` – each profile feature is scaled to [0, 1]; similarity = 1 − RMS difference. Past workspaces of
  the same task with a best model and similarity ≥ 0.90 are added to the prompt (profiles are kept in
  `workspaces/index.json`, so no workspace file is opened).
* `llm_rag_core.ModelAdvisor` – FAISS (all-MiniLM-L6-v2, k = 3) over one chunk per rule, queried with a short
  description of the conditions that hold; Groq model from `GROQ_MODEL` (default `openai/gpt-oss-120b`);
  pydantic-validated JSON with one retry; returns `source` (`llm` / `fallback`), `error`, the retrieved rules
  and the raw answer. `check_llm_health()` looks the model up via the Groq API (no tokens).
* `model_matcher` – exact → alias table → rapidfuzz ratio ≥ 85 → unsupported / unmatched, in the LLM's order.

## 4. Training, metrics, tuning

* `model_registry` – the trainable models per task and their estimator classes.
* `ModelTrainer` – requires data from `AutoPreprocessor`; seeds every estimator that takes `random_state`;
  optional balanced class weights (sample weights); computes `predict_proba` once per model; records failures;
  optional `PrefitVotingEnsemble` (average of already-trained models, no refit, no learned weights).
* Metrics – regression: RMSE, MAE, R², MAPE; classification: accuracy, balanced accuracy, precision/recall/F1
  (minority class as positive for binary tasks), MCC, Cohen's kappa, ROC-AUC and PR-AUC (from probabilities
  or decision scores), per-class report.
* `leaderboard` – ranking metric by task (RMSE; accuracy; minority-class F1 when the minority is < 10%),
  best-model bookkeeping, majority-class baseline.
* `ModelTuner` – Optuna TPE within a time budget; `SEARCH_SPACES` is the only search space and the UI's range
  inputs override its bounds; StratifiedKFold / KFold / TimeSeriesSplit; tuned model refitted on the training
  split and scored on the same test split.

## 5. Explainability (`shap_explainer.py`)

Routing by estimator type: tree ensembles, decision trees, hist gradient boosting and XGBoost →
`TreeExplainer`; linear models → `LinearExplainer`; everything else (SVM, KNN, AdaBoost – which shap's
TreeExplainer does not support –, the voting ensemble) → `KernelExplainer` on a 10-centre k-means background
with `nsamples` scaled to the feature count. Rows are explained in chunks so progress, a time budget (default
120 s, checked between chunks) and cancellation work; an overrun returns the finished rows with status
`budget_exceeded`. Additivity failures are retried without the check and reported. Only bool/object columns
are cast (to float64).

## 6. Persistence and serving

`WorkspaceManager` writes `<id>.json`, `<id>_{pipeline,training_data,trained_models,state}.pkl`,
`<id>_manifest.json` (SHA-256 of each pickle, verified before unpickling) and `data/uploads/<id>_data.csv`
under `AUTOML_DATA_ROOT`. Only load workspaces you trust. The API caches loaded artefacts per (workspace,
model) keyed on file modification times, serves XGBoost from a NumPy array with one thread per request, and
returns decoded class labels with per-class probabilities.

## 7. UI (`app_frontend/`)

* `ui/theme.py` – design tokens (colour, spacing, radius, type) for light and dark; generates
  `.streamlit/config.toml`; CSS variables use `light-dark()` so they follow the active theme instantly;
  colour-blind-safe categorical order validated for both surfaces. `tests/test_theme.py` checks WCAG contrast.
* `ui/state.py` – workspace session, step status (done / stale / error / running / locked) from upstream
  fingerprints.
* `ui/jobs.py` – background threads with progress and cancel, polled by a fragment.
* `ui/charts.py` – Plotly figures: one y-axis, thin marks, identity never by colour alone (legend + line style
  + marker), table view and CSV/PNG download for every chart.

## 8. Known limits

* KernelExplainer on large data is slow by nature; expect the budget to stop it.
* SVM/KNN training is slow above ~20k rows, exact Gradient Boosting above ~100k.
* Time-series models (Prophet, ARIMA, SARIMAX, optional LSTM) are trained on a chronological split but are not
  served by `/predict`.
* Workspaces prepared before the pipeline rewrite must be re-prepared to be served.
