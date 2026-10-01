# AutoML Assistant

A workspace-based AutoML tool for tabular data: upload a CSV, prepare it with a leakage-free preprocessing
pipeline, get model suggestions from a retrieval-augmented LLM, train and compare models, explain them with
SHAP, tune them with Optuna and serve them through a FastAPI `/predict` endpoint.

The app is a multipage Streamlit UI (`app_frontend/`) over a Python backend (`app_backend/`).

## Workflow

| Step | What happens |
|---|---|
| 1. Data | CSV upload (encoding detected, max 200 MB), preview, column summary, column removal. |
| 2. Prepare | Pick target(s), split and options. The pipeline is **fitted on the training rows only** and replayed by `transform()` for test rows and serving; a train/serve parity check runs after every fit. |
| 3. Recommend | Dataset statistics + the 3 most relevant rules from `knowledge_base/ml_rules.txt` go to a Groq LLM (`GROQ_MODEL`, default `openai/gpt-oss-120b`). Output is schema-validated; names are matched to trainable models (exact / alias / fuzzy / unsupported / unmatched). If the LLM is unavailable the page says **Fallback used**. |
| 4. Train & Compare | Selected models in a background job; leaderboard ranked by RMSE (regression), accuracy, or minority-class F1 when the minority class is < 10% (with the majority-class baseline); failed models listed with their errors. Optional voting ensemble (plain average, clearly labelled). |
| 5. Explain | SHAP (Tree / Linear / Kernel explainer by model type) in a background job with a time budget, progress and cancel. |
| 6. Tune | Optuna on the training split with stratified CV and imbalance-aware scoring; before/after on the same test split. |
| 7. Deploy | Readiness check against the real `/predict` endpoint, request schema, generated curl/Python examples, training-script and report export. |

Steps are gated (a step is locked until its prerequisite ran) and marked **stale** when something upstream
changed.

## Quick start

Validated with Python 3.11/3.12 and the pinned versions in `requirements.txt`
(scikit-learn 1.3.2, XGBoost 2.0.3, SHAP 0.46.0, pandas 2.2.3, numpy 1.26.4).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Create `.env` in the repository root (never commit it):

```env
GROQ_API_KEY=your_groq_api_key
# optional
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REPORT_MODEL=openai/gpt-oss-20b
```

Run the UI and (optionally) the API:

```bash
streamlit run app_frontend/main_ui.py
uvicorn app_backend.main_api:app --port 8000     # interactive docs at http://localhost:8000/docs
```

Without `GROQ_API_KEY` everything works except the LLM parts, which fall back to default models / a template
report and say so. The FAISS index of the rules is built automatically on first use (and rebuilt when
`ml_rules.txt` changes); `python app_backend/llm_rag_core.py` rebuilds it by hand.

Environment variables: `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_REPORT_MODEL`, `AUTOML_DATA_ROOT` (where
`workspaces/` and `data/uploads/` live; default the repository root), `API_BASE_URL` (where the UI reaches the API for the Deploy
check; default `http://127.0.0.1:8000`), `API_PUBLIC_URL` (the API address shown in links and examples; default
`http://localhost:8000`).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness and cache statistics |
| `GET /workspaces`, `GET /workspaces/{id}/models` | workspaces and trained models |
| `GET /workspaces/{id}/schema` | the columns a request needs (kind, dtype, example, categories) |
| `POST /predict` | `{"workspace_id", "model_name", "data": [rows]}` → predictions (original class labels), per-class probabilities, warnings |
| `POST /predict/csv/{id}/{model}` | CSV upload → rows with prediction columns |

Missing required columns or non-numeric values in numeric columns → **HTTP 422** naming the columns. Extra
columns are ignored and unseen categories handled; both are reported in `warnings`. Loaded artefacts are
cached per workspace/model and reloaded when the files change.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"            # ~2 minutes, no network (Groq and the retriever are mocked)
pytest -m slow                  # needs the full benchmark datasets: python ablation/run_ablation.py prepare
```

## Evaluation

`ablation/` contains the ablation-study harness and its results. `ablation/results/` is the baseline run on the
code before the fixes on this branch; `ablation/results_after_fixes/` is the re-run after them
(`python ablation/run_ablation.py --out results_after_fixes all --workers 2`).

## Security

Workspaces store models and pipelines as Python pickles (with SHA-256 checksums that detect modification).
Unpickling can execute code: only open workspaces and model files you created or otherwise trust, and do not
expose the API to untrusted users without authentication.

## Layout

```text
app_frontend/  main_ui.py (entry, navigation, stepper) · views/ (one module per page) · ui/ (theme tokens,
               components, charts, jobs, state) · assets/style.css
app_backend/   preprocessing_engine/ (AutoPreprocessor) · model_registry · model_trainer · model_tuner ·
               model_matcher · shap_explainer · llm_rag_core · meta_learning · statistical_engine ·
               leaderboard · workspace_manager · main_api · code_generator · report_generator · ensemble
knowledge_base/ ml_rules.txt (the FAISS index is generated locally)
samples/       small public samples (Adult, California Housing) used by the app and the tests
tests/         pytest suite
ablation/      ablation-study harness and results
```

More detail: `PROJECT_DOCUMENTATION.md` and the in-app **Docs** page.
