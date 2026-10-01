# Ablation study harness

Reproduces the ablation study of the RAG-AutoML paper against the code in this
repository. The application is **not modified**: the harness imports it and only
wraps, subclasses or monkeypatches components inside its own process.

| Ablation | What is changed (one thing at a time) | How |
|---|---|---|
| A. RAG engine | Full vs w/o RAG (empty context) vs w/o meta-learning (`similar_workspaces=None`) vs w/o fuzzy matcher (raw names to `ModelTrainer`); brute force = all supported models | `rag_ablation.py`: `ModelAdvisor` with a swapped retriever/LLM, the real `WorkspaceManager` in a temp dir, a verbatim port of the UI matcher, the real `ModelTrainer.run_selected_models` dispatch |
| B. Preprocessor | Stateful (`AutoPreprocessor.save` → `load` → `transform`) vs stateless (fresh `AutoPreprocessor` fitted per serving batch of 1/16/256/full rows); schema perturbations; real `/predict` via `TestClient` | `preproc_ablation.py` |
| C. SHAP | Routing + float32 vs forced `TreeExplainer` vs forced `KernelExplainer` (300 s timeout) vs no float32 cast | `shap_ablation.py`: subclasses of `SHAPExplainer`, each job in a forked child process |

`checks.py` adds deterministic checks of specific claims (matcher behaviour on
typical hallucinated names, meta-learning similarity rule, leakage through
target encoding, retrieval latency, …).

## Running

```bash
pip install -r requirements.txt fastapi httpx matplotlib psutil
# versions used for the reported numbers (the paper states scikit-learn 1.3 / XGBoost 2.0):
pip install "numpy==1.26.4" "pandas==2.2.3" "scikit-learn==1.3.2" "xgboost==2.0.3" "shap==0.46.0"
export GROQ_API_KEY=...            # read from the environment, never logged

python ablation/run_ablation.py all --workers 4 --calls 10
# or step by step:
python ablation/run_ablation.py prepare
python ablation/run_ablation.py checks
python ablation/run_ablation.py units --workers 4        # 3 datasets x 5 seeds
python ablation/run_ablation.py llm --calls 10           # needs the seed-0 units
python ablation/run_ablation.py aggregate
python ablation/run_ablation.py figures
```

Each unit runs single-threaded (`OMP_NUM_THREADS=1`), so timings are comparable
across models. `llm_calls.jsonl` is append-only and resumable.

`--out NAME` (before the sub-command) writes to `ablation/NAME/` and `ablation/figures_<suffix>/` instead of
`results/` and `figures/`; the re-run after the fixes on the `ui-redesign-and-fixes` branch used
`--out results_after_fixes`. The harness also runs on Windows (SHAP jobs use `spawn` when `fork` is not
available). Adaptations made for the post-fix code (public APIs changed): the matcher and best-model
bookkeeping come from `app_backend.model_matcher` / `app_backend.leaderboard` instead of the old UI ports,
meta-learning uses the profile similarity, SHAP records the app's own `budget_exceeded` status, the endpoint
test passes the training-time test representation explicitly (the pipeline pickle no longer carries data),
and the "as shipped" LLM configuration is the app's configured default model (`GROQ_MODEL`).

Maintenance commands (both refit only the models they need, with the same seeds,
and assert that the refit reproduces the brute-force metrics):

```bash
python ablation/run_ablation.py preproc-only --dataset credit --seeds 0 1   # redo ablation B for existing units
python ablation/run_ablation.py shap-rerun --workers 1                     # redo SHAP jobs killed by SIGKILL
python ablation/section_numbers.py                                         # numbers + ablation_section.tex
python ablation/merge_paper.py --original draft.tex --out merged.tex       # merge into the paper draft
```

KernelExplainer on KNN needs about 2 GB per job; with four units in parallel on a
16 GB machine the container's OOM killer can terminate a SHAP worker
(`status=crashed_process`, `exitcode=-9`). Such jobs are environment failures,
not application behaviour: `shap-rerun` repeats them and marks the record with
`rerun: true` and the reason. In the reported run this affected five jobs
(KNN on Adult seeds 2–3, Random Forest/forced Tree on California seed 4).

## Protocol notes

* Datasets: Adult (OpenML 1590, label `>50K`→1), Credit Card Fraud (OpenML 1597,
  29 features, no `Time` column), California Housing (scikit-learn, target in
  $100k). Each is round-tripped through CSV so dtypes match a UI upload.
* The app's `AutoPreprocessor.fit_transform` is applied to the full dataset (as
  the UI does); only the split seed (`splitter.random_state`) varies: 0–4.
* Models are fit with the app's defaults. Estimators with `random_state=None`
  draw from NumPy's global RNG, which is seeded before every fit.
* Model fits are deterministic given the seed, so the metrics of an LLM-selected
  set are looked up from the brute-force run of the same (dataset, seed) rather
  than refit; the name→class dispatch still runs through `run_selected_models`.
* The Groq model hard-coded in `llm_rag_core.py` (`llama-3.3-70b-versatile`) is
  no longer served. `as_shipped` records what the unmodified app does (every call
  falls back); the other LLM configurations substitute `openai/gpt-oss-120b`
  (`--model` to change) with the app's temperature, prompt, retriever and parser.

## Outputs

* `results/raw/` – one JSON per (dataset, seed) unit and `llm_calls.jsonl`
* `results/*.csv`, `results/summary.json`, `results/code_checks.json`
* `results/table_ablation.tex` – auto-generated table (source of the paper table)
* `results/paper_numbers.json` – every number quoted in the subsection
* `figures/*.pdf|png` – 300 dpi, colour-blind-safe palette
* `ablation_section.tex` – the LaTeX subsection (rendered from `ablation_section_template.tex`);
  `future_work_paragraph.tex` – replacement for the Future Work "Ablation Studies" paragraph.
  The merged paper (`paper_with_ablation.tex`) is produced locally and not committed.
