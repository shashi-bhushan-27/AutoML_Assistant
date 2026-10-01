# Contributing

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt -r requirements-dev.txt
```

`constraints.txt` records the full set of versions the test suite last passed with
(`pip install -r requirements.txt -c constraints.txt` reproduces it).

## Run

```bash
streamlit run app_frontend/main_ui.py
uvicorn app_backend.main_api:app --port 8000
```

Set `AUTOML_DATA_ROOT` to keep test workspaces out of the repository folder.

## Test

```bash
pytest -m "not slow"                     # unit, API (TestClient) and UI (AppTest) tests, no network
python ablation/run_ablation.py prepare  # downloads Adult, Credit Card Fraud, California Housing
pytest -m slow                           # parity and accuracy checks on the full datasets
```

After changing `app_frontend/ui/theme.py`, regenerate the Streamlit config with
`python -m app_frontend.ui.theme` (a test fails if it is out of date).

## Rules for changes

* One preprocessing pipeline: training, `/predict` and the exported script all go through `AutoPreprocessor`.
  Anything fitted must be fitted on the training rows and replayed by `transform()`; extend
  `tests/test_preprocessing.py` when you add a step.
* Say what happened: fallbacks, skipped steps, failed models and budget stops are shown in the UI, never
  swallowed. Do not add UI or documentation claims the code does not back.
* Charts: one y-axis, colours from `ui/theme.py`, a table view for every chart.
* Never log or commit `GROQ_API_KEY` or `.env`.
