"""
Shared helpers for the RAG-AutoML ablation harness.

Nothing in here changes application behaviour: the app modules are imported
unmodified and only wrapped, subclassed or monkeypatched *inside the harness
process* where an ablation needs it.
"""
import io
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ABLATION_DIR)
RESULTS_DIR = os.path.join(ABLATION_DIR, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
FIG_DIR = os.path.join(ABLATION_DIR, "figures")
CACHE_DIR = os.path.join(ABLATION_DIR, ".cache")
TMP_DIR = os.path.join(ABLATION_DIR, "_tmp")

for _d in (RESULTS_DIR, RAW_DIR, FIG_DIR, CACHE_DIR, TMP_DIR):
    os.makedirs(_d, exist_ok=True)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["adult", "credit", "california"]

DATASET_META = {
    "adult": {"label": "Adult", "target": "class", "task": "Classification"},
    "credit": {"label": "Credit Card Fraud", "target": "Class", "task": "Classification"},
    "california": {"label": "California Housing", "target": "MedHouseVal", "task": "Regression"},
}

# Models used by the SHAP and preprocessing ablations (task-specific linear model).
SHAP_MODELS = {
    "Classification": ["Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression", "SVM", "KNN"],
    "Regression": ["Random Forest", "XGBoost", "Gradient Boosting", "Ridge", "SVM", "KNN"],
}
PREPROC_MODELS = {
    "Classification": ["XGBoost", "Random Forest", "Logistic Regression"],
    "Regression": ["XGBoost", "Random Forest", "Ridge"],
}


def _csv_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """Serialise to CSV and read back so dtypes match what a user upload gives the app."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def load_dataset(name: str) -> pd.DataFrame:
    """Load one benchmark dataset exactly as a CSV upload would present it to the app.

    * Adult: OpenML 1590 (48,842 rows, 14 features). The income label is stored
      as an integer 0/1 (">50K" -> 1). With the raw string label the app's
      XGBoost path fails (XGBClassifier rejects string classes); see
      ``check_string_label_xgboost`` in run_ablation.py.
    * Credit Card Fraud: OpenML 1597 (284,807 rows, 30 features, label 0/1).
    * California Housing: sklearn ``fetch_california_housing`` (20,640 rows,
      8 features, target in units of $100k).
    """
    cache = os.path.join(CACHE_DIR, f"{name}.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)

    from sklearn.datasets import fetch_openml, fetch_california_housing

    if name == "adult":
        bunch = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        df["class"] = (df["class"].astype(str).str.strip().str.startswith(">50K")).astype(int)
    elif name == "credit":
        bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        df["Class"] = df["Class"].astype(str).str.strip().astype(float).astype(int)
    elif name == "california":
        bunch = fetch_california_housing(as_frame=True, data_home=CACHE_DIR)
        df = bunch.frame.copy()
    else:
        raise ValueError(name)

    df = _csv_roundtrip(df)
    df.to_pickle(cache)
    return df


def primary_metrics(task: str):
    """(selection metric used by the app, secondary metric, higher_is_better)."""
    if task == "Regression":
        return "RMSE", "R²", False
    return "Accuracy", "F1 Score", True


def environment_info() -> dict:
    import importlib

    versions = {}
    for mod in ["numpy", "pandas", "sklearn", "xgboost", "shap", "scipy", "faiss",
                "langchain_core", "langchain_groq", "langchain_classic", "langchain_community",
                "sentence_transformers", "torch", "fastapi", "statsmodels"]:
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - informational only
            versions[mod] = f"unavailable ({type(exc).__name__})"
    try:
        import psutil
        mem_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        mem_gb = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_gb": mem_gb,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "packages": versions,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return str(o)


def dump_json(obj, path):
    with open(path, "w") as f:
        f.write(json.dumps(_clean_nans(obj), indent=2, cls=NpEncoder))


def _clean_nans(obj):
    if isinstance(obj, dict):
        return {str(k): _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating) and not np.isfinite(obj):
        return None
    return obj


def load_json(path):
    with open(path) as f:
        return json.load(f)
