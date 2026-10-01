"""
Training-script export.

The exported script re-runs the *same* ``AutoPreprocessor`` and model registry that the
app used (same seed, split and configuration) instead of re-implementing the
preprocessing by hand; the previous hand-written version used different encodings
(``get_dummies(drop_first=True)``) and so did not reproduce the app's features. It
therefore needs this repository on the Python path.
"""
import json
from datetime import datetime
from pprint import pformat
from typing import Dict, Optional

import numpy as np

from app_backend.task_types import normalize_task_type


def _safe_json(obj) -> Dict:
    if not obj:
        return {}
    return json.loads(json.dumps(obj, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


def _targets(target_col):
    if isinstance(target_col, (list, tuple)):
        return [str(c) for c in target_col if c]
    return [str(target_col)] if target_col else []


def generate_training_code(dataset_name: str, target_col, model_name: str, best_params: dict = None,
                           task_type: str = "Regression", preprocessor=None, seed: Optional[int] = None,
                           class_weight: Optional[str] = None) -> str:
    targets = _targets(target_col)
    task = (normalize_task_type(task_type) or normalize_task_type("Regression")).value
    params = _safe_json(best_params)
    base_model = model_name.replace(" (Tuned)", "")
    p = preprocessor
    seed = seed if seed is not None else (p.random_state if p is not None else 42)
    cfg = {
        "target_col": targets[0] if len(targets) == 1 else targets,
        "task_type": task,
        "test_size": getattr(p, "test_size", 0.2),
        "is_time_series": bool(getattr(p, "is_time_series", False)),
        "date_col": getattr(p, "date_col", None),
        "apply_smote": bool(getattr(p, "apply_smote", False)),
        "random_state": seed,
    }
    steps = ""
    if p is not None and getattr(p, "full_log", None):
        steps = "\n".join(f"#   - {s['step']}: {s['action']} (fitted on {s.get('fitted_on', 'n/a')})"
                          for s in p.full_log if s.get("status") == "applied")
    multi = len(targets) > 1
    wrapper = ("MultiOutputRegressor" if task == "Regression" else "MultiOutputClassifier") if multi else None
    weight_line = ""
    if class_weight == "balanced" and task == "Classification" and not multi:
        weight_line = ("from sklearn.utils.class_weight import compute_sample_weight\n"
                       "fit_kwargs = {'sample_weight': compute_sample_weight('balanced', y_train)}\n")
    else:
        weight_line = "fit_kwargs = {}\n"
    metric_block = (
        "from sklearn.metrics import mean_squared_error, r2_score\n"
        "print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)), ' R2:', r2_score(y_test, pred))\n"
        if task == "Regression" else
        "from sklearn.metrics import accuracy_score, classification_report\n"
        "print('Accuracy:', accuracy_score(y_test, pred))\n"
        "print(classification_report(y_test, pred, zero_division=0))\n")
    return f'''# =============================================================================
# AutoML Assistant - exported training script
# Generated : {datetime.now():%Y-%m-%d %H:%M}
# Model     : {model_name}
# Task      : {task}   Target(s): {", ".join(targets)}
# Dataset   : {dataset_name}
#
# This script re-runs the app's own preprocessing pipeline (AutoPreprocessor) with
# the same configuration and split seed, so the features match the ones the app
# trained on. Run it from the repository root (or add the repository to PYTHONPATH):
#   pip install -r requirements.txt && python this_script.py
#
# Steps the app applied (see the Prepare page):
{steps or "#   (pipeline not fitted when the script was generated)"}
# =============================================================================
import pickle

import numpy as np
import pandas as pd

from app_backend.model_registry import build_model
from app_backend.preprocessing_engine.engine import AutoPreprocessor

df = pd.read_csv("{dataset_name}")

prep = AutoPreprocessor(**{pformat(cfg)}, verbose=False)
out = prep.fit_transform(df=df)
X_train, X_test, y_train, y_test = out["X_train"], out["X_test"], out["y_train"], out["y_test"]
print(f"train {{X_train.shape}}  test {{X_test.shape}}")

best_params = {pformat(params)}
model = build_model("{base_model}", "{task}", random_state={seed}, params=best_params)
{f"from sklearn.multioutput import {wrapper}{chr(10)}model = {wrapper}(model)" if wrapper else ""}
{weight_line}model.fit(X_train, y_train, **fit_kwargs)

pred = model.predict(X_test)
{metric_block}
with open("{base_model.replace(" ", "_").lower()}_model.pkl", "wb") as f:
    pickle.dump(model, f)
prep.save("preprocessor_pipeline.pkl")  # use prep.transform(new_rows) before model.predict at serving time
'''
