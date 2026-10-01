"""
Metric choice and best-model bookkeeping shared by the UI, the workspace store,
meta-learning and the ablation harness (so they cannot disagree again).

* Regression: RMSE (lower is better).
* Classification: Accuracy, unless the minority class is under 10% of the training
  rows, in which case F1 of the minority class is used (accuracy of an all-majority
  predictor is already > 90% there). The majority-class baseline is reported next to
  the leaderboard.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app_backend.task_types import is_regression

IMBALANCE_THRESHOLD = 0.10


def minority_share(y) -> Optional[float]:
    if y is None or isinstance(y, pd.DataFrame):
        return None
    shares = pd.Series(np.asarray(y)).value_counts(normalize=True)
    return float(shares.min()) if len(shares) >= 2 else None


def choose_primary_metric(task_type, minority: Optional[float] = None) -> Dict[str, Any]:
    if is_regression(task_type):
        return {"metric": "RMSE", "higher_is_better": False, "reason": "regression: root mean squared error"}
    if minority is not None and minority < IMBALANCE_THRESHOLD:
        return {"metric": "F1 Score", "higher_is_better": True,
                "reason": f"minority class is {minority:.2%} of training rows (< 10%): F1 of the minority class "
                          "instead of accuracy"}
    return {"metric": "Accuracy", "higher_is_better": True, "reason": "classes reasonably balanced"}


def successful(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    if "Error" in results_df.columns:
        err = results_df["Error"]
        results_df = results_df[err.isna() | (err.astype(str).str.len() == 0)]
    return results_df


def select_best_model(results_df: pd.DataFrame, task_type, minority: Optional[float] = None,
                      metric: str = None) -> Optional[Dict[str, Any]]:
    """Best successful model under the task's primary metric (or ``metric`` if given)."""
    choice = choose_primary_metric(task_type, minority)
    if metric:
        choice = {"metric": metric, "higher_is_better": metric not in ("RMSE", "MAE", "MAPE (%)"),
                  "reason": "chosen by the user"}
    ok = successful(results_df)
    col = choice["metric"]
    if ok.empty or col not in ok.columns or ok[col].notna().sum() == 0:
        return None
    values = ok[col].astype(float)
    idx = values.idxmax() if choice["higher_is_better"] else values.idxmin()
    return {"model": ok.loc[idx, "Model"], "score": float(values.loc[idx]), **choice}


def majority_baseline(y_train, y_test, pos_label=None) -> Dict[str, float]:
    """Metrics of always predicting the most frequent training class."""
    from sklearn.metrics import accuracy_score, f1_score

    majority = pd.Series(np.asarray(y_train)).mode().iloc[0]
    y_true = np.asarray(y_test)
    pred = np.full(len(y_true), majority)
    out = {"majority_class": majority, "Accuracy": float(accuracy_score(y_true, pred))}
    labels = np.unique(y_true)
    if len(labels) == 2:
        pos = pos_label if pos_label is not None else pd.Series(np.asarray(y_train)).value_counts().idxmin()
        out["F1 Score"] = float(f1_score(y_true, pred, pos_label=pos, zero_division=0))
    else:
        out["F1 Score"] = float(f1_score(y_true, pred, average="weighted", zero_division=0))
    return out
