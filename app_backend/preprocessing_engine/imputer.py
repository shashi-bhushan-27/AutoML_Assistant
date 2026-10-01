"""
Imputer Module
Missing-value handling fitted on the training partition.

Numeric columns: median. Categorical columns: most frequent value. Columns with
more than 50% missing values in the training partition are dropped. A fill value
is stored for *every* column (not only those with gaps in training), so a gap that
first appears at serving time is filled with the training statistic instead of
reaching the model as NaN.

(The previous single-column KNNImputer was removed: KNN on one column is a mean of
arbitrary neighbours, and it was not replayed at serving.)
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
Columns = Dict[str, np.ndarray]


class SmartImputer:
    def __init__(self, drop_threshold_pct: float = 50.0):
        self.log: List[Dict[str, Any]] = []
        self.drop_threshold_pct = drop_threshold_pct
        self.imputers: Dict[str, Dict[str, Any]] = {}
        self.strategies: Dict[str, str] = {}
        self.dropped_columns: List[str] = []

    def _log(self, step, action, reason, status="applied", fitted_on="train"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    def fit(self, cols: Columns) -> "SmartImputer":
        for col, a in cols.items():
            missing = pd.isna(a)
            missing_pct = float(missing.mean() * 100) if len(a) else 0.0
            if missing_pct > self.drop_threshold_pct:
                self.dropped_columns.append(col)
                self.strategies[col] = "drop"
                self._log("Imputation", f"Dropping column: {col}",
                          f"{missing_pct:.1f}% missing in the training rows (> {self.drop_threshold_pct:.0f}%)")
                continue
            if a.dtype.kind == "f":
                value = float(np.nanmedian(a)) if (~missing).any() else 0.0
                strategy = "median"
            else:
                mode = pd.Series(a[~missing]).mode()
                value = mode.iloc[0] if len(mode) else "__MISSING__"
                strategy = "mode"
            self.imputers[col] = {"strategy": strategy, "value": value}
            self.strategies[col] = strategy
            if missing_pct > 0:
                shown = f"{value:.4g}" if isinstance(value, float) else repr(value)
                self._log("Imputation", f"{strategy.title()} impute: {col}",
                          f"{missing_pct:.1f}% missing in training rows; fill value {shown}")
        if not any(e["step"] == "Imputation" for e in self.log):
            self._log("Imputation", "No missing values in training rows",
                      "Fill values are still stored for every column in case serving data has gaps",
                      status="skipped")
        return self

    def transform(self, cols: Columns, warn: List[str] = None) -> Columns:
        out: Columns = {}
        for col, a in cols.items():
            if col in self.dropped_columns:
                continue
            info = self.imputers.get(col)
            if info is not None:
                missing = pd.isna(a)
                if missing.any():
                    a = a.copy()
                    a[missing] = info["value"]
                    if warn is not None:
                        warn.append(f"{col}: {int(missing.sum())} missing value(s) filled with the training "
                                    f"{info['strategy']}")
            out[col] = a
        return out
