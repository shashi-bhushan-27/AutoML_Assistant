"""
Scaler Module
Feature scaling fitted on the training rows. Binary / one-hot indicator columns (two
distinct values) are left as 0/1; continuous columns are scaled with RobustScaler
when more than 5% of training values lie beyond Q1/Q3 ± 3×IQR in some column,
otherwise StandardScaler. ``apply`` uses the fitted parameters directly on the
feature matrix (same arithmetic as scikit-learn's ``transform``).
"""
import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class SmartScaler:
    def __init__(self, strategy: str = "auto"):
        self.log: List[Dict[str, Any]] = []
        self.strategy = strategy
        self.scaler = None
        self.columns: List[str] = []
        self.chosen_strategy: str = None
        self._offset = self._factor = None

    def _log(self, step, action, reason, status="applied", fitted_on="train"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    @staticmethod
    def _has_outliers(M: np.ndarray) -> bool:
        for j in range(M.shape[1]):
            q1, q3 = np.quantile(M[:, j], [0.25, 0.75])
            iqr = q3 - q1
            if iqr and ((M[:, j] < q1 - 3 * iqr) | (M[:, j] > q3 + 3 * iqr)).mean() > 0.05:
                return True
        return False

    def fit(self, cols: Dict[str, np.ndarray]) -> "SmartScaler":
        self.columns = [c for c, a in cols.items() if np.unique(a).size > 2]
        if not self.columns:
            self._log("Scaling", "No continuous columns to scale", "Only indicator columns", status="skipped")
            return self
        M = np.column_stack([cols[c] for c in self.columns]).astype(float)
        strategy = self.strategy
        if strategy == "auto":
            strategy = "robust" if self._has_outliers(M) else "standard"
        self.chosen_strategy = strategy
        self.scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[strategy]().fit(M)
        if strategy == "standard":
            self._offset, self._factor = self.scaler.mean_, self.scaler.scale_
        elif strategy == "robust":
            self._offset, self._factor = self.scaler.center_, self.scaler.scale_
        reason = ("More than 5% extreme values in a column, RobustScaler" if strategy == "robust"
                  else "No heavy outliers, StandardScaler") if self.strategy == "auto" else f"{strategy} requested"
        self._log("Scaling", f"{strategy.capitalize()} scaling on {len(self.columns)} continuous columns",
                  reason + "; 0/1 indicator columns are not scaled")
        return self

    def apply(self, M: np.ndarray, positions: List[int]) -> np.ndarray:
        """Scale the given column positions of the feature matrix in place."""
        if self.scaler is None:
            return M
        if self.chosen_strategy == "minmax":
            M[:, positions] = M[:, positions] * self.scaler.scale_ + self.scaler.min_
        else:
            M[:, positions] = (M[:, positions] - self._offset) / self._factor
        return M
