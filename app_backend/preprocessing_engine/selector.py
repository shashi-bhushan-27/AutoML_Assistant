"""
Feature Selector Module
Removes constant and highly correlated features, decided on the training rows only.

The variance filter used to drop every column with variance < 0.01 on the raw scale,
which is scale-dependent and removed rare one-hot levels. It now drops only columns
that are constant in the training rows. The correlation filter (|r| > 0.95, keep the
first of each pair) is unchanged but fitted on train.
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureSelector:
    def __init__(self, correlation_threshold: float = 0.95):
        self.log: List[Dict[str, Any]] = []
        self.correlation_threshold = correlation_threshold
        self.dropped_variance: List[str] = []
        self.dropped_correlation: List[str] = []
        self.correlated_with: Dict[str, str] = {}
        self.selected_features: List[str] = []

    def _log(self, step, action, reason, status="applied", fitted_on="train"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    def fit(self, cols: Dict[str, np.ndarray]) -> "FeatureSelector":
        names = list(cols)
        self.dropped_variance = [c for c in names if np.unique(cols[c][~np.isnan(cols[c])]).size <= 1]
        if self.dropped_variance:
            self._log("Constant Filter", f"Dropped {len(self.dropped_variance)} constant columns",
                      f"Single value in the training rows: {self.dropped_variance[:8]}")
        else:
            self._log("Constant Filter", "No constant columns", "Every feature varies in the training rows",
                      status="skipped")
        remaining = [c for c in names if c not in self.dropped_variance]
        self.dropped_correlation = []
        if len(remaining) >= 2:
            corr = pd.DataFrame({c: cols[c] for c in remaining}).corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            for col in upper.columns:
                partners = upper.index[upper[col] > self.correlation_threshold].tolist()
                if partners:
                    self.dropped_correlation.append(col)
                    self.correlated_with[col] = partners[0]
        if self.dropped_correlation:
            pairs = [f"{c}~{self.correlated_with[c]}" for c in self.dropped_correlation[:6]]
            self._log("Correlation Filter", f"Dropped {len(self.dropped_correlation)} columns",
                      f"|r| > {self.correlation_threshold} with a kept column: {pairs}")
        else:
            self._log("Correlation Filter", "No highly correlated pairs",
                      f"All |r| <= {self.correlation_threshold}", status="skipped")
        dropped = set(self.dropped_variance) | set(self.dropped_correlation)
        self.selected_features = [c for c in names if c not in dropped]
        return self

    def get_dropped_features(self) -> Dict[str, List[str]]:
        return {"low_variance": self.dropped_variance, "high_correlation": self.dropped_correlation}
