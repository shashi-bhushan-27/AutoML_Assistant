"""
Data Splitter Module
Train/test split: stratified for single-target classification, chronological for
time series, random otherwise. The split runs *before* any statistic is fitted.
"""
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from app_backend.task_types import is_classification

logger = logging.getLogger(__name__)


class DataSplitter:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.log: List[Dict[str, Any]] = []
        self.test_size = test_size
        self.random_state = random_state
        self.strategy: str = None

    def _log(self, step, action, reason, status="applied", fitted_on="all rows"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    def split(self, X: pd.DataFrame, y, task_type: str = "Classification", is_time_series: bool = False,
              date_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
        if is_time_series:
            split_idx = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            self.strategy = "chronological"
            self._log("Data Split", f"Chronological split at row {split_idx}",
                      f"Train: {len(X_train)}, Test: {len(X_test)} (no shuffling)")
            return X_train, X_test, y_train, y_test

        single_target = not isinstance(y, pd.DataFrame)
        if is_classification(task_type) and single_target:
            try:
                parts = train_test_split(X, y, test_size=self.test_size, stratify=y,
                                         random_state=self.random_state)
                self.strategy = "stratified"
                self._log("Data Split", "Stratified split",
                          f"Train: {len(parts[0])}, Test: {len(parts[1])}, seed {self.random_state}")
                return tuple(parts)
            except ValueError as exc:  # e.g. a class with a single row
                logger.warning("Stratified split failed (%s); using a random split", exc)
                reason = f"stratification impossible ({exc})"
        else:
            reason = "regression" if not is_classification(task_type) else "multi-output target"
        parts = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.strategy = "random"
        self._log("Data Split", "Random split",
                  f"Train: {len(parts[0])}, Test: {len(parts[1])}, seed {self.random_state} ({reason})")
        return tuple(parts)
