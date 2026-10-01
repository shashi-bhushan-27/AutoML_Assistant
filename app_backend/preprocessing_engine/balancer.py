"""
Class Imbalance Handler Module
Detects imbalance on the training labels and, when the user opts in, applies
SMOTE to the training rows only. The outcome (applied / skipped / failed and why)
is always recorded in ``smote_outcome`` so the UI can show it.
"""
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

IMBALANCE_THRESHOLD = 0.10  # minority share below which the task counts as imbalanced


class ImbalanceHandler:
    def __init__(self, threshold: float = IMBALANCE_THRESHOLD):
        self.log: List[Dict[str, Any]] = []
        self.threshold = threshold
        self.is_imbalanced = False
        self.minority_share: float = None
        self.minority_class = None
        self.class_distribution: Dict = {}
        self.recommendation: str = None
        self.smote_outcome: Dict[str, Any] = {"status": "not requested"}

    def _log(self, step, action, reason, status="applied", fitted_on="train"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    def analyze(self, y: pd.Series) -> Dict[str, Any]:
        if y is None or len(y) == 0 or isinstance(y, pd.DataFrame):
            return {"imbalanced": False}
        shares = pd.Series(y).value_counts(normalize=True)
        self.class_distribution = {str(k): float(v) for k, v in shares.items()}
        if len(shares) < 2:
            self._log("Imbalance Check", "Single class in training rows", "Cannot balance", status="skipped")
            return {"imbalanced": False, "distribution": self.class_distribution}
        self.minority_share = float(shares.min())
        self.minority_class = shares.idxmin()
        self.is_imbalanced = self.minority_share < self.threshold
        if self.is_imbalanced:
            self.recommendation = "Use F1 / ROC-AUC, class weights, optionally SMOTE"
            self._log("Imbalance Check", f"Imbalanced: minority class {self.minority_share:.2%}",
                      f"Below {self.threshold:.0%}; leaderboard defaults to F1 and shows the majority baseline")
        else:
            self._log("Imbalance Check", "Classes reasonably balanced",
                      f"Minority class {self.minority_share:.2%} of training rows", status="skipped")
        return {"imbalanced": self.is_imbalanced, "min_ratio": self.minority_share,
                "distribution": self.class_distribution, "recommendation": self.recommendation}

    def apply_smote(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        counts = pd.Series(y).value_counts()
        before = {str(k): int(v) for k, v in counts.items()}
        outcome = {"status": "skipped", "rows_before": int(len(X)), "rows_after": int(len(X)),
                   "class_counts_before": before, "class_counts_after": before}
        if len(counts) < 2:
            outcome["reason"] = "only one class in the training rows"
        elif counts.min() / counts.max() >= 0.8:
            outcome["reason"] = f"classes already balanced (minority/majority = {counts.min() / counts.max():.2f})"
        elif counts.min() < 2:
            outcome["reason"] = "the minority class has fewer than 2 training rows"
        else:
            try:
                from imblearn.over_sampling import SMOTE
            except ImportError:
                outcome.update(status="failed", reason="imbalanced-learn is not installed")
            else:
                try:
                    k = int(min(5, counts.min() - 1))
                    Xr, yr = SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)
                    Xr = pd.DataFrame(Xr, columns=X.columns)
                    yr = pd.Series(yr, name=getattr(y, "name", None))
                    after = {str(k2): int(v) for k2, v in yr.value_counts().items()}
                    outcome.update(status="applied", rows_after=int(len(Xr)), class_counts_after=after,
                                   reason=f"SMOTE with k_neighbors={k} on the training rows only")
                    self.smote_outcome = outcome
                    self._log("SMOTE", f"Resampled training rows {len(X):,} -> {len(Xr):,}", outcome["reason"])
                    return Xr, yr
                except Exception as exc:  # recorded and shown, never hidden
                    logger.exception("SMOTE failed")
                    outcome.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        self.smote_outcome = outcome
        self._log("SMOTE", f"SMOTE {outcome['status']}", outcome["reason"],
                  status="skipped" if outcome["status"] != "applied" else "applied")
        return X, y

    def get_class_weights(self):
        return "balanced" if self.is_imbalanced else None
