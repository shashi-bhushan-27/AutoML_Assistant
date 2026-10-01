"""
Encoder Module
Categorical encoding with a strategy chosen by cardinality, fitted on the training
partition with a fixed vocabulary.

  * <= 2 levels      binary code 0/1 (unseen level -> -1, reported as a warning)
  * 3-10 levels      one-hot over the training vocabulary; an unseen level gives an
                     all-zero row (reported as a warning). Columns are float 0/1,
                     never bool, so downstream code (SHAP, scalers) sees numbers.
  * 11-100 levels    target encoding (scikit-learn ``TargetEncoder``: smoothed means;
                     the training rows get 5-fold cross-fitted values, every other row
                     the full-training mapping) for regression and binary
                     classification; frequency encoding for multi-class.
  * > 100 levels     frequency encoding (training frequencies; unseen -> 0)

The previous version recomputed ``pd.get_dummies(drop_first=True)`` per request, so a
one-row request produced no dummy columns at all, and target encoding used the full
label vector including the test rows.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
Columns = Dict[str, np.ndarray]
MISSING_TOKEN = "__MISSING__"


def as_str(a: np.ndarray) -> np.ndarray:
    missing = pd.isna(a)
    return np.array([MISSING_TOKEN if m else str(v) for v, m in zip(a, missing)], dtype=object)


def _lookup(values: np.ndarray, mapping: Dict[str, float], default: float) -> Tuple[np.ndarray, int]:
    get = mapping.get
    out = np.array([get(v, np.nan) for v in values], dtype=float)
    unseen = np.isnan(out)
    if unseen.any():
        out[unseen] = default
    return out, int(unseen.sum())


class SmartEncoder:
    def __init__(self, random_state: int = 42):
        self.log: List[Dict[str, Any]] = []
        self.encoders: Dict[str, Dict[str, Any]] = {}
        self.strategies: Dict[str, str] = {}
        self.ohe_columns: List[str] = []
        self.output_columns: List[str] = []
        self.feature_origin: Dict[str, str] = {}
        self.random_state = random_state

    def _log(self, step, action, reason, status="applied", fitted_on="train"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    @staticmethod
    def _select_strategy(n_unique: int) -> str:
        if n_unique <= 2:
            return "label"
        if n_unique <= 10:
            return "onehot"
        if n_unique <= 100:
            return "target"
        return "frequency"

    def fit(self, cols: Columns, y=None, target_kind: Optional[str] = None) -> Tuple[Columns, Columns]:
        """Learn the encodings. Returns (encoded training columns, cross-fitted target-encoded training values).

        ``target_kind`` is "continuous", "binary" or None (multi-class / multi-output / no target), which decides
        whether 11-100-level columns are target- or frequency-encoded.
        """
        out: Columns = {}
        te_train: Columns = {}
        n_cat = 0
        for col, a in cols.items():
            if a.dtype != object:
                out[col] = a.astype(float)
                self.feature_origin[col] = col
                continue
            n_cat += 1
            values = as_str(a)
            cats = sorted(pd.unique(values))
            strategy = self._select_strategy(len(cats))
            note = ""
            if strategy == "target" and (y is None or target_kind not in ("continuous", "binary")):
                strategy, note = "frequency", " (target encoding needs a numeric or binary target)"
            self.strategies[col] = strategy
            if strategy == "label":
                mapping = {c: float(i) for i, c in enumerate(cats)}
                self.encoders[col] = {"type": "label", "mapping": mapping, "categories": cats, "default": -1.0}
                self._log("Encoding", f"Binary encode: {col}", f"{len(cats)} levels: {cats}")
            elif strategy == "onehot":
                names = [f"{col}_{c}" for c in cats]
                self.ohe_columns.extend(names)
                self.encoders[col] = {"type": "onehot", "categories": cats, "columns": names}
                self._log("Encoding", f"One-hot encode: {col}",
                          f"{len(cats)} levels -> {len(names)} indicator columns (fixed training vocabulary)")
            elif strategy == "target":
                from sklearn.preprocessing import TargetEncoder

                te = TargetEncoder(target_type=target_kind, smooth="auto", cv=5, shuffle=True,
                                   random_state=self.random_state)
                te_train[col] = te.fit_transform(values.reshape(-1, 1), np.asarray(y))[:, 0].astype(float)
                mapping = {str(c): float(e) for c, e in zip(te.categories_[0], te.encodings_[0])}
                self.encoders[col] = {"type": "target", "encoder": te, "mapping": mapping,
                                      "default": float(te.target_mean_)}
                self._log("Encoding", f"Target encode: {col}",
                          f"{len(cats)} levels; smoothed training-target means, 5-fold cross-fitted on the "
                          "training rows (other rows use the full-training mapping)")
            else:
                freq = pd.Series(values).value_counts(normalize=True)
                self.encoders[col] = {"type": "frequency", "mapping": {str(k): float(v) for k, v in freq.items()},
                                      "default": 0.0}
                self._log("Encoding", f"Frequency encode: {col}", f"{len(cats)} levels; training frequencies{note}")
            out.update(self._encode(col, values))
        if n_cat == 0:
            self._log("Encoding", "No categorical columns", "All features are numeric", status="skipped")
        for col, info in self.encoders.items():
            for name in info.get("columns", [col]):
                self.feature_origin[name] = col
        self.output_columns = list(out)
        return out, te_train

    def _encode(self, col: str, values: np.ndarray, warn: List[str] = None) -> Columns:
        info = self.encoders[col]
        kind = info["type"]
        if kind == "onehot":
            cats = np.asarray(info["categories"], dtype=object)
            hits = values[:, None] == cats[None, :]
            unseen = int((~hits.any(axis=1)).sum())
            enc = {name: hits[:, j].astype(float) for j, name in enumerate(info["columns"])}
        else:
            vec, unseen = _lookup(values, info["mapping"], info["default"])
            enc = {col: vec}
        if unseen and warn is not None:
            warn.append(f"{col}: {unseen} value(s) not seen in training ({kind} encoding fallback used)")
        return enc

    def transform(self, cols: Columns, warn: List[str] = None) -> Columns:
        out: Columns = {}
        for col, a in cols.items():
            if col in self.encoders:
                out.update(self._encode(col, as_str(a), warn))
            else:
                out[col] = np.asarray(a, dtype=float)
        return out
