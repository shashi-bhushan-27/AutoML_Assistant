"""
Transformer Module
Type correction, boolean normalisation, datetime features, outlier capping and
skewness correction as a *fitted* transformer.

``fit`` learns everything from the training partition (which columns are boolean,
numeric-as-text or dates, the IQR caps and the per-column skew transform);
``transform`` replays exactly those decisions on new rows.

Data move through the pipeline as ``{column: numpy array}`` so that one code path
produces both the training representation and the serving representation.
"""
import logging
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BOOL_PAIRS = [({"yes"}, {"no"}), ({"true"}, {"false"}), ({"t"}, {"f"}), ({"y"}, {"n"})]
DATETIME_PARTS = ("year", "month", "day", "dayofweek", "hour")
Columns = Dict[str, np.ndarray]


def _as_key(value) -> str:
    return str(value).strip().lower()


def _parse_datetime(values) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(pd.Series(values), errors="coerce")


def _n_distinct(a: np.ndarray) -> int:
    a = a[~np.isnan(a)]
    return int(np.unique(a).size) if a.size else 0


class FeatureTransformer:
    """Fitted type/outlier/skew transformer (see module docstring)."""

    def __init__(self, cap_outliers: bool = True, fix_skew: bool = True, skew_threshold: float = 1.0,
                 iqr_k: float = 1.5, zero_share_skip: float = 0.5):
        self.log: List[Dict[str, Any]] = []
        self.cap_outliers = cap_outliers
        self.fix_skew = fix_skew
        self.skew_threshold = skew_threshold
        self.iqr_k = iqr_k
        self.zero_share_skip = zero_share_skip

        self.bool_maps: Dict[str, Dict[str, int]] = {}
        self.numeric_input_cols: List[str] = []
        self.numeric_converted: List[str] = []
        self.categorical_cols: List[str] = []
        self.datetime_cols: List[str] = []
        self.outlier_caps: Dict[str, Tuple[float, float]] = {}
        self.capping_skipped: Dict[str, str] = {}
        self.transforms: Dict[str, str] = {}
        self.output_columns: List[str] = []
        self.feature_origin: Dict[str, str] = {}
        self.is_fitted = False

    def _log(self, step, action, reason, status="applied", fitted_on="train"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    # ── types ────────────────────────────────────────────────────────────────
    def _learn_types(self, cols: Columns):
        for col, a in cols.items():
            if a.dtype == bool:
                self.bool_maps[col] = {"true": 1, "false": 0}
                continue
            if a.dtype.kind in "iuf":
                self.numeric_input_cols.append(col)
                continue
            non_null = a[~pd.isna(a)]
            keys = set(_as_key(v) for v in pd.unique(non_null))
            if len(keys) == 2:
                for true_set, false_set in _BOOL_PAIRS:
                    if keys == true_set | false_set:
                        self.bool_maps[col] = {next(iter(true_set)): 1, next(iter(false_set)): 0}
                        break
            if col in self.bool_maps:
                self._log("Boolean Normalize", f"Converted: {col}", f"Values {sorted(keys)} mapped to 1/0")
                continue
            if len(non_null) == 0:
                self.categorical_cols.append(col)
                continue
            converted = pd.to_numeric(pd.Series(non_null), errors="coerce")
            if converted.notna().mean() > 0.9:
                self.numeric_converted.append(col)
                self._log("Type Correction", f"Converted to numeric: {col}",
                          f"{converted.notna().mean():.0%} of non-null values are numbers")
                continue
            sample = pd.Series(non_null[:200]).astype(str)
            if _parse_datetime(sample).notna().mean() > 0.8 and _parse_datetime(non_null).notna().mean() > 0.8:
                self.datetime_cols.append(col)
                self._log("DateTime Features", f"Extracted from: {col}", "Created " + ", ".join(DATETIME_PARTS))
                continue
            self.categorical_cols.append(col)

    def _apply_types(self, cols: Columns, warn: List[str] = None) -> Columns:
        out: Columns = {}
        numeric = set(self.numeric_input_cols) | set(self.numeric_converted)
        for col, a in cols.items():
            if col in self.bool_maps:
                mapping = self.bool_maps[col]
                missing = pd.isna(a)
                vals = np.array([np.nan if m else mapping.get(_as_key(v), np.nan) for v, m in zip(a, missing)],
                                dtype=float)
                unseen = int((~missing & np.isnan(vals)).sum())
                if unseen and warn is not None:
                    warn.append(f"{col}: {unseen} value(s) are not {sorted(mapping)}; treated as missing")
                out[col] = vals
            elif col in self.datetime_cols:
                parsed = _parse_datetime(a)
                for part in DATETIME_PARTS:
                    out[f"{col}_{part}"] = getattr(parsed.dt, part).to_numpy(dtype=float, na_value=np.nan)
            elif col in numeric:
                if a.dtype.kind in "iufb":
                    out[col] = a.astype(float)
                else:
                    conv = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
                    bad = int((~pd.isna(a) & np.isnan(conv)).sum())
                    if bad and warn is not None:
                        warn.append(f"{col}: {bad} non-numeric value(s) treated as missing")
                    out[col] = conv
            else:  # categorical at fit time: keep the raw values for the encoder
                out[col] = a.astype(object)
        return out

    # ── outliers ─────────────────────────────────────────────────────────────
    def _numeric_candidates(self, cols: Columns) -> List[str]:
        return [c for c, a in cols.items() if a.dtype.kind == "f" and _n_distinct(a) > 2]

    def _learn_caps(self, cols: Columns):
        for col in self._numeric_candidates(cols):
            s = cols[col][~np.isnan(cols[col])]
            q1, q3 = np.quantile(s, [0.25, 0.75])
            iqr = q3 - q1
            zero_share = float((s == 0).mean())
            if iqr == 0:
                self.capping_skipped[col] = (f"IQR is 0 ({zero_share:.0%} zeros); capping would turn the "
                                             "column into a constant")
                continue
            if zero_share >= self.zero_share_skip:
                self.capping_skipped[col] = f"mostly zero ({zero_share:.0%} zeros); capping skipped"
                continue
            lower, upper = q1 - self.iqr_k * iqr, q3 + self.iqr_k * iqr
            n_out = int(((s < lower) | (s > upper)).sum())
            if n_out == 0:
                continue
            if np.unique(np.clip(s, lower, upper)).size <= 1:
                self.capping_skipped[col] = "capping would leave a single value; skipped"
                continue
            self.outlier_caps[col] = (float(lower), float(upper))
            self._log("Outlier Handling", f"Capped {col}",
                      f"{n_out} training values outside [{lower:.3g}, {upper:.3g}] (Q1/Q3 ± {self.iqr_k}×IQR)")
        for col, why in self.capping_skipped.items():
            self._log("Outlier Handling", f"Not capped: {col}", why, status="skipped")
        if not self.outlier_caps and not self.capping_skipped:
            self._log("Outlier Handling", "No outliers detected", "All values within Q1/Q3 ± 1.5×IQR",
                      status="skipped")

    def _apply_caps(self, cols: Columns) -> Columns:
        for col, (lo, hi) in self.outlier_caps.items():
            if col in cols:
                cols[col] = np.clip(cols[col], lo, hi)
        return cols

    # ── skew ─────────────────────────────────────────────────────────────────
    def _learn_skew(self, cols: Columns):
        for col in self._numeric_candidates(cols):
            s = cols[col][~np.isnan(cols[col])]
            if len(s) < 3:
                continue
            skewness = pd.Series(s).skew()
            if not np.isfinite(skewness) or abs(skewness) <= self.skew_threshold:
                continue
            if (s > 0).all():
                self.transforms[col] = "log1p"
                self._log("Skewness Fix", f"Log transform: {col}", f"Training skewness {skewness:.2f}")
            elif (s >= 0).all():
                self.transforms[col] = "sqrt"
                self._log("Skewness Fix", f"Sqrt transform: {col}", f"Training skewness {skewness:.2f}, has zeros")

    def _apply_skew(self, cols: Columns, warn: List[str] = None) -> Columns:
        for col, fn in self.transforms.items():
            if col not in cols:
                continue
            a = cols[col]
            neg = int((a < 0).sum())
            if neg and warn is not None:
                warn.append(f"{col}: {neg} negative value(s) clipped to 0 before {fn}")
            a = np.clip(a, 0, None)
            cols[col] = np.log1p(a) if fn == "log1p" else np.sqrt(a)
        return cols

    # ── public ───────────────────────────────────────────────────────────────
    def fit(self, cols: Columns) -> Columns:
        """Learn from the training columns; returns their transformed values."""
        self._learn_types(cols)
        out = self._apply_types(cols)
        if self.cap_outliers:
            self._learn_caps(out)
            out = self._apply_caps(out)
        if self.fix_skew:
            self._learn_skew(out)
            out = self._apply_skew(out)
        self.output_columns = list(out)
        self.feature_origin = {}
        for col in out:
            self.feature_origin[col] = next((d for d in self.datetime_cols if col.startswith(f"{d}_")
                                             and col[len(d) + 1:] in DATETIME_PARTS), col)
        self.is_fitted = True
        return out

    def transform(self, cols: Columns, warn: List[str] = None) -> Columns:
        out = self._apply_types(cols, warn)
        out = self._apply_caps(out)
        return self._apply_skew(out, warn)
