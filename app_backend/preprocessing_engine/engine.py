"""
AutoPreprocessor: the single preprocessing pipeline used by the UI, the training
code, the ``/predict`` API and the code export.

Order of operations in ``fit_transform``:

  1. ingestion (all rows, structural only): drop exact duplicate rows and all-empty columns
  2. profile (all rows, descriptive only; used for display and task detection)
  3. drop rows whose target is missing; label-encode classification targets (class vocabulary)
  4. train/test split (stratified for single-target classification, chronological for time series)
  5. fit on the TRAINING rows only, then apply to both partitions:
       types/booleans/dates -> outlier caps -> skew transforms -> imputation
       -> encoding -> constant/correlation filter -> scaling
  6. optional SMOTE on the training rows only

``transform(raw_rows)`` replays step 5 with the fitted state and returns the
columns in training order, so ``transform(raw test rows)`` reproduces the
training-time representation of those rows exactly (tested in
``tests/test_preprocessing.py``). Data are excluded from the pickle.
"""
import logging
import pickle
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app_backend.task_types import detect_task_type, is_classification, normalize_task_type

from .balancer import ImbalanceHandler
from .encoder import SmartEncoder
from .imputer import SmartImputer
from .ingestion import DataIngestor
from .profiler import DatasetProfiler
from .scaler import SmartScaler
from .selector import FeatureSelector
from .splitter import DataSplitter
from .transformer import FeatureTransformer

logger = logging.getLogger(__name__)

PIPELINE_VERSION = 2


class SchemaError(ValueError):
    """Raised when serving rows do not match the training schema (the API maps it to HTTP 422)."""

    def __init__(self, missing: List[str] = None, invalid: Dict[str, int] = None):
        self.missing = list(missing or [])
        self.invalid = dict(invalid or {})
        parts = []
        if self.missing:
            parts.append(f"missing required column(s): {self.missing}")
        if self.invalid:
            parts.append("non-numeric values in numeric column(s): "
                         + ", ".join(f"{c} ({n} rows)" for c, n in self.invalid.items()))
        super().__init__("; ".join(parts) or "schema mismatch")


class AutoPreprocessor:
    """See the module docstring."""

    def __init__(self, target_col=None, task_type: str = "auto", is_time_series: bool = False,
                 date_col: str = None, test_size: float = 0.2, apply_smote: bool = False,
                 random_state: int = 42, verbose: bool = True, cap_outliers: bool = True,
                 drop_duplicates: bool = True):
        if isinstance(target_col, str):
            self.target_cols = [target_col]
        elif isinstance(target_col, (list, tuple)) and len(target_col) > 0:
            self.target_cols = list(target_col)
        else:
            self.target_cols = []
        self.target_col = self.target_cols[0] if self.target_cols else None
        self.is_multi_output = len(self.target_cols) > 1
        tt = normalize_task_type(task_type)
        self.task_type = tt.value if tt else "auto"
        self.is_time_series = is_time_series
        self.date_col = date_col
        self.test_size = test_size
        self.apply_smote = apply_smote
        self.verbose = verbose
        self.pipeline_version = PIPELINE_VERSION

        self.ingestor = DataIngestor(drop_duplicates=drop_duplicates)
        self.profiler = DatasetProfiler()
        self.transformer = FeatureTransformer(cap_outliers=cap_outliers)
        self.imputer = SmartImputer()
        self.encoder = SmartEncoder(random_state=random_state)
        self.selector = FeatureSelector()
        self.scaler = SmartScaler()
        self.balancer = ImbalanceHandler()
        self.splitter = DataSplitter(test_size=test_size, random_state=random_state)

        self.profile: Dict = {}
        self.log: List[Dict] = []
        self.full_log: List[Dict] = []
        self.is_fitted = False

        # fitted state
        self.input_columns_: List[str] = []
        self.required_columns_: List[str] = []
        self.input_schema_: Dict[str, Dict[str, Any]] = {}
        self.feature_names_: List[str] = []
        self.feature_origin_: Dict[str, str] = {}
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.classes_: Optional[list] = None
        self.target_kind_: Optional[str] = None
        self.n_rows_input_ = self.n_rows_used_ = self.n_train_ = self.n_test_ = 0
        self.rows_dropped_missing_target_ = 0

        # data (kept in memory for the current session, never pickled)
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train = None
        self.y_test = None

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def random_state(self) -> int:
        return self.splitter.random_state

    def _print(self, msg: str):
        if self.verbose:
            logger.info("[AutoPreprocessor] %s", msg)

    def _log(self, step, action, reason, status="applied", fitted_on="all rows"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    def _collect_logs(self):
        self.full_log = []
        for part in (self.ingestor.log, self.profiler.log, self.log, self.splitter.log, self.transformer.log,
                     self.imputer.log, self.encoder.log, self.selector.log, self.scaler.log, self.balancer.log):
            self.full_log.extend(part)

    def _fit_target(self, y_raw):
        """Label-encode classification targets (class vocabulary from all rows); coerce regression targets."""
        cols = [y_raw] if isinstance(y_raw, pd.Series) else [y_raw[c] for c in y_raw.columns]
        out = []
        for s in cols:
            if is_classification(self.task_type):
                values = s if pd.api.types.is_numeric_dtype(s) else s.astype(str)
                le = LabelEncoder().fit(values)
                self.label_encoders_[s.name] = le
                out.append(pd.Series(le.transform(values), index=s.index, name=s.name))
                self._log("Target Encoding", f"Label-encoded target '{s.name}'",
                          f"Classes {list(le.classes_)[:10]} -> 0..{len(le.classes_) - 1}; predictions are "
                          "decoded back to these labels", fitted_on="all rows (class vocabulary only)")
            else:
                num = pd.to_numeric(s, errors="coerce")
                if num.isna().any():
                    raise ValueError(f"Target '{s.name}' has non-numeric values; it cannot be used for regression.")
                out.append(num.astype(float))
        if len(out) == 1:
            le = self.label_encoders_.get(out[0].name)
            self.classes_ = list(le.classes_) if le is not None else None
            return out[0]
        return pd.concat(out, axis=1)

    def _target_kind(self, y) -> Optional[str]:
        if y is None or isinstance(y, pd.DataFrame):
            return None
        if not is_classification(self.task_type):
            return "continuous"
        return "binary" if pd.Series(y).nunique() == 2 else "multiclass"

    def decode_target(self, y_encoded, column: str = None) -> np.ndarray:
        """Map encoded class codes back to the original labels (identity for regression)."""
        column = column or self.target_col
        le = self.label_encoders_.get(column)
        arr = np.asarray(y_encoded)
        if le is None:
            return arr
        return le.inverse_transform(arr.astype(int))

    def _assemble(self, cols: Dict[str, np.ndarray], index) -> pd.DataFrame:
        """Select the training features in training order, scale them and wrap them in a DataFrame."""
        M = np.column_stack([np.asarray(cols[f], dtype=float) for f in self.feature_names_]) \
            if self.feature_names_ else np.empty((len(index), 0))
        pos = {f: i for i, f in enumerate(self.feature_names_)}
        self.scaler.apply(M, [pos[c] for c in self.scaler.columns])
        return pd.DataFrame(M, index=index, columns=self.feature_names_)

    def _transform_steps(self, X: pd.DataFrame, warn: List[str] = None) -> pd.DataFrame:
        """The single code path that turns raw rows into model features (training and serving)."""
        needed = self.required_columns_ or self.input_columns_
        cols = {c: X[c].to_numpy() for c in needed}
        cols = self.transformer.transform(cols, warn)
        cols = self.imputer.transform(cols, warn)
        cols = self.encoder.transform(cols, warn)
        return self._assemble(cols, X.index)

    def _build_schema(self, X_train_raw: pd.DataFrame):
        t_origin = self.transformer.feature_origin
        e_origin = self.encoder.feature_origin
        self.feature_origin_ = {f: t_origin.get(e_origin.get(f, f), e_origin.get(f, f)) for f in self.feature_names_}
        used = set(self.feature_origin_.values())
        self.required_columns_ = [c for c in self.input_columns_ if c in used]
        schema = {}
        for col in self.input_columns_:
            s = X_train_raw[col]
            if col in self.transformer.bool_maps:
                kind = "boolean"
            elif col in self.transformer.datetime_cols:
                kind = "datetime"
            elif col in self.transformer.numeric_converted or pd.api.types.is_numeric_dtype(s):
                kind = "numeric"
            else:
                kind = "categorical"
            non_null = s.dropna()
            example = non_null.iloc[0] if len(non_null) else None
            if isinstance(example, np.generic):
                example = example.item()
            entry = {"kind": kind, "dtype": str(s.dtype), "required": col in used, "example": example,
                     "nullable": True}
            enc = self.encoder.encoders.get(col)
            if kind in ("categorical", "boolean"):
                if enc and "categories" in enc:
                    entry["categories"] = [str(c) for c in enc["categories"]][:50]
                else:
                    entry["categories"] = [str(c) for c in non_null.astype(str).value_counts().index[:50]]
            if kind == "numeric" and len(non_null):
                num = pd.to_numeric(non_null, errors="coerce")
                entry["min"], entry["max"] = float(num.min()), float(num.max())
            schema[col] = entry
        self.input_schema_ = schema

    # ── public API ───────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame = None, file_path: str = None) -> Dict[str, Any]:
        self._print("Starting preprocessing pipeline")
        self.encoder.random_state = self.random_state
        df, _ = self.ingestor.ingest(file_path=file_path, df=df)
        self.n_rows_input_ = len(df) + self.ingestor.duplicates_removed
        self.profile = self.profiler.generate_profile(df, target_col=self.target_col)

        valid_targets = [c for c in self.target_cols if c in df.columns]
        if self.target_cols and not valid_targets:
            raise ValueError(f"Target column(s) {self.target_cols} not found in the data.")
        if valid_targets:
            missing_y = df[valid_targets].isna().any(axis=1)
            if missing_y.any():
                self.rows_dropped_missing_target_ = int(missing_y.sum())
                df = df.loc[~missing_y]
                self._log("Missing Target", f"Dropped {self.rows_dropped_missing_target_:,} rows",
                          "Rows without a target value cannot be used for training or evaluation")
            if self.task_type == "auto":
                self.task_type = detect_task_type(df[valid_targets[0]]).value
            self.is_multi_output = len(valid_targets) > 1
            y_raw = df[valid_targets[0]] if len(valid_targets) == 1 else df[valid_targets]
            X = df.drop(columns=valid_targets)
            y = self._fit_target(y_raw)
        else:
            X, y = df, None

        if self.is_time_series and self.date_col and self.date_col in X.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                order = pd.to_datetime(X[self.date_col], errors="coerce").argsort(kind="stable").values
            X = X.iloc[order]
            y = y.iloc[order] if y is not None else None
            self._log("Time Ordering", f"Sorted rows by '{self.date_col}'",
                      "Chronological split: train on the past, test on the future")

        self.input_columns_ = list(X.columns)
        self.n_rows_used_ = len(X)

        if y is not None:
            X_tr, X_te, y_tr, y_te = self.splitter.split(X, y, task_type=self.task_type,
                                                         is_time_series=self.is_time_series,
                                                         date_col=self.date_col)
        else:
            X_tr, X_te, y_tr, y_te = X, None, None, None
            self.splitter._log("Data Split", "No split", "No target column; all rows are used to fit",
                               status="skipped")

        # ---- fit on the training partition only ----
        self._print("Fitting transforms on the training rows")
        cols = self.transformer.fit({c: X_tr[c].to_numpy() for c in X_tr.columns})
        cols = self.imputer.fit(cols).transform(cols)
        self.target_kind_ = self._target_kind(y_tr)
        cols, te_train = self.encoder.fit(cols, y=y_tr if self.target_kind_ in ("binary", "continuous") else None,
                                          target_kind=self.target_kind_)
        cols.update(te_train)  # training rows get cross-fitted target encodings
        self.selector.fit(cols)
        self.feature_names_ = list(self.selector.selected_features)
        self.scaler.fit({f: cols[f] for f in self.feature_names_})
        self._build_schema(X_tr)
        X_train = self._assemble(cols, X_tr.index)
        X_test = self._transform_steps(X_te) if X_te is not None else None

        if is_classification(self.task_type) and y_tr is not None and not isinstance(y_tr, pd.DataFrame):
            self.balancer.analyze(y_tr)
            if self.apply_smote:
                X_train, y_tr = self.balancer.apply_smote(X_train, y_tr, random_state=self.random_state)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_tr, y_te
        self.n_train_ = len(X_train)
        self.n_test_ = len(X_test) if X_test is not None else 0
        self._collect_logs()
        self.is_fitted = True
        self._print("Preprocessing complete")
        return {"X_train": self.X_train, "X_test": self.X_test, "y_train": self.y_train, "y_test": self.y_test,
                "profile": self.profile, "logs": self.full_log}

    def validate_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Schema check for serving rows: missing required columns, non-numeric values, extra columns."""
        missing = [c for c in self.required_columns_ if c not in df.columns]
        invalid = {}
        for col in self.required_columns_:
            # columns that already arrive with a numeric dtype are valid by construction; only text needs parsing
            if col in df.columns and df[col].dtype.kind not in "iufb" \
                    and self.input_schema_.get(col, {}).get("kind") == "numeric":
                s = df[col]
                bad = s.notna() & pd.to_numeric(s, errors="coerce").isna()
                if bad.any():
                    invalid[col] = int(bad.sum())
        extra = [c for c in df.columns if c not in self.input_columns_ and c not in self.target_cols]
        return {"missing": missing, "invalid": invalid, "extra": extra}

    def transform(self, df: pd.DataFrame, return_warnings: bool = False):
        """Replay the fitted pipeline on raw rows. Raises SchemaError for missing/invalid required columns."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        if getattr(self, "pipeline_version", 1) != PIPELINE_VERSION:
            raise ValueError("This pipeline was saved by an older version of the app; re-run the Prepare step.")
        warn: List[str] = []
        df = df.drop(columns=[c for c in self.target_cols if c in df.columns])
        check = self.validate_input(df)
        if check["missing"] or check["invalid"]:
            raise SchemaError(missing=check["missing"], invalid=check["invalid"])
        if check["extra"]:
            warn.append(f"Ignored {len(check['extra'])} column(s) the model does not use: {check['extra'][:10]}")
        Xt = self._transform_steps(df, warn)  # only the required columns are read
        return (Xt, warn) if return_warnings else Xt

    def parity_check(self, raw_rows: pd.DataFrame, X_ref: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
        """Per-feature comparison of transform(raw_rows) with the training-time representation X_ref."""
        X_serv = self.transform(raw_rows)
        X_ref = X_ref.loc[raw_rows.index]
        rows = []
        for col in self.feature_names_:
            a = X_serv[col].to_numpy(dtype=float)
            b = X_ref[col].to_numpy(dtype=float) if col in X_ref.columns else np.full(len(a), np.nan)
            close = np.isclose(a, b, rtol=tol, atol=tol, equal_nan=True)
            diff = np.abs(a - b)
            rows.append({"feature": col, "source_column": self.feature_origin_.get(col, col),
                         "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
                         "rows_differing": int((~close).sum()), "passed": bool(close.all())})
        return pd.DataFrame(rows)

    # ── reporting ────────────────────────────────────────────────────────────
    def get_report(self) -> Dict[str, Any]:
        applied = [e for e in self.full_log if e.get("status") == "applied"]
        skipped = [e for e in self.full_log if e.get("status") == "skipped"]
        return {
            "summary": {
                "total_steps": len(self.full_log), "applied": len(applied), "skipped": len(skipped),
                "task_type": self.task_type, "features_final": len(self.feature_names_),
                "train_samples": self.n_train_, "test_samples": self.n_test_,
                "rows_input": self.n_rows_input_, "duplicates_removed": self.ingestor.duplicates_removed,
                "rows_dropped_missing_target": self.rows_dropped_missing_target_,
                "split_strategy": self.splitter.strategy, "split_seed": self.random_state,
                "test_size": self.test_size,
            },
            "applied_steps": applied,
            "skipped_steps": skipped,
            "steps": self.full_log,
            "capping_skipped": dict(self.transformer.capping_skipped),
            "dropped_features": {
                "constant_in_train": list(self.selector.dropped_variance),
                "high_correlation": {c: self.selector.correlated_with.get(c) for c in self.selector.dropped_correlation},
                "mostly_missing": list(self.imputer.dropped_columns),
                "empty": list(self.ingestor.empty_columns_dropped),
            },
            "smote": self.balancer.smote_outcome,
            "imbalance": {"minority_share": self.balancer.minority_share,
                          "is_imbalanced": self.balancer.is_imbalanced},
            "classes": [str(c) for c in self.classes_] if self.classes_ is not None else None,
            "profile": self.profile,
            "class_weights": self.balancer.get_class_weights(),
        }

    def get_markdown_report(self) -> str:
        r = self.get_report()
        lines = ["## Preprocessing summary", f"- Task type: {r['summary']['task_type']}",
                 f"- Final features: {r['summary']['features_final']}",
                 f"- Train / test rows: {r['summary']['train_samples']} / {r['summary']['test_samples']}", "",
                 "### Steps"]
        for s in r["steps"]:
            lines.append(f"- **{s['step']}** ({s.get('status')}, fitted on {s.get('fitted_on', 'n/a')}): "
                         f"{s['action']} — {s['reason']}")
        return "\n".join(lines)

    # ── persistence ──────────────────────────────────────────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("X_train", "X_test", "y_train", "y_test"):
            state[key] = None
        return state

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> "AutoPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)

    def export_data(self, train_path: str, test_path: str = None):
        if self.X_train is not None:
            train_df = self.X_train.copy()
            if self.y_train is not None:
                train_df[self.target_col] = np.asarray(self.y_train)
            train_df.to_csv(train_path, index=False)
        if test_path and self.X_test is not None:
            test_df = self.X_test.copy()
            if self.y_test is not None:
                test_df[self.target_col] = np.asarray(self.y_test)
            test_df.to_csv(test_path, index=False)
