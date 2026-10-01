"""
SHAP explanations with explicit routing, a time budget and cancellation.

Routing (by estimator type, not by class-name substrings):
  * tree   - scikit-learn tree ensembles / decision trees and XGBoost -> TreeExplainer
             (if TreeExplainer rejects the model, e.g. AdaBoost, it falls back to kernel)
  * linear - LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression -> LinearExplainer
  * kernel - everything else (SVM, KNN, the voting ensemble) -> KernelExplainer on a
             k-means summary of the training rows, ``nsamples`` scaled to the feature count,
             explained row by row so it can report progress, stop at the time budget and be
             cancelled. A budget overrun returns the rows finished so far with status
             ``budget_exceeded``.

Only bool/object columns are cast (to float64); numeric columns are left alone. The
previous blanket float32 cast made KNN KernelExplainer ~15x slower and multiplied
peak memory. TreeExplainer additivity failures are retried with
``check_additivity=False`` and reported as a warning.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - shap is in requirements.txt
    SHAP_AVAILABLE = False

DEFAULT_TIME_BUDGET_S = 120.0
KMEANS_BACKGROUND = 10


@dataclass
class ShapResult:
    status: str = "not run"            # ok | budget_exceeded | cancelled | error
    explainer: Optional[str] = None
    route: Optional[str] = None
    route_reason: str = ""
    output_explained: str = ""
    rows_requested: int = 0
    rows_done: int = 0
    runtime_s: float = 0.0
    seconds_per_row: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _tree_types():
    from sklearn import ensemble, tree

    names = ["RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
             "GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier",
             "HistGradientBoostingRegressor"]
    types = tuple(getattr(ensemble, n) for n in names if hasattr(ensemble, n))
    return types + (tree.DecisionTreeClassifier, tree.DecisionTreeRegressor)


def _linear_types():
    from sklearn import linear_model as lm

    return (lm.LinearRegression, lm.Ridge, lm.Lasso, lm.ElasticNet, lm.LogisticRegression)


class SHAPExplainer:
    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str = "Regression",
                 feature_names: Optional[list] = None, class_index: Optional[int] = None,
                 class_names: Optional[list] = None):
        self.model = model
        self.X_train = self._ensure_dataframe(X_train, feature_names)
        self.X_test = self._ensure_dataframe(X_test, feature_names)
        self.task_type = task_type
        self.feature_names = list(self.X_train.columns)
        classes = getattr(model, "classes_", None)
        self.n_classes = len(classes) if classes is not None else 0
        self.class_index = class_index if class_index is not None else (1 if self.n_classes == 2 else 0)
        self.class_names = class_names
        self._explainer = None
        self._shap_values: Optional[np.ndarray] = None
        self._expected_value: Optional[float] = None
        self.X_explained: Optional[pd.DataFrame] = None
        self.last_result = ShapResult()

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_dataframe(X, feature_names=None) -> pd.DataFrame:
        """DataFrame with a clean index; only bool/object columns are cast (to float64)."""
        if isinstance(X, pd.DataFrame):
            df = X.reset_index(drop=True)
        elif isinstance(X, np.ndarray):
            cols = feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
        else:
            df = pd.DataFrame(X)
        cast = [c for c in df.columns if df[c].dtype == bool or df[c].dtype == object]
        if cast:
            df = df.copy()
            df[cast] = df[cast].astype(np.float64)
        return df

    def _get_model_type(self) -> str:
        m = self.model
        if isinstance(m, _tree_types()) or type(m).__module__.startswith("xgboost"):
            return "tree"
        if isinstance(m, _linear_types()):
            return "linear"
        return "kernel"

    def route_reason(self) -> str:
        route, cls = self._get_model_type(), type(self.model).__name__
        if cls.startswith("AdaBoost"):
            return (f"{cls}: shap's TreeExplainer does not support AdaBoost, so the model-agnostic "
                    "KernelExplainer is used (slow; k-means background and a time budget)")
        return {"tree": f"{cls} is a tree model: exact TreeExplainer (fast)",
                "linear": f"{cls} is a linear model: LinearExplainer (fast)",
                "kernel": f"{cls} has no model-specific explainer: KernelExplainer (model-agnostic, slow; "
                          "uses a k-means background and a time budget)"}[route]

    def _class_label(self) -> str:
        if self.class_names is not None and self.class_index < len(self.class_names):
            return str(self.class_names[self.class_index])
        classes = getattr(self.model, "classes_", None)
        return str(classes[self.class_index]) if classes is not None else str(self.class_index)

    def _kernel_output(self):
        """Function of a numpy batch -> 1-D output to explain, plus its description."""
        cols = self.feature_names
        m = self.model
        frame = lambda A: pd.DataFrame(np.asarray(A, dtype=float), columns=cols)  # noqa: E731
        if self.n_classes:
            if hasattr(m, "predict_proba") and getattr(m, "voting", "soft") == "soft":
                try:
                    m.predict_proba(self.X_train.head(1))
                    idx = self.class_index
                    return (lambda A: m.predict_proba(frame(A))[:, idx]), f"P(class = {self._class_label()})"
                except (AttributeError, NotImplementedError):
                    pass
            if hasattr(m, "decision_function") and self.n_classes == 2:
                return (lambda A: np.ravel(m.decision_function(frame(A)))), \
                    f"decision function (positive = class {self._class_label()})"
            return (lambda A: np.asarray(m.predict(frame(A)), dtype=float)), "predicted class code"
        # regression; the wrapper also avoids shap touching XGBoost's read-only feature_names_in_
        return (lambda A: np.asarray(m.predict(frame(A)), dtype=float)), "prediction"

    def _nsamples(self) -> int:
        return int(min(2048, 20 * len(self.feature_names) + 100))

    @staticmethod
    def _select_output(raw, idx):
        if isinstance(raw, list):
            return np.asarray(raw[idx if idx < len(raw) else -1])
        raw = np.asarray(raw)
        if raw.ndim == 3:
            return raw[:, :, idx if idx < raw.shape[2] else -1]
        return raw

    @staticmethod
    def _select_expected(ev, idx):
        arr = np.atleast_1d(np.asarray(ev, dtype=float))
        return float(arr[idx] if idx < len(arr) else arr[-1])

    # ── build ────────────────────────────────────────────────────────────────
    def build_explainer(self) -> bool:
        if not SHAP_AVAILABLE:
            self.last_result = ShapResult(status="error", error="shap is not installed")
            return False
        route = self._get_model_type()
        res = ShapResult(route=route, route_reason=self.route_reason())
        try:
            if route == "tree":
                try:
                    self._explainer = shap.TreeExplainer(self.model)
                    boosted = type(self.model).__module__.startswith("xgboost") or "Boosting" in type(self.model).__name__
                    if not self.n_classes:
                        res.output_explained = "prediction"
                    elif boosted:
                        res.output_explained = f"log-odds of class {self._class_label()}"
                    else:
                        res.output_explained = f"P(class = {self._class_label()})"
                except Exception as exc:  # e.g. AdaBoost is not supported by TreeExplainer
                    res.warnings.append(f"TreeExplainer rejected {type(self.model).__name__} "
                                        f"({type(exc).__name__}); using KernelExplainer instead")
                    route = res.route = "kernel"
            if route == "linear":
                background = self.X_train.sample(min(len(self.X_train), 1000), random_state=0)
                self._explainer = shap.LinearExplainer(self.model, background)
                res.output_explained = (f"log-odds of class {self._class_label()}" if self.n_classes
                                        else "prediction")
            if route == "kernel":
                fn, desc = self._kernel_output()
                k = min(KMEANS_BACKGROUND, len(self.X_train))
                background = shap.kmeans(self.X_train.to_numpy(dtype=float), k)
                self._explainer = shap.KernelExplainer(fn, background)
                res.output_explained = desc
            res.explainer = type(self._explainer).__name__
            self.last_result = res
            return True
        except Exception as exc:
            logger.exception("SHAP explainer build failed")
            res.status, res.error = "error", f"{type(exc).__name__}: {exc}"
            self.last_result = res
            return False

    def estimate_seconds_per_row(self, probe_rows: int = 5) -> Optional[float]:
        """Cheap runtime estimate before starting.

        Kernel route: KernelExplainer evaluates the model on ``nsamples x background`` synthetic rows per
        explained row, so one model call on 500 rows is timed and scaled. Tree/linear routes: a few rows
        are explained directly.
        """
        if self._explainer is None and not self.build_explainer():
            return None
        route = self.last_result.route
        if route == "kernel":
            fn, _ = self._kernel_output()
            probe = self.X_train.sample(min(500, len(self.X_train)), replace=len(self.X_train) < 500,
                                        random_state=0).to_numpy(dtype=float)
            t0 = time.perf_counter()
            fn(probe)
            per_eval = (time.perf_counter() - t0) / len(probe)
            return per_eval * self._nsamples() * min(KMEANS_BACKGROUND, len(self.X_train))
        sample = self.X_test.head(probe_rows)
        t0 = time.perf_counter()
        try:
            self._explainer.shap_values(sample, **({"check_additivity": False} if route == "tree" else {}))
        except Exception:  # the real run reports errors; the estimate just gives up
            return None
        return (time.perf_counter() - t0) / max(len(sample), 1)

    estimate_kernel_seconds_per_row = estimate_seconds_per_row

    # ── compute ──────────────────────────────────────────────────────────────
    def explain(self, max_rows: int = 200, time_budget_s: float = DEFAULT_TIME_BUDGET_S,
                progress_cb: Callable[[int, int], None] = None, cancel_event=None) -> ShapResult:
        if self._explainer is None and not self.build_explainer():
            return self.last_result
        res = self.last_result
        X_sample = self.X_test.head(max_rows)
        res.rows_requested = len(X_sample)
        t0 = time.perf_counter()
        try:
            if res.route in ("tree", "linear"):
                # explained in chunks so progress, the time budget and cancellation work here too
                chunks, done, res.status = [], 0, "running"
                check = True
                for start in range(0, len(X_sample), 10):
                    if cancel_event is not None and cancel_event.is_set():
                        res.status = "cancelled"
                        break
                    if time_budget_s and time.perf_counter() - t0 > time_budget_s:
                        res.status = "budget_exceeded"
                        break
                    part = X_sample.iloc[start:start + 10]
                    if res.route == "tree":
                        try:
                            raw = self._explainer.shap_values(part, check_additivity=check)
                        except Exception as exc:
                            if "additiv" not in str(exc).lower():
                                raise
                            if check:
                                res.warnings.append(
                                    "TreeExplainer additivity check failed (SHAP values do not sum to the model "
                                    "output); recomputed with check_additivity=False - treat these values with "
                                    "caution")
                            check = False
                            raw = self._explainer.shap_values(part, check_additivity=False)
                    else:
                        raw = self._explainer.shap_values(part)
                    chunks.append(self._select_output(raw, self.class_index).reshape(len(part), -1))
                    done += len(part)
                    if res.seconds_per_row is None:
                        res.seconds_per_row = (time.perf_counter() - t0) / len(part)
                    if progress_cb:
                        progress_cb(done, len(X_sample))
                values = np.vstack(chunks) if chunks else None
                expected = self._select_expected(self._explainer.expected_value, self.class_index)
            else:
                rows, done = [], 0
                arr = X_sample.to_numpy(dtype=float)
                res.status = "running"
                for i in range(len(arr)):
                    if cancel_event is not None and cancel_event.is_set():
                        res.status = "cancelled"
                        break
                    if time_budget_s and time.perf_counter() - t0 > time_budget_s:
                        res.status = "budget_exceeded"
                        break
                    r0 = time.perf_counter()
                    rows.append(np.asarray(self._explainer.shap_values(arr[i:i + 1], nsamples=self._nsamples(),
                                                                       silent=True, l1_reg=False)).reshape(1, -1))
                    done += 1
                    if res.seconds_per_row is None:
                        res.seconds_per_row = time.perf_counter() - r0
                    if progress_cb:
                        progress_cb(done, len(arr))
                values = np.vstack(rows) if rows else None
                expected = self._select_expected(self._explainer.expected_value, 0)
            res.rows_done = done
            res.runtime_s = time.perf_counter() - t0
            if values is None or done == 0:
                if res.status not in ("cancelled", "budget_exceeded"):
                    res.status = "error"
                    res.error = "no rows explained"
                self.last_result = res
                return res
            values = np.asarray(values, dtype=float)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            self._shap_values = values
            self._expected_value = expected
            self.X_explained = X_sample.head(done)
            if res.status in ("not run", "running"):
                res.status = "ok"
            if not np.isfinite(values).all():
                res.warnings.append("some SHAP values are not finite")
        except Exception as exc:
            logger.exception("SHAP value computation failed")
            res.status, res.error = "error", f"{type(exc).__name__}: {exc}"
            res.runtime_s = time.perf_counter() - t0
        self.last_result = res
        return res

    def compute_shap_values(self, max_rows: int = 200, time_budget_s: float = DEFAULT_TIME_BUDGET_S,
                            progress_cb=None, cancel_event=None) -> Optional[np.ndarray]:
        """Backward-compatible wrapper: the SHAP matrix (possibly partial, see ``last_result.status``) or None."""
        if self._shap_values is not None:
            return self._shap_values
        self.explain(max_rows, time_budget_s, progress_cb, cancel_event)
        return self._shap_values

    # ── chart data (all use the rows actually explained) ─────────────────────
    def get_feature_importance_df(self, top_n: int = 15) -> Optional[pd.DataFrame]:
        v = self._shap_values
        if v is None:
            return None
        return (pd.DataFrame({"Feature": self.feature_names[:v.shape[1]], "SHAP Importance": np.abs(v).mean(axis=0)})
                .sort_values("SHAP Importance", ascending=False).head(top_n).reset_index(drop=True))

    def get_beeswarm_data(self, top_n: int = 12) -> Optional[pd.DataFrame]:
        v = self._shap_values
        if v is None:
            return None
        order = np.argsort(np.abs(v).mean(axis=0))[::-1][:top_n]
        rows = []
        for idx in order:
            feat = self.feature_names[idx]
            fv = self.X_explained.iloc[:, idx].to_numpy(dtype=float)
            lo, hi = np.nanmin(fv), np.nanmax(fv)
            norm = (fv - lo) / (hi - lo) if hi > lo else np.zeros_like(fv)
            for sv, x, n in zip(v[:, idx], fv, norm):
                rows.append({"Feature": feat, "SHAP Value": float(sv), "Feature Value": float(x),
                             "Feature Value (norm)": float(n)})
        return pd.DataFrame(rows)

    def get_waterfall_data(self, row_index: int = 0, top_n: int = 12) -> Optional[Dict[str, Any]]:
        v = self._shap_values
        if v is None:
            return None
        row_index = min(max(row_index, 0), v.shape[0] - 1)
        contrib = v[row_index]
        order = np.argsort(np.abs(contrib))[::-1]
        top, rest = order[:top_n], order[top_n:]
        feats = [self.feature_names[i] for i in top]
        values = [float(contrib[i]) for i in top]
        if len(rest):
            feats.append(f"{len(rest)} other features")
            values.append(float(contrib[rest].sum()))
        row = self.X_explained.iloc[row_index]
        return {"base_value": float(self._expected_value), "features": feats, "shap_contributions": values,
                "feature_values": [float(row.iloc[i]) for i in top],
                "prediction": float(self._expected_value + contrib.sum())}

    def get_dependence_data(self, feature_name: str) -> Optional[pd.DataFrame]:
        v = self._shap_values
        if v is None or feature_name not in self.feature_names:
            return None
        i = self.feature_names.index(feature_name)
        return pd.DataFrame({"Feature Value": self.X_explained.iloc[:, i].to_numpy(dtype=float),
                             "SHAP Value": v[:, i].astype(float)})

    @staticmethod
    def is_available() -> bool:
        return SHAP_AVAILABLE
