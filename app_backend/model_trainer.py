"""
ModelTrainer: trains registry models on the split produced by AutoPreprocessor.

There is exactly one preprocessing path: data must be set with
``set_preprocessed_data`` (the old internal ``AdvancedPreprocessor`` path, which
used a different, non-stratified split, was removed). Every estimator that takes a
``random_state`` gets the workspace seed. Failures are recorded in
``failed_models`` with their error text and returned as rows with an ``Error``
column; nothing is swallowed.
"""
import logging
import time
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from app_backend import model_registry
from app_backend.ensemble import PrefitVotingEnsemble
from app_backend.task_types import TaskType, is_regression, normalize_task_type

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, df, target_col, task_type="Regression", is_time_series=False, date_col=None,
                 random_state: int = 42, class_weight: Optional[str] = None):
        self.df = df
        if isinstance(target_col, (list, tuple)):
            self.target_cols = list(target_col)
        else:
            self.target_cols = [target_col] if target_col else []
        self.target_col = self.target_cols[0] if self.target_cols else None
        self.is_multi_output = len(self.target_cols) > 1
        self.task_type = (normalize_task_type(task_type) or TaskType.REGRESSION).value
        self.is_time_series = is_time_series
        self.date_col = date_col
        self.random_state = random_state
        self.class_weight = class_weight  # None or "balanced" (applied through sample weights)
        self.results: List[Dict] = []
        self.failed_models: List[Dict[str, str]] = []
        self.trained_models: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.prediction_probas: Dict[str, np.ndarray] = {}
        self.prediction_scores: Dict[str, np.ndarray] = {}
        self.notes: Dict[str, str] = {}

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.df_train = self.df_test = None
        self._data_is_set = False

    # ── data ──────────────────────────────────────────────────────────────────
    def set_preprocessed_data(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self._data_is_set = True
        # raw rows for the time-series models (Prophet/ARIMA need the date column and raw target)
        if isinstance(self.df, pd.DataFrame) and not self.df.empty and hasattr(X_test, "index"):
            try:
                self.df_train = self.df.loc[X_train.index]
                self.df_test = self.df.loc[X_test.index]
            except KeyError:  # e.g. SMOTE re-indexed the training rows
                self.df_train = self.df_test = None

    def get_supported_models(self) -> List[str]:
        return model_registry.supported_models(self.task_type, self.is_time_series)

    @property
    def pos_label(self):
        """Positive class for binary metrics: the minority class of the training labels."""
        if is_regression(self.task_type) or self.y_train is None or self.is_multi_output:
            return None
        counts = pd.Series(np.asarray(self.y_train)).value_counts()
        return counts.idxmin() if len(counts) == 2 else None

    # ── evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, y_true, y_pred, model_name, training_time, model_obj=None, proba=None, scores=None):
        from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                     classification_report, cohen_kappa_score, f1_score, matthews_corrcoef,
                                     mean_absolute_error, mean_squared_error, precision_score, r2_score,
                                     recall_score, roc_auc_score)
        from sklearn.preprocessing import label_binarize

        metrics: Dict[str, Any] = {"Model": model_name, "Time (s)": round(float(training_time), 4)}
        y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)

        if self.is_multi_output:
            per_target, agg = {}, {}
            for i, tname in enumerate(self.target_cols):
                yt = y_true_arr[:, i] if y_true_arr.ndim > 1 else y_true_arr
                yp = y_pred_arr[:, i] if y_pred_arr.ndim > 1 else y_pred_arr
                if is_regression(self.task_type):
                    r = {"RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
                         "MAE": float(mean_absolute_error(yt, yp)), "R²": float(r2_score(yt, yp))}
                else:
                    r = {"Accuracy": float(accuracy_score(yt, yp)),
                         "F1 Score": float(f1_score(yt, yp, average="weighted", zero_division=0))}
                per_target[tname] = {k: round(v, 4) for k, v in r.items()}
                for k, v in r.items():
                    agg.setdefault(k, []).append(v)
            for k, vals in agg.items():
                metrics[k] = round(float(np.mean(vals)), 4)
            metrics["Targets"] = len(self.target_cols)
            metrics["Per-Target Metrics"] = per_target
            return metrics

        if is_regression(self.task_type):
            y_pred_arr = y_pred_arr.astype(float)
            metrics["RMSE"] = round(float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))), 4)
            metrics["MAE"] = round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 4)
            metrics["R²"] = round(float(r2_score(y_true_arr, y_pred_arr)), 4)
            non_zero = np.abs(y_true_arr) > 1e-8
            if non_zero.any():
                mape = np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / np.abs(y_true_arr[non_zero])))
                metrics["MAPE (%)"] = round(float(mape * 100), 4)
            return metrics

        classes = np.unique(np.concatenate([np.unique(y_true_arr), np.asarray(self.y_train).ravel()])) \
            if self.y_train is not None else np.unique(y_true_arr)
        is_binary = len(classes) == 2
        if is_binary:
            pos = self.pos_label if self.pos_label is not None else classes[1]
            kw = {"average": "binary", "pos_label": pos}
        else:
            pos, kw = None, {"average": "weighted"}
        metrics["Accuracy"] = round(float(accuracy_score(y_true_arr, y_pred_arr)), 4)
        metrics["Balanced Accuracy"] = round(float(balanced_accuracy_score(y_true_arr, y_pred_arr)), 4)
        metrics["F1 Score"] = round(float(f1_score(y_true_arr, y_pred_arr, zero_division=0, **kw)), 4)
        metrics["Precision"] = round(float(precision_score(y_true_arr, y_pred_arr, zero_division=0, **kw)), 4)
        metrics["Recall"] = round(float(recall_score(y_true_arr, y_pred_arr, zero_division=0, **kw)), 4)
        metrics["MCC"] = round(float(matthews_corrcoef(y_true_arr, y_pred_arr)), 4)
        metrics["Cohen Kappa"] = round(float(cohen_kappa_score(y_true_arr, y_pred_arr)), 4)
        if is_binary:
            metrics["Positive class"] = pos.item() if isinstance(pos, np.generic) else pos
        try:
            if proba is not None:
                model_classes = list(getattr(model_obj, "classes_", classes))
                if is_binary:
                    col = model_classes.index(pos)
                    metrics["AUC-ROC"] = round(float(roc_auc_score(y_true_arr == pos, proba[:, col])), 4)
                    metrics["PR-AUC"] = round(float(average_precision_score(y_true_arr == pos, proba[:, col])), 4)
                else:
                    y_bin = label_binarize(y_true_arr, classes=model_classes)
                    metrics["AUC-ROC"] = round(float(roc_auc_score(y_bin, proba, multi_class="ovr",
                                                                   average="weighted")), 4)
            elif scores is not None and is_binary and np.ndim(scores) == 1:
                # decision_function scores (e.g. SVC without probability=True); higher = classes_[1]
                model_classes = list(getattr(model_obj, "classes_", classes))
                s = scores if model_classes[1] == pos else -scores
                metrics["AUC-ROC"] = round(float(roc_auc_score(y_true_arr == pos, s)), 4)
                metrics["PR-AUC"] = round(float(average_precision_score(y_true_arr == pos, s)), 4)
        except ValueError as exc:  # e.g. a single class in y_true
            metrics["AUC note"] = f"AUC not computed: {exc}"
        report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
        metrics["Per-Class Report"] = {k: v for k, v in report.items()
                                       if k not in ("accuracy", "macro avg", "weighted avg")}
        return metrics

    # ── training ──────────────────────────────────────────────────────────────
    def _fail(self, name: str, error: str, elapsed: float = 0.0) -> Dict[str, Any]:
        self.failed_models.append({"Model": name, "Error": error})
        logger.warning("Model %s failed: %s", name, error)
        return {"Model": name, "Error": error, "Time (s)": round(float(elapsed), 4)}

    def _sample_weight(self, name: str):
        if self.class_weight != "balanced" or is_regression(self.task_type) or self.is_multi_output:
            return None
        if not model_registry.supports_sample_weight(name):
            self.notes[name] = "class weights not supported by this model; trained unweighted"
            return None
        from sklearn.utils.class_weight import compute_sample_weight

        self.notes[name] = "trained with balanced class weights"
        return compute_sample_weight("balanced", np.asarray(self.y_train))

    def train_sklearn_model(self, name, model_class=None, **kwargs):
        from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

        start = time.time()
        try:
            base = model_registry.build_model(name, self.task_type, self.random_state, kwargs) \
                if model_class is None or name in model_registry.MODEL_SPECS else model_class(**kwargs)
            if self.is_multi_output:
                model = MultiOutputRegressor(base) if is_regression(self.task_type) else MultiOutputClassifier(base)
                model.fit(self.X_train, self.y_train)
            else:
                model = base
                sw = self._sample_weight(name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    if sw is not None:
                        model.fit(self.X_train, self.y_train, sample_weight=sw)
                    else:
                        model.fit(self.X_train, self.y_train)
            fit_time = time.time() - start
            preds = model.predict(self.X_test)
            proba = scores = None
            if not self.is_multi_output and not is_regression(self.task_type):
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(self.X_test)  # computed once, reused below
                    except (AttributeError, NotImplementedError):
                        proba = None
                if proba is None and hasattr(model, "decision_function"):
                    scores = model.decision_function(self.X_test)
            self.trained_models[name] = model
            self.predictions[name] = preds
            if proba is not None:
                self.prediction_probas[name] = proba
            if scores is not None:
                self.prediction_scores[name] = scores
            res = self.evaluate(self.y_test, preds, name, fit_time, model_obj=model, proba=proba, scores=scores)
            if name in self.notes:
                res["Notes"] = self.notes[name]
            return res
        except Exception as exc:
            return self._fail(name, f"{type(exc).__name__}: {exc}", time.time() - start)

    def run_selected_models(self, selected_model_names, progress_cb: Callable[[int, int, str], None] = None,
                            cancel_event=None, add_ensemble: bool = False) -> pd.DataFrame:
        """Train the named models one at a time. Returns one row per model (failed ones carry ``Error``)."""
        if not self._data_is_set:
            raise RuntimeError("No training data: run the Prepare step (AutoPreprocessor) first.")
        supported = set(self.get_supported_models())
        results = []
        names = list(selected_model_names)
        for i, name in enumerate(names):
            if cancel_event is not None and cancel_event.is_set():
                results.append(self._fail(str(name), "Cancelled before training started"))
                continue
            if progress_cb:
                progress_cb(i, len(names), str(name))
            if not isinstance(name, str):
                results.append(self._fail(str(name), f"Invalid model name (not a string): {name!r}"))
                continue
            logger.info("Training %s", name)
            if name == "Prophet":
                results.append(self.train_prophet())
            elif name == "ARIMA":
                results.append(self.train_arima())
            elif name == "SARIMAX":
                results.append(self.train_sarimax())
            elif name == "LSTM":
                results.append(self.train_lstm())
            elif name in supported:
                results.append(self.train_sklearn_model(name, model_registry.estimator_class(name, self.task_type)))
            else:
                results.append(self._fail(name, f"Not supported for {self.task_type} tasks"))
        if add_ensemble:
            ens = self.build_ensemble([r["Model"] for r in results if r and not r.get("Error")])
            if ens is not None:
                results.append(ens)
        if progress_cb:
            progress_cb(len(names), len(names), "done")
        self.results = [r for r in results if r is not None]
        return pd.DataFrame(self.results)

    def build_ensemble(self, member_names: List[str]) -> Optional[Dict[str, Any]]:
        members = {n: self.trained_models[n] for n in member_names
                   if n in self.trained_models and n not in model_registry.TIME_SERIES_MODELS + ["LSTM"]}
        if len(members) < 2 or self.is_multi_output:
            return None
        start = time.time()
        try:
            ens = PrefitVotingEnsemble(members, self.task_type)
            preds = ens.predict(self.X_test)
            proba = ens.predict_proba(self.X_test) if ens.voting == "soft" else None
            name = model_registry.ENSEMBLE_NAME
            self.trained_models[name] = ens
            self.predictions[name] = preds
            if proba is not None:
                self.prediction_probas[name] = proba
            res = self.evaluate(self.y_test, preds, name, time.time() - start, model_obj=ens, proba=proba)
            res["Notes"] = ens.describe()
            return res
        except Exception as exc:
            return self._fail(model_registry.ENSEMBLE_NAME, f"{type(exc).__name__}: {exc}")

    # ── time-series models (regression on a chronological split) ──────────────
    def _ts_ready(self, name):
        if not self.is_time_series:
            return self._fail(name, f"{name} is only available for time-series data")
        if self.df_train is None or self.date_col is None:
            return self._fail(name, f"{name} needs the raw rows and a date column")
        return None

    def train_prophet(self):
        err = self._ts_ready("Prophet")
        if err:
            return err
        start = time.time()
        try:
            from prophet import Prophet

            train_df = self.df_train[[self.date_col, self.target_col]].rename(
                columns={self.date_col: "ds", self.target_col: "y"})
            model = Prophet()
            model.fit(train_df)
            future = self.df_test[[self.date_col]].rename(columns={self.date_col: "ds"})
            preds = model.predict(future)["yhat"].values
            self.trained_models["Prophet"] = model
            self.predictions["Prophet"] = preds
            return self.evaluate(self.y_test, preds, "Prophet", time.time() - start)
        except Exception as exc:
            return self._fail("Prophet", f"{type(exc).__name__}: {exc}", time.time() - start)

    def train_arima(self, order=(1, 1, 1)):
        err = self._ts_ready("ARIMA")
        if err:
            return err
        start = time.time()
        try:
            from statsmodels.tsa.arima.model import ARIMA

            fit = ARIMA(np.asarray(self.y_train, dtype=float), order=order).fit()
            preds = fit.forecast(steps=len(self.y_test))
            self.trained_models["ARIMA"] = fit
            self.predictions["ARIMA"] = preds
            return self.evaluate(self.y_test, preds, "ARIMA", time.time() - start)
        except Exception as exc:
            return self._fail("ARIMA", f"{type(exc).__name__}: {exc}", time.time() - start)

    def train_sarimax(self):
        err = self._ts_ready("SARIMAX")
        if err:
            return err
        start = time.time()
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            fit = SARIMAX(np.asarray(self.y_train, dtype=float), order=(1, 1, 1),
                          seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            preds = fit.forecast(steps=len(self.y_test))
            self.trained_models["SARIMAX"] = fit
            self.predictions["SARIMAX"] = preds
            return self.evaluate(self.y_test, preds, "SARIMAX", time.time() - start)
        except Exception as exc:
            return self._fail("SARIMAX", f"{type(exc).__name__}: {exc}", time.time() - start)

    def train_lstm(self):
        """Univariate Keras LSTM (optional; requires tensorflow)."""
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.models import Sequential
        except ImportError:
            return self._fail("LSTM", "TensorFlow is not installed")
        if not self.is_time_series:
            return self._fail("LSTM", "LSTM is only available for time-series data")
        start = time.time()
        try:
            lookback = 10
            y_tr = np.asarray(self.y_train, dtype=np.float32)
            y_te = np.asarray(self.y_test, dtype=np.float32)
            lo, span = y_tr.min(), y_tr.max() - y_tr.min() + 1e-8
            tr_n = (y_tr - lo) / span
            Xs = np.array([tr_n[i - lookback:i] for i in range(lookback, len(tr_n))])[..., None]
            ys = tr_n[lookback:]
            if len(Xs) < 5:
                return self._fail("LSTM", "Not enough training rows for LSTM")
            model = Sequential([LSTM(64, return_sequences=True, input_shape=(lookback, 1)), Dropout(0.2),
                                LSTM(32), Dropout(0.2), Dense(1)])
            model.compile(optimizer="adam", loss="mse")
            model.fit(Xs, ys, epochs=50, batch_size=16, verbose=0,
                      callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)])
            history, out = list(tr_n[-lookback:]), []
            for _ in range(len(y_te)):
                nxt = float(model.predict(np.array(history[-lookback:], dtype=np.float32)[None, :, None],
                                          verbose=0)[0, 0])
                out.append(nxt)
                history.append(nxt)
            preds = np.array(out) * span + lo
            self.trained_models["LSTM"] = model
            self.predictions["LSTM"] = preds
            return self.evaluate(y_te, preds, "LSTM", time.time() - start)
        except Exception as exc:
            return self._fail("LSTM", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}",
                              time.time() - start)
