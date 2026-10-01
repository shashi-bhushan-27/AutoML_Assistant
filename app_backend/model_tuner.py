"""
Optuna hyperparameter tuning on the training split only.

* The search space per model is declared once in ``SEARCH_SPACES``; the UI shows
  inputs only for these parameters and ``param_ranges`` overrides their bounds
  (the old UI offered range inputs that the tuner ignored).
* Cross-validation: StratifiedKFold for classification, KFold for regression,
  TimeSeriesSplit for time series.
* Scoring: RMSE for regression; for classification F1 of the minority class when it is
  under 10% of the training rows, otherwise accuracy.
* The tuned model is refit on the full training split and evaluated on the same test
  split as the untuned model, so before/after numbers are comparable.
"""
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score

from app_backend import model_registry
from app_backend.leaderboard import choose_primary_metric, minority_share
from app_backend.task_types import is_regression

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# name -> {param: ("int"|"float"|"logfloat"|"cat", low, high | choices)}
SEARCH_SPACES: Dict[str, Dict[str, Tuple]] = {
    "XGBoost": {"n_estimators": ("int", 100, 1000), "learning_rate": ("logfloat", 0.01, 0.3),
                "max_depth": ("int", 3, 12), "subsample": ("float", 0.5, 1.0),
                "colsample_bytree": ("float", 0.5, 1.0)},
    "Random Forest": {"n_estimators": ("int", 50, 500), "max_depth": ("int", 5, 30),
                      "min_samples_split": ("int", 2, 10), "min_samples_leaf": ("int", 1, 5)},
    "Extra Trees": {"n_estimators": ("int", 50, 500), "max_depth": ("int", 5, 30),
                    "min_samples_split": ("int", 2, 10), "min_samples_leaf": ("int", 1, 5)},
    "Gradient Boosting": {"n_estimators": ("int", 50, 500), "learning_rate": ("logfloat", 0.01, 0.2),
                          "max_depth": ("int", 3, 8), "subsample": ("float", 0.6, 1.0)},
    "Hist Gradient Boosting": {"max_iter": ("int", 100, 1000), "learning_rate": ("logfloat", 0.01, 0.3),
                               "max_leaf_nodes": ("int", 15, 255), "l2_regularization": ("float", 0.0, 1.0)},
    "AdaBoost": {"n_estimators": ("int", 50, 500), "learning_rate": ("logfloat", 0.01, 2.0)},
    "Decision Tree": {"max_depth": ("int", 2, 30), "min_samples_split": ("int", 2, 20),
                      "min_samples_leaf": ("int", 1, 10)},
    "SVM": {"C": ("logfloat", 0.1, 100.0), "kernel": ("cat", ["linear", "rbf"]), "gamma": ("cat", ["scale", "auto"])},
    "KNN": {"n_neighbors": ("int", 3, 30), "weights": ("cat", ["uniform", "distance"]), "p": ("int", 1, 2)},
    "Ridge": {"alpha": ("logfloat", 0.01, 100.0)},
    "Lasso": {"alpha": ("logfloat", 0.0001, 10.0)},
    "ElasticNet": {"alpha": ("logfloat", 0.001, 10.0), "l1_ratio": ("float", 0.1, 0.9)},
    "Logistic Regression": {"C": ("logfloat", 0.01, 100.0)},
}


def search_space(model_name: str) -> Dict[str, Tuple]:
    base = model_name.replace(" (Tuned)", "")
    return dict(SEARCH_SPACES.get(base, {}))


class ModelTuner:
    def __init__(self, trainer):
        self.trainer = trainer
        self.X_train = trainer.X_train
        self.y_train = trainer.y_train
        self.task_type = trainer.task_type
        self.is_time_series = trainer.is_time_series
        self.random_state = getattr(trainer, "random_state", 42)

    def scoring(self) -> Dict[str, Any]:
        choice = choose_primary_metric(self.task_type, minority_share(self.y_train))
        if choice["metric"] == "RMSE":
            return {**choice, "scorer": "neg_root_mean_squared_error"}
        if choice["metric"] == "F1 Score":
            pos = self.trainer.pos_label
            return {**choice, "scorer": make_scorer(f1_score, pos_label=pos, zero_division=0)}
        return {**choice, "scorer": "accuracy"}

    def _cv(self, folds: int):
        if self.is_time_series:
            return TimeSeriesSplit(n_splits=folds)
        if is_regression(self.task_type):
            return KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)

    @staticmethod
    def _suggest(trial, space: Dict[str, Tuple], ranges: Dict[str, Tuple]) -> Dict[str, Any]:
        params = {}
        for name, spec in space.items():
            kind = spec[0]
            if kind == "cat":
                choices = list(ranges.get(name) or spec[1])
                params[name] = trial.suggest_categorical(name, choices)
                continue
            low, high = ranges.get(name, (spec[1], spec[2]))
            if kind == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high), log=(kind == "logfloat"))
        return params

    def tune_model(self, model_name: str, time_budget: float = 60, cv_folds: int = 3,
                   param_ranges: Optional[Dict[str, Tuple]] = None, n_trials: Optional[int] = None,
                   progress_cb: Callable[[int, float, Optional[float]], None] = None, cancel_event=None,
                   custom_params=None) -> Dict[str, Any]:
        """Tune ``model_name``; returns metrics on the test split plus trials, best params and CV score."""
        base_name = model_name.replace(" (Tuned)", "")
        if base_name in model_registry.TIME_SERIES_MODELS + ["LSTM"]:
            return {"Error": f"Tuning is not implemented for {base_name}."}
        space = search_space(base_name)
        if not space:
            return {"Error": f"{base_name} has no tunable hyperparameters in this app."}
        ranges = dict(param_ranges or custom_params or {})
        unknown = sorted(set(ranges) - set(space))
        if unknown:
            return {"Error": f"Parameters not in the search space of {base_name}: {unknown}"}
        for name, bounds in ranges.items():
            if space[name][0] != "cat" and bounds[0] > bounds[1]:
                return {"Error": f"Invalid range for {name}: low > high"}
        scoring = self.scoring()
        cv = self._cv(cv_folds)
        start = time.time()

        def objective(trial):
            params = self._suggest(trial, space, ranges)
            model = model_registry.build_model(base_name, self.task_type, self.random_state, params)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring["scorer"],
                                     n_jobs=-1, error_score="raise")
            return float(np.mean(scores))

        def callback(study, trial):
            if cancel_event is not None and cancel_event.is_set():
                study.stop()
            if progress_cb:
                best = study.best_value if any(t.state == optuna.trial.TrialState.COMPLETE
                                               for t in study.trials) else None
                progress_cb(len(study.trials), time.time() - start, best)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, timeout=time_budget, n_trials=n_trials, callbacks=[callback],
                       catch=(Exception,))
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            return {"Error": "No trial completed within the time budget" +
                             (f" ({len(failed)} failed trials)" if failed else "")}
        best_params = study.best_params
        final = model_registry.build_model(base_name, self.task_type, self.random_state, best_params)
        final.fit(self.X_train, self.y_train)
        tuned_name = f"{base_name} (Tuned)"
        self.trainer.trained_models[tuned_name] = final
        preds = final.predict(self.trainer.X_test)
        proba = final.predict_proba(self.trainer.X_test) if (hasattr(final, "predict_proba")
                                                            and not is_regression(self.task_type)) else None
        self.trainer.predictions[tuned_name] = preds
        if proba is not None:
            self.trainer.prediction_probas[tuned_name] = proba
        res = self.trainer.evaluate(self.trainer.y_test, preds, tuned_name, time.time() - start,
                                    model_obj=final, proba=proba)
        cv_value = study.best_value
        res.update({
            "trials": len(study.trials), "trials_completed": len(complete),
            "Best Params": best_params,
            "cv_metric": scoring["metric"], "cv_folds": cv_folds, "cv_scheme": type(cv).__name__,
            "cv_best_score": -cv_value if scoring["metric"] == "RMSE" else cv_value,
            "param_ranges_used": {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in
                                  {n: ranges.get(n, space[n][1:] if space[n][0] != "cat" else space[n][1])
                                   for n in space}.items()},
            "cancelled": bool(cancel_event is not None and cancel_event.is_set()),
        })
        return res
