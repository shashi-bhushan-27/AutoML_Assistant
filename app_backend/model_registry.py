"""
Single registry of the models the app can train. Used by the trainer, the tuner,
the name matcher, the code export and the UI, so every part agrees on which
names exist and which estimator class each one means.
"""
import importlib
from typing import Any, Dict, List, Optional

from app_backend.task_types import TaskType, normalize_task_type

# name -> {task: "module.Class"}, plus capabilities
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "XGBoost": {"Classification": "xgboost.XGBClassifier", "Regression": "xgboost.XGBRegressor",
                "family": "boosted trees", "random_state": True, "sample_weight": True,
                "defaults": {"enable_categorical": False}},
    "Random Forest": {"Classification": "sklearn.ensemble.RandomForestClassifier",
                      "Regression": "sklearn.ensemble.RandomForestRegressor",
                      "family": "tree ensemble", "random_state": True, "sample_weight": True},
    "Extra Trees": {"Classification": "sklearn.ensemble.ExtraTreesClassifier",
                    "Regression": "sklearn.ensemble.ExtraTreesRegressor",
                    "family": "tree ensemble", "random_state": True, "sample_weight": True},
    "Gradient Boosting": {"Classification": "sklearn.ensemble.GradientBoostingClassifier",
                          "Regression": "sklearn.ensemble.GradientBoostingRegressor",
                          "family": "boosted trees", "random_state": True, "sample_weight": True},
    "Hist Gradient Boosting": {"Classification": "sklearn.ensemble.HistGradientBoostingClassifier",
                               "Regression": "sklearn.ensemble.HistGradientBoostingRegressor",
                               "family": "boosted trees", "random_state": True, "sample_weight": True},
    "AdaBoost": {"Classification": "sklearn.ensemble.AdaBoostClassifier",
                 "Regression": "sklearn.ensemble.AdaBoostRegressor",
                 "family": "boosted trees", "random_state": True, "sample_weight": True},
    "Decision Tree": {"Classification": "sklearn.tree.DecisionTreeClassifier",
                      "Regression": "sklearn.tree.DecisionTreeRegressor",
                      "family": "single tree", "random_state": True, "sample_weight": True},
    "SVM": {"Classification": "sklearn.svm.SVC", "Regression": "sklearn.svm.SVR",
            "family": "kernel", "random_state": False, "sample_weight": True},
    "KNN": {"Classification": "sklearn.neighbors.KNeighborsClassifier",
            "Regression": "sklearn.neighbors.KNeighborsRegressor",
            "family": "neighbours", "random_state": False, "sample_weight": False},
    "Logistic Regression": {"Classification": "sklearn.linear_model.LogisticRegression",
                            "family": "linear", "random_state": True, "sample_weight": True,
                            "defaults": {"max_iter": 1000}},
    "Linear Regression": {"Regression": "sklearn.linear_model.LinearRegression",
                          "family": "linear", "random_state": False, "sample_weight": True},
    "Ridge": {"Regression": "sklearn.linear_model.Ridge", "family": "linear", "random_state": True,
              "sample_weight": True},
    "Lasso": {"Regression": "sklearn.linear_model.Lasso", "family": "linear", "random_state": True,
              "sample_weight": True},
    "ElasticNet": {"Regression": "sklearn.linear_model.ElasticNet", "family": "linear", "random_state": True,
                   "sample_weight": True},
}

TIME_SERIES_MODELS = ["Prophet", "ARIMA", "SARIMAX"]
ENSEMBLE_NAME = "Voting Ensemble"


def supported_models(task_type, is_time_series: bool = False) -> List[str]:
    task = normalize_task_type(task_type) or TaskType.REGRESSION
    names = [n for n, spec in MODEL_SPECS.items() if task.value in spec]
    if is_time_series and task == TaskType.REGRESSION:
        names += TIME_SERIES_MODELS
        try:
            import tensorflow  # noqa: F401
            names.append("LSTM")
        except ImportError:
            pass
    return sorted(names)


def estimator_path(name: str, task_type) -> Optional[str]:
    task = normalize_task_type(task_type)
    spec = MODEL_SPECS.get(name)
    return spec.get(task.value) if spec and task else None


def estimator_class(name: str, task_type):
    path = estimator_path(name, task_type)
    if path is None:
        return None
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def build_model(name: str, task_type, random_state: Optional[int] = None, params: Dict[str, Any] = None):
    """Instantiate a registry model with its defaults, the seed (if the estimator takes one) and ``params``."""
    cls = estimator_class(name, task_type)
    if cls is None:
        raise ValueError(f"'{name}' is not supported for {normalize_task_type(task_type)} tasks")
    spec = MODEL_SPECS[name]
    kwargs = dict(spec.get("defaults", {}))
    if spec.get("random_state") and random_state is not None:
        kwargs["random_state"] = random_state
    kwargs.update(params or {})
    return cls(**kwargs)


def supports_sample_weight(name: str) -> bool:
    return bool(MODEL_SPECS.get(name, {}).get("sample_weight"))
