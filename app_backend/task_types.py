"""
Canonical task type used everywhere (UI, workspace JSON, trainer, tuner, API).

Before this module the statistical engine wrote "Classification"/"Regression"
while the preprocessor and profiler wrote "classification"/"regression", and the
two used different cardinality thresholds (``< 10`` vs ``<= 10``). A lowercase
comparison in the UI therefore never recorded a best model for regression
workspaces. Everything now goes through :func:`normalize_task_type` and
:func:`detect_task_type`.
"""
from enum import Enum
from typing import Optional

import pandas as pd

# A numeric target with at most this many distinct values is treated as classification.
MAX_CLASSES_FOR_NUMERIC_TARGET = 10


class TaskType(str, Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"

    def __str__(self) -> str:  # json / f-strings show the canonical casing
        return self.value


def normalize_task_type(value) -> Optional[TaskType]:
    """Map any casing ("regression", "Regression", TaskType.REGRESSION) to TaskType. ``None``/"auto" -> None."""
    if value is None:
        return None
    if isinstance(value, TaskType):
        return value
    text = str(value).strip().lower()
    if text in ("", "auto", "unknown", "none"):
        return None
    if text.startswith("class"):
        return TaskType.CLASSIFICATION
    if text.startswith("regr"):
        return TaskType.REGRESSION
    raise ValueError(f"Unknown task type: {value!r}")


def is_classification(value) -> bool:
    return normalize_task_type(value) == TaskType.CLASSIFICATION


def is_regression(value) -> bool:
    return normalize_task_type(value) == TaskType.REGRESSION


def detect_task_type(y: pd.Series) -> TaskType:
    """Classification for object/category/bool targets or numeric targets with <= 10 distinct values."""
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if y.dtype == object or y.dtype.name in ("category", "bool", "string"):
        return TaskType.CLASSIFICATION
    if y.nunique(dropna=True) <= MAX_CLASSES_FOR_NUMERIC_TARGET:
        return TaskType.CLASSIFICATION
    return TaskType.REGRESSION
