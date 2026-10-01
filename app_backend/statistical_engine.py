"""
Dataset statistics sent to the recommendation engine (and stored as the
workspace profile for meta-learning). The input dataframe is never modified.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew

from app_backend.task_types import TaskType, detect_task_type

logger = logging.getLogger(__name__)


def get_column_types(df):
    """Numeric, categorical and datetime columns (datetime = datetime dtype, or a date/time-named column
    whose values all parse as dates)."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif ("date" in str(col).lower() or "time" in str(col).lower()) and df[col].dtype == object:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(df[col].dropna().head(500), errors="raise")
                date_cols.append(col)
            except (ValueError, TypeError):
                pass
    return num_cols, cat_cols, date_cols


def check_stationarity(series):
    """Augmented Dickey-Fuller test."""
    from statsmodels.tsa.stattools import adfuller

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 20:
        return "Unknown (too little data)"
    try:
        p_value = adfuller(clean)[1]
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("ADF test failed: %s", exc)
        return "Check failed"
    return "Stationary (Good for ARIMA)" if p_value < 0.05 else "Non-Stationary (Needs Differencing/Transformation)"


def analyze_dataset(df, target_col=None):
    """Statistics for the LLM prompt plus ``profile`` (meta-learning features). Does not modify ``df``."""
    if isinstance(target_col, (list, tuple)):
        target_col = target_col[0] if target_col else None
    stats = {"rows": int(df.shape[0]), "columns": int(df.shape[1]),
             "missing_values": int(df.isnull().sum().sum())}
    num_cols, cat_cols, date_cols = get_column_types(df)
    stats["numerical_columns"] = num_cols
    stats["categorical_columns"] = cat_cols
    stats["datetime_columns"] = date_cols

    if date_cols:
        stats["is_time_series"] = True
        stats["time_column"] = date_cols[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                times = pd.to_datetime(df[date_cols[0]], errors="coerce").sort_values()
            stats["time_frequency"] = str(times.diff().median())
        except (ValueError, TypeError):
            stats["time_frequency"] = "Irregular"
    else:
        stats["is_time_series"] = False
        stats["time_frequency"] = "None"

    features = [c for c in df.columns if c != target_col]
    feat_num = [c for c in num_cols if c != target_col]
    feat_cat = [c for c in cat_cols if c != target_col]
    stats["n_features"] = len(features)
    cells = max(len(df) * max(len(features), 1), 1)
    missing_rate = float(df[features].isnull().sum().sum() / cells) if features else 0.0
    skews = [abs(float(skew(df[c].dropna()))) for c in feat_num if df[c].dropna().nunique() > 1]
    profile = {
        "log_rows": float(np.log10(max(len(df), 1))),
        "n_features": float(len(features)),
        "categorical_share": float(len(feat_cat) / max(len(features), 1)),
        "missing_rate": missing_rate,
        "minority_share": 0.0,
        "mean_abs_skew": float(np.mean(skews)) if skews else 0.0,
        "linearity": 0.0,
    }

    if target_col and target_col in df.columns:
        y = df[target_col]
        task = detect_task_type(y)
        stats["task_type"] = task.value
        if task == TaskType.CLASSIFICATION:
            shares = y.value_counts(normalize=True)
            stats["n_classes"] = int(len(shares))
            stats["minority_class_share"] = round(float(shares.min()), 4) if len(shares) else None
            stats["class_distribution"] = {str(k): round(float(v), 4) for k, v in shares.head(20).items()}
            profile["minority_share"] = float(shares.min()) if len(shares) >= 2 else 0.0
            y_num = pd.Series(pd.factorize(y)[0], index=y.index).where(y.notna())
        else:
            y_num = pd.to_numeric(y, errors="coerce")
            stats["target_mean"] = round(float(y_num.mean()), 4)
            stats["target_std"] = round(float(y_num.std()), 4)
            stats["target_skewness"] = round(float(skew(y_num.dropna())), 4)
        if feat_num:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = df[feat_num].corrwith(y_num).abs()
            lin = float(corr.mean()) if corr.notna().any() else 0.0
            profile["linearity"] = lin
            stats["linearity"] = ("High (linear models might work)" if lin > 0.5
                                  else "Low (non-linear / tree models likely better)")
        if task == TaskType.REGRESSION and stats["is_time_series"]:
            stats["stationarity"] = check_stationarity(y)
    stats["profile"] = profile
    return stats
