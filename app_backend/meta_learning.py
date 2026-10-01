"""
Dataset profiles and profile similarity for meta-learning.

Each workspace stores a profile of seven meta-features. Each feature is mapped to
[0, 1] with the fixed scale below, and the similarity of two profiles is

    similarity = 1 - sqrt(mean_i (a_i - b_i)^2)            (1 = identical, 0 = maximally different)

A past workspace is used as a prior only if it has the same task type, has a
recorded best model, and its similarity is >= SIMILARITY_THRESHOLD. The old rule
("same task and rows within 50% or columns within 30%") never matched any pair of
the benchmark datasets, and the documented profile similarity did not exist.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

SIMILARITY_THRESHOLD = 0.90

# feature -> (description, scaler to [0, 1])
PROFILE_FEATURES = {
    "log_rows": ("log10(rows) / 6", lambda v: min(max(v, 0.0) / 6.0, 1.0)),
    "n_features": ("log10(features + 1) / 3", lambda v: min(math.log10(max(v, 0) + 1) / 3.0, 1.0)),
    "categorical_share": ("share of categorical features", lambda v: min(max(v, 0.0), 1.0)),
    "missing_rate": ("share of missing cells", lambda v: min(max(v, 0.0), 1.0)),
    "minority_share": ("minority class share x 2 (0 for regression)", lambda v: min(max(v, 0.0) * 2.0, 1.0)),
    "mean_abs_skew": ("mean |skewness| of numeric features / 10", lambda v: min(max(v, 0.0) / 10.0, 1.0)),
    "linearity": ("mean |corr(feature, target)|", lambda v: min(max(v, 0.0), 1.0)),
}


def vectorize(profile: Dict[str, float]) -> np.ndarray:
    return np.array([scale(float(profile.get(k) or 0.0)) for k, (_, scale) in PROFILE_FEATURES.items()])


def similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    va, vb = vectorize(a), vectorize(b)
    return float(1.0 - np.sqrt(np.mean((va - vb) ** 2)))


def explain(a: Dict[str, float], b: Dict[str, float], top: int = 2) -> Tuple[List[str], List[str]]:
    """(closest features, most different features) between two profiles."""
    diffs = np.abs(vectorize(a) - vectorize(b))
    names = list(PROFILE_FEATURES)
    order = np.argsort(diffs)
    return [names[i] for i in order[:top]], [names[i] for i in order[::-1][:top] if diffs[i] > 0.05]


def valid_profile(profile: Optional[Dict]) -> bool:
    return isinstance(profile, dict) and all(k in profile for k in PROFILE_FEATURES)
