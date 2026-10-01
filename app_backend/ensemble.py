"""
Optional voting ensemble over models that were already trained on the same split.

This is a plain average (soft voting over class probabilities, or the mean of
regression predictions) of the selected models. The members are not re-fitted and
no weights are learned. It is offered as an explicit option in the UI and labelled
as such; the LLM does not choose its members.
"""
from typing import Dict, List

import numpy as np


class PrefitVotingEnsemble:
    def __init__(self, models: Dict[str, object], task_type: str):
        self.models = dict(models)
        self.members: List[str] = list(models)
        self.task_type = task_type
        first = next(iter(self.models.values()))
        self.classes_ = getattr(first, "classes_", None)
        self.voting = "mean"
        if self.classes_ is not None:
            self.voting = "soft" if all(hasattr(m, "predict_proba") for m in self.models.values()) else "hard"

    def fit(self, X, y=None, **kwargs):  # members are already fitted
        return self

    def predict_proba(self, X):
        if self.voting != "soft":
            raise AttributeError("predict_proba is only available for soft voting")
        return np.mean([m.predict_proba(X) for m in self.models.values()], axis=0)

    def predict(self, X):
        if self.classes_ is None:
            return np.mean([np.asarray(m.predict(X), dtype=float) for m in self.models.values()], axis=0)
        if self.voting == "soft":
            return np.asarray(self.classes_)[np.argmax(self.predict_proba(X), axis=1)]
        votes = np.stack([np.asarray(m.predict(X)) for m in self.models.values()], axis=1)
        out = []
        for row in votes:
            vals, counts = np.unique(row, return_counts=True)
            out.append(vals[np.argmax(counts)])
        return np.asarray(out)

    def describe(self) -> str:
        kind = {"soft": "soft-voting average of class probabilities", "hard": "majority vote",
                "mean": "mean of predictions"}[self.voting]
        return f"{kind} over {', '.join(self.members)} (members trained above, not re-fitted, equal weights)"
