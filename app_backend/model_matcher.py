"""
Map model names written by an LLM to the models the app can train.

Order of checks for each name:
  1. exact     the name is already a supported model name
  2. alias     normalised name (lower-case, alphanumerics only, parenthetical qualifiers
               and generic words such as "model" removed) is in the alias table
               (scikit-learn / XGBoost class names, abbreviations, spelled-out names)
  3. fuzzy     rapidfuzz ratio >= 85 against the alias table (typos such as "Random Forrest")
  4. otherwise the name is reported, never silently dropped or coerced:
       unsupported  a real model family the app does not train (LightGBM, CatBoost, MLP, ...),
                    or a supported family that does not apply to this task
       unmatched    nothing plausible (e.g. the hallucination "RandomForestClassifierRegressor")

The old UI matcher used substring containment, which accepted hallucinations
("RandomForestClassifierRegressor" -> Random Forest), mapped
HistGradientBoostingClassifier to Gradient Boosting and returned the selection in
set order.
"""
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from app_backend import model_registry
from app_backend.task_types import TaskType, normalize_task_type

FUZZY_THRESHOLD = 85.0

ALIASES: Dict[str, List[str]] = {
    "XGBoost": ["xgboost", "xgb", "xgbclassifier", "xgbregressor", "xgboostclassifier", "xgboostregressor",
                "extremegradientboosting"],
    "Random Forest": ["randomforest", "randomforests", "rf", "randomforestclassifier", "randomforestregressor"],
    "Extra Trees": ["extratrees", "extratree", "extratreesclassifier", "extratreesregressor",
                    "extremelyrandomizedtrees"],
    "Gradient Boosting": ["gradientboosting", "gbm", "gb", "gbdt", "gradientboostingmachine",
                          "gradientboostingclassifier", "gradientboostingregressor", "gradientboostedtrees",
                          "gradientboostedtree", "gradientboostingtrees"],
    "Hist Gradient Boosting": ["histgradientboosting", "histgradientboostingclassifier",
                               "histgradientboostingregressor", "hgb", "hgbt", "histogrambasedgradientboosting"],
    "AdaBoost": ["adaboost", "adaboostclassifier", "adaboostregressor", "adaptiveboosting"],
    "Decision Tree": ["decisiontree", "decisiontrees", "decisiontreeclassifier", "decisiontreeregressor", "cart"],
    "SVM": ["svm", "svc", "svr", "supportvectormachine", "supportvectormachines", "supportvectorclassifier",
            "supportvectorregressor", "supportvectorregression", "linearsvm", "linearsvc", "linearsvr",
            "kernelsvm", "rbfsvm"],
    "KNN": ["knn", "kneighbors", "kneighborsclassifier", "kneighborsregressor", "knearestneighbors",
            "knearestneighbor", "knearestneighbours", "nearestneighbors", "knnclassifier", "knnregressor"],
    "Logistic Regression": ["logisticregression", "logreg", "logit"],
    "Linear Regression": ["linearregression", "ols", "ordinaryleastsquares"],
    "Ridge": ["ridge", "ridgeregression", "ridgeregressor"],
    "Lasso": ["lasso", "lassoregression"],
    "ElasticNet": ["elasticnet", "elasticnetregression"],
    "Prophet": ["prophet", "fbprophet", "facebookprophet"],
    "ARIMA": ["arima"],
    "SARIMAX": ["sarimax", "sarima"],
    "LSTM": ["lstm"],
}

NOTES = {"linearsvm": "the app's SVM uses an RBF kernel by default (the tuner can search kernel=linear)",
         "linearsvc": "the app's SVM uses an RBF kernel by default (the tuner can search kernel=linear)"}

# Real model families the app does not train (reported as "unsupported", not as hallucinations).
UNSUPPORTED = {
    "lightgbm": "LightGBM", "lgbm": "LightGBM", "lgbmclassifier": "LightGBM", "lgbmregressor": "LightGBM",
    "catboost": "CatBoost", "catboostclassifier": "CatBoost", "catboostregressor": "CatBoost",
    "neuralnetwork": "Neural Network", "neuralnetworks": "Neural Network", "mlp": "MLP",
    "mlpclassifier": "MLP", "mlpregressor": "MLP", "deeplearning": "Deep learning",
    "multilayerperceptron": "MLP", "naivebayes": "Naive Bayes", "gaussiannb": "Naive Bayes",
    "multinomialnb": "Naive Bayes", "bernoullinb": "Naive Bayes", "isolationforest": "Isolation Forest",
    "stacking": "Stacking ensemble", "stackingensemble": "Stacking ensemble",
    "stackingclassifier": "Stacking ensemble", "stackingregressor": "Stacking ensemble",
    "votingclassifier": "Voting ensemble (available as an option on the Train page)",
    "votingregressor": "Voting ensemble (available as an option on the Train page)",
    "votingensemble": "Voting ensemble (available as an option on the Train page)",
    "bagging": "Bagging", "baggingclassifier": "Bagging", "baggingregressor": "Bagging",
    "ridgeclassifier": "RidgeClassifier", "bayesianridge": "Bayesian Ridge", "sgdclassifier": "SGD",
    "sgdregressor": "SGD", "gaussianprocess": "Gaussian Process", "lda": "LDA",
    "lineardiscriminantanalysis": "LDA", "qda": "QDA", "tabnet": "TabNet", "transformer": "Transformer",
    "transformers": "Transformer", "gru": "GRU", "tcn": "TCN", "ets": "ETS", "tbats": "TBATS", "var": "VAR",
    "huberregressor": "Huber regression", "quantileregression": "Quantile regression",
}

GENERIC_WORDS = ("model", "models", "algorithm", "estimator", "sklearn", "scikitlearn")

DEFAULTS = {TaskType.CLASSIFICATION: ["XGBoost", "Random Forest", "Logistic Regression"],
            TaskType.REGRESSION: ["XGBoost", "Random Forest", "Gradient Boosting"]}


@dataclass
class MatchResult:
    raw: str
    matched: Optional[str]
    match_type: str          # exact | alias | fuzzy | unsupported | unmatched
    score: float = 0.0
    note: str = ""

    def to_dict(self):
        return asdict(self)


def normalize_name(name: str) -> Tuple[str, str]:
    """(normalised key, removed qualifier text)."""
    text = str(name)
    qualifiers = " ".join(re.findall(r"\(([^)]*)\)", text))
    text = re.sub(r"\([^)]*\)", " ", text).lower()
    key = re.sub(r"[^a-z0-9]", "", text)
    for word in GENERIC_WORDS:
        if key.endswith(word) and len(key) > len(word) + 2:
            key = key[: -len(word)]
        if key.startswith(word) and len(key) > len(word) + 2:
            key = key[len(word):]
    return key, qualifiers.strip()


def _alias_index() -> Dict[str, str]:
    index = {}
    for canonical, keys in ALIASES.items():
        index[re.sub(r"[^a-z0-9]", "", canonical.lower())] = canonical
        for k in keys:
            index[k] = canonical
    return index


_ALIAS_INDEX = _alias_index()


def match_name(raw, supported: List[str]) -> MatchResult:
    if not isinstance(raw, str):
        return MatchResult(str(raw), None, "unmatched", note="not a string")
    if raw in supported:
        return MatchResult(raw, raw, "exact", 100.0)
    key, qualifier = normalize_name(raw)
    note = f"qualifier '({qualifier})' ignored" if qualifier else ""

    def resolve(canonical: str, kind: str, score: float) -> MatchResult:
        extra = NOTES.get(key, "")
        full_note = "; ".join(x for x in (note, extra) if x)
        if canonical in supported:
            return MatchResult(raw, canonical, kind, score, full_note)
        why = ("time-series only" if canonical in model_registry.TIME_SERIES_MODELS + ["LSTM"]
               else "not available for this task")
        return MatchResult(raw, None, "unsupported", score, f"{canonical}: {why}")

    if key in _ALIAS_INDEX:
        return resolve(_ALIAS_INDEX[key], "alias", 100.0)
    if key in UNSUPPORTED:
        return MatchResult(raw, None, "unsupported", 100.0, f"{UNSUPPORTED[key]} is not trained by this app")
    if len(key) >= 4:
        from rapidfuzz import fuzz, process

        candidates = list(_ALIAS_INDEX) + list(UNSUPPORTED)
        best = process.extractOne(key, candidates, scorer=fuzz.ratio)
        if best and best[1] >= FUZZY_THRESHOLD:
            hit, score = best[0], float(best[1])
            if hit in _ALIAS_INDEX:
                return resolve(_ALIAS_INDEX[hit], "fuzzy", score)
            return MatchResult(raw, None, "unsupported", score, f"{UNSUPPORTED[hit]} is not trained by this app")
    return MatchResult(raw, None, "unmatched", note="no supported model with this name")


def match_names(names, supported: List[str]) -> List[MatchResult]:
    return [match_name(n, supported) for n in (names or [])]


def select_models(names, supported: List[str], task_type) -> Tuple[List[str], List[MatchResult], bool]:
    """(ordered, de-duplicated selection, per-name results, whether the task default was used)."""
    results = match_names(names, supported)
    selection: List[str] = []
    for r in results:
        if r.matched and r.matched not in selection:
            selection.append(r.matched)
    used_default = False
    if not selection:
        used_default = True
        task = normalize_task_type(task_type) or TaskType.REGRESSION
        selection = [m for m in DEFAULTS[task] if m in supported]
    return selection, results, used_default
