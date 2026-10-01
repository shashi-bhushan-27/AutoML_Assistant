"""B-10: table-driven test of the model-name matcher over the 30 probe names of ablation/checks.py."""
import ast
import os

import pytest

from app_backend.model_matcher import match_name, select_models
from app_backend.model_registry import supported_models
from tests.conftest import ROOT

CLS = supported_models("Classification")
REG = supported_models("Regression")

# probe -> (classification result, regression result); result = canonical model or match type
EXPECTED = {
    "Random Forest": ("Random Forest", "Random Forest"),
    "random forest": ("Random Forest", "Random Forest"),
    "RandomForest": ("Random Forest", "Random Forest"),
    "Random Forest Classifier": ("Random Forest", "Random Forest"),
    "RandomForestClassifier": ("Random Forest", "Random Forest"),
    "RandomForestClassifierRegressor": ("unmatched", "unmatched"),       # hallucination
    "XGBoost": ("XGBoost", "XGBoost"),
    "XGBClassifier": ("XGBoost", "XGBoost"),
    "XGBRegressor": ("XGBoost", "XGBoost"),
    "xgboost (weighted)": ("XGBoost", "XGBoost"),
    "LightGBM": ("unsupported", "unsupported"),
    "CatBoost": ("unsupported", "unsupported"),
    "HistGradientBoostingClassifier": ("Hist Gradient Boosting", "Hist Gradient Boosting"),
    "Gradient Boosting Machine": ("Gradient Boosting", "Gradient Boosting"),
    "Support Vector Machine": ("SVM", "SVM"),
    "SVC": ("SVM", "SVM"),
    "Linear SVM": ("SVM", "SVM"),
    "K-Nearest Neighbors": ("KNN", "KNN"),
    "KNeighborsClassifier": ("KNN", "KNN"),
    "Logistic Regression": ("Logistic Regression", "unsupported"),
    "LogisticRegression": ("Logistic Regression", "unsupported"),
    "Neural Network": ("unsupported", "unsupported"),
    "MLPClassifier": ("unsupported", "unsupported"),
    "Stacking Ensemble": ("unsupported", "unsupported"),
    "Isolation Forest": ("unsupported", "unsupported"),
    "Naive Bayes": ("unsupported", "unsupported"),
    "Ridge Classifier": ("unsupported", "unsupported"),
    "Linear Regression": ("unsupported", "Linear Regression"),
    "Bayesian Ridge": ("unsupported", "unsupported"),
    "Prophet": ("unsupported", "unsupported"),                          # not a time-series workspace
}


def test_probe_list_matches_ablation_checks():
    """The table covers exactly HALLUCINATION_PROBES from ablation/checks.py."""
    with open(os.path.join(ROOT, "ablation", "checks.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    probes = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "HALLUCINATION_PROBES")
    assert list(EXPECTED) == probes


@pytest.mark.parametrize("name", list(EXPECTED))
@pytest.mark.parametrize("task,supported,col", [("Classification", CLS, 0), ("Regression", REG, 1)])
def test_probe(name, task, supported, col):
    expected = EXPECTED[name][col]
    r = match_name(name, supported)
    if expected in ("unmatched", "unsupported"):
        assert r.matched is None and r.match_type == expected, r
    else:
        assert r.matched == expected, r
        assert r.match_type in ("exact", "alias", "fuzzy")


@pytest.mark.parametrize("typo,expected", [("Random Forrest", "Random Forest"), ("XGBoostt", "XGBoost"),
                                           ("Logistic Regresion", "Logistic Regression"),
                                           ("Decision Trees", "Decision Tree")])
def test_typos_are_fuzzy_matched(typo, expected):
    r = match_name(typo, CLS)
    assert r.matched == expected and r.match_type in ("alias", "fuzzy") and r.score >= 85


def test_selection_is_ordered_and_deduplicated():
    sel, results, used_default = select_models(["SVC", "XGBClassifier", "Support Vector Machine", "LightGBM",
                                                "Random Forest"], CLS, "Classification")
    assert sel == ["SVM", "XGBoost", "Random Forest"]
    assert [r.match_type for r in results] == ["alias", "alias", "alias", "unsupported", "exact"]
    assert not used_default


def test_nothing_matched_uses_flagged_default():
    sel, results, used_default = select_models(["LightGBM", "CatBoost", {"weird": 1}], CLS, "Classification")
    assert used_default and sel == ["XGBoost", "Random Forest", "Logistic Regression"]
    assert results[2].match_type == "unmatched"


def test_time_series_models_only_in_time_series_workspaces():
    ts = supported_models("Regression", is_time_series=True)
    assert match_name("Prophet", ts).matched == "Prophet"
    assert "time-series" in match_name("Prophet", REG).note
