"""B-13: the tuner uses the ranges the UI offers; stratified CV; imbalance-aware scoring; same test split."""
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from app_backend.model_trainer import ModelTrainer
from app_backend.model_tuner import SEARCH_SPACES, ModelTuner, search_space
from app_backend.preprocessing_engine.engine import AutoPreprocessor


def _trainer(df, target, models):
    p = AutoPreprocessor(target_col=target, verbose=False, random_state=0)
    out = p.fit_transform(df=df)
    tr = ModelTrainer(df, target, p.task_type, random_state=0)
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    tr.run_selected_models(models)
    return tr


def test_every_registry_model_with_params_has_a_space():
    assert search_space("XGBoost (Tuned)") == SEARCH_SPACES["XGBoost"]
    assert search_space("Linear Regression") == {}


def test_custom_ranges_are_used(adult_df):
    tr = _trainer(adult_df.head(1200), "class", ["Random Forest"])
    tuner = ModelTuner(tr)
    res = tuner.tune_model("Random Forest", time_budget=60, cv_folds=3, n_trials=4,
                           param_ranges={"n_estimators": (20, 30), "max_depth": (3, 4)})
    assert "Error" not in res, res
    assert 20 <= res["Best Params"]["n_estimators"] <= 30 and 3 <= res["Best Params"]["max_depth"] <= 4
    assert res["param_ranges_used"]["n_estimators"] == [20, 30]
    assert res["trials"] == 4 and res["cv_scheme"] == "StratifiedKFold"
    assert isinstance(tuner._cv(3), StratifiedKFold)


def test_unknown_or_invalid_ranges_are_rejected(adult_df):
    tr = _trainer(adult_df.head(600), "class", ["Random Forest"])
    assert "not in the search space" in ModelTuner(tr).tune_model("Random Forest", param_ranges={"gamma": (1, 2)})["Error"]
    assert "low > high" in ModelTuner(tr).tune_model("Random Forest", param_ranges={"max_depth": (9, 3)})["Error"]
    assert "no tunable" in ModelTuner(tr).tune_model("Linear Regression")["Error"]


def test_imbalanced_scoring_is_f1(credit_like_df):
    tr = _trainer(credit_like_df, "Class", ["Logistic Regression"])
    tuner = ModelTuner(tr)
    assert tuner.scoring()["metric"] == "F1 Score"
    res = tuner.tune_model("Logistic Regression", n_trials=3, time_budget=60)
    assert res["cv_metric"] == "F1 Score" and "F1 Score" in res


def test_tuned_model_is_scored_on_the_same_test_split(california_df):
    tr = _trainer(california_df.head(1200), "MedHouseVal", ["Ridge"])
    res = ModelTuner(tr).tune_model("Ridge", n_trials=5, time_budget=60)
    model = tr.trained_models["Ridge (Tuned)"]
    rmse = float(np.sqrt(np.mean((model.predict(tr.X_test) - np.asarray(tr.y_test)) ** 2)))
    assert res["RMSE"] == pytest.approx(rmse, abs=1e-3)
    assert res["cv_metric"] == "RMSE" and res["cv_best_score"] > 0
