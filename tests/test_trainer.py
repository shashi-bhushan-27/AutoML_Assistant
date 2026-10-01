"""B-7 (string labels, pos_label), B-15 (seeds, single predict_proba, failure list, ensemble), B-16 (one pipeline)."""
import pandas as pd
import pytest

from app_backend import model_registry
from app_backend.model_trainer import ModelTrainer
from app_backend.preprocessing_engine.engine import AutoPreprocessor


@pytest.fixture(scope="module")
def adult_split(adult_df):
    p = AutoPreprocessor(target_col="class", verbose=False, random_state=0)
    out = p.fit_transform(df=adult_df)
    return adult_df, p, out


def _trainer(df, p, out, **kw):
    tr = ModelTrainer(df, "class", p.task_type, **kw)
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    return tr


def test_string_labels_train_xgboost_and_logistic(adult_split):
    """B-7: with '>50K' labels XGBoost failed ('Invalid classes inferred') and F1 failed for LR."""
    df, p, out = adult_split
    res = _trainer(df, p, out).run_selected_models(["XGBoost", "Logistic Regression"])
    assert "Error" not in res.columns or res["Error"].isna().all(), res.get("Error")
    assert (res["F1 Score"] > 0.4).all() and (res["AUC-ROC"] > 0.8).all()


def test_positive_class_is_minority(adult_split):
    df, p, out = adult_split
    tr = _trainer(df, p, out)
    res = tr.run_selected_models(["Logistic Regression"]).iloc[0]
    minority = pd.Series(out["y_train"]).value_counts().idxmin()
    assert res["Positive class"] == minority
    assert p.decode_target([minority])[0] == ">50K"


def test_seeded_models_are_reproducible(adult_split):
    df, p, out = adult_split
    a = _trainer(df, p, out, random_state=7).run_selected_models(["Random Forest"]).iloc[0]["Accuracy"]
    b = _trainer(df, p, out, random_state=7).run_selected_models(["Random Forest"]).iloc[0]["Accuracy"]
    assert a == b
    tr = _trainer(df, p, out, random_state=7)
    tr.run_selected_models(["Random Forest"])
    assert tr.trained_models["Random Forest"].random_state == 7


def test_predict_proba_is_computed_once(adult_split, monkeypatch):
    df, p, out = adult_split
    calls = {"n": 0}
    real = model_registry.build_model

    def counting(name, task, random_state=None, params=None):
        model = real(name, task, random_state, params)
        orig = model.predict_proba

        def wrapped(X):
            calls["n"] += 1
            return orig(X)

        model.predict_proba = wrapped
        return model

    monkeypatch.setattr(model_registry, "build_model", counting)
    _trainer(df, p, out).run_selected_models(["Logistic Regression"])
    assert calls["n"] == 1


def test_failures_are_listed_not_swallowed(adult_split):
    df, p, out = adult_split
    tr = _trainer(df, p, out)
    res = tr.run_selected_models(["XGBoost", "LightGBM", {"not": "a name"}, "Linear Regression"])
    failed = {f["Model"]: f["Error"] for f in tr.failed_models}
    assert set(failed) == {"LightGBM", "{'not': 'a name'}", "Linear Regression"}
    assert "Not supported for Classification" in failed["LightGBM"]
    assert res["Error"].notna().sum() == 3 and res.loc[res.Model == "XGBoost", "Error"].isna().all()


def test_training_requires_the_pipeline():
    """B-16: the old fallback to a second, non-stratified preprocessor is gone."""
    tr = ModelTrainer(pd.DataFrame({"a": [1, 2], "y": [0, 1]}), "y", "Classification")
    with pytest.raises(RuntimeError, match="Prepare"):
        tr.run_selected_models(["XGBoost"])


def test_voting_ensemble_is_labelled(adult_split):
    df, p, out = adult_split
    tr = _trainer(df, p, out)
    res = tr.run_selected_models(["XGBoost", "Logistic Regression"], add_ensemble=True)
    row = res[res.Model == model_registry.ENSEMBLE_NAME].iloc[0]
    assert "soft-voting" in row["Notes"] and "not re-fitted" in row["Notes"]
    assert 0.5 < row["Accuracy"] <= 1


def test_class_weights_option(credit_like_df):
    p = AutoPreprocessor(target_col="Class", verbose=False)
    out = p.fit_transform(df=credit_like_df)
    tr = ModelTrainer(credit_like_df, "Class", p.task_type, class_weight="balanced")
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    res = tr.run_selected_models(["Logistic Regression", "KNN"]).set_index("Model")
    assert res.loc["Logistic Regression", "Notes"] == "trained with balanced class weights"
    assert "not supported" in res.loc["KNN", "Notes"]


def test_regression_metrics(california_df):
    p = AutoPreprocessor(target_col="MedHouseVal", verbose=False)
    out = p.fit_transform(df=california_df)
    tr = ModelTrainer(california_df, "MedHouseVal", p.task_type)
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    res = tr.run_selected_models(["XGBoost", "Ridge"]).set_index("Model")
    assert res.loc["XGBoost", "R²"] > 0.6 and res.loc["XGBoost", "RMSE"] < res.loc["Ridge", "RMSE"]
