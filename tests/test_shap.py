"""B-12: SHAP routing by type, dtype handling, time budget, cancellation, additivity retry."""
import threading
import time

import numpy as np
import pandas as pd
import pytest

from app_backend.model_trainer import ModelTrainer
from app_backend.preprocessing_engine.engine import AutoPreprocessor
from app_backend.shap_explainer import SHAPExplainer

FAMILIES = {"Classification": ["Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression", "SVM", "KNN"],
            "Regression": ["Random Forest", "XGBoost", "Gradient Boosting", "Ridge", "SVM", "KNN"]}
EXPECTED_ROUTE = {"Random Forest": "tree", "XGBoost": "tree", "Gradient Boosting": "tree",
                  "Logistic Regression": "linear", "Ridge": "linear", "SVM": "kernel", "KNN": "kernel",
                  "AdaBoost": "kernel", "Hist Gradient Boosting": "tree", "Extra Trees": "tree",
                  "Decision Tree": "tree", "Voting Ensemble": "kernel"}


def _trained(df, target, models, ensemble=False):
    p = AutoPreprocessor(target_col=target, verbose=False, random_state=0)
    out = p.fit_transform(df=df)
    tr = ModelTrainer(df, target, p.task_type, random_state=0)
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    tr.run_selected_models(models, add_ensemble=ensemble)
    return p, out, tr


@pytest.fixture(scope="module")
def cls_models(adult_df):
    return _trained(adult_df.head(1500), "class", FAMILIES["Classification"] + ["AdaBoost", "Hist Gradient Boosting"],
                    ensemble=True)


@pytest.fixture(scope="module")
def reg_models(california_df):
    return _trained(california_df.head(1500), "MedHouseVal", FAMILIES["Regression"])


@pytest.mark.parametrize("task", ["Classification", "Regression"])
def test_all_six_families_finish_or_report_budget(task, cls_models, reg_models):
    """Acceptance: every family returns values within the budget or a clean 'budget_exceeded'."""
    p, out, tr = cls_models if task == "Classification" else reg_models
    budget = 8.0
    for name in FAMILIES[task]:
        ex = SHAPExplainer(tr.trained_models[name], out["X_train"], out["X_test"], task_type=task)
        assert ex._get_model_type() == EXPECTED_ROUTE[name], name
        t0 = time.perf_counter()
        res = ex.explain(max_rows=40, time_budget_s=budget)
        wall = time.perf_counter() - t0
        assert res.status in ("ok", "budget_exceeded"), (name, res.status, res.error)
        assert res.rows_done >= 1 and ex.get_feature_importance_df() is not None, name
        per_row = res.seconds_per_row or 0
        assert wall <= budget + 3 * per_row + 5, (name, wall)
        assert np.isfinite(ex._shap_values).all()


def test_routing_is_by_type_not_name(cls_models):
    """'boost' in the class name used to route AdaBoost to TreeExplainer, which rejects it in shap 0.46
    ("Model type not yet supported by TreeExplainer"). Routing is now by estimator type."""
    import shap

    p, out, tr = cls_models
    ada = tr.trained_models["AdaBoost"]
    with pytest.raises(Exception, match="not yet supported"):
        shap.TreeExplainer(ada)
    ex = SHAPExplainer(ada, out["X_train"], out["X_test"], task_type="Classification")
    assert ex._get_model_type() == "kernel"
    res = ex.explain(max_rows=5, time_budget_s=30)
    assert res.status in ("ok", "budget_exceeded") and res.explainer == "KernelExplainer"
    # a non-tree model whose class name contains 'forest' is not sent to TreeExplainer
    fake = type("RandomForestLookalike", (), {"predict": lambda self, X: np.zeros(len(X))})()
    assert SHAPExplainer(fake, out["X_train"], out["X_test"])._get_model_type() == "kernel"


def test_ensemble_and_hist_gradient_boosting(cls_models):
    p, out, tr = cls_models
    for name in ("Voting Ensemble", "Hist Gradient Boosting"):
        ex = SHAPExplainer(tr.trained_models[name], out["X_train"], out["X_test"], task_type="Classification")
        res = ex.explain(max_rows=3, time_budget_s=30)
        assert res.status in ("ok", "budget_exceeded"), (name, res.error)


def test_only_bool_and_object_columns_are_cast():
    X = pd.DataFrame({"f32": np.ones(3, dtype=np.float32), "i": [1, 2, 3], "b": [True, False, True],
                      "o": np.array([1, 0, 1], dtype=object)})
    out = SHAPExplainer._ensure_dataframe(X)
    assert out.dtypes.to_dict() == {"f32": np.float32, "i": np.int64, "b": np.float64, "o": np.float64}


def test_xgboost_through_kernel_explainer(cls_models):
    """shap 0.46 cannot set XGBoost's read-only feature_names_in_; the kernel path wraps predict."""
    p, out, tr = cls_models

    class ForcedKernel(SHAPExplainer):
        def _get_model_type(self):
            return "kernel"

    ex = ForcedKernel(tr.trained_models["XGBoost"], out["X_train"], out["X_test"], task_type="Classification")
    res = ex.explain(max_rows=2, time_budget_s=60)
    assert res.status == "ok" and res.explainer == "KernelExplainer", res.error


def test_cancel_stops_kernel_run(cls_models):
    p, out, tr = cls_models
    ex = SHAPExplainer(tr.trained_models["KNN"], out["X_train"], out["X_test"], task_type="Classification")
    cancel = threading.Event()
    seen = []

    def progress(done, total):
        seen.append(done)
        if done >= 2:
            cancel.set()

    res = ex.explain(max_rows=50, time_budget_s=120, progress_cb=progress, cancel_event=cancel)
    assert res.status == "cancelled" and res.rows_done == 2 and seen == [1, 2]


def test_estimate_before_start(cls_models):
    p, out, tr = cls_models
    ex = SHAPExplainer(tr.trained_models["SVM"], out["X_train"], out["X_test"], task_type="Classification")
    est = ex.estimate_seconds_per_row()
    assert est is not None and est > 0
    assert ex.last_result.route == "kernel" and "KernelExplainer" in ex.route_reason()


def test_additivity_failure_is_retried_with_warning(reg_models):
    p, out, tr = reg_models
    ex = SHAPExplainer(tr.trained_models["Gradient Boosting"], out["X_train"], out["X_test"], task_type="Regression")
    assert ex.build_explainer()
    real = ex._explainer.shap_values
    calls = []

    def flaky(X, check_additivity=True, **kw):
        calls.append(check_additivity)
        if check_additivity:
            raise Exception("Additivity check failed in TreeExplainer!")
        return real(X, check_additivity=False)

    ex._explainer.shap_values = flaky
    res = ex.explain(max_rows=20, time_budget_s=30)
    assert res.status == "ok" and any("additivity" in w for w in res.warnings)
    assert calls[:2] == [True, False] and all(c is False for c in calls[2:])
