"""B-5 (task-type casing / best model for regression), B-6 (dataset_shape=None), B-11 (meta-learning),
B-19 (artefact checksums), B-17 (analyze_dataset does not mutate its input)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from app_backend import meta_learning
from app_backend.leaderboard import choose_primary_metric, select_best_model
from app_backend.statistical_engine import analyze_dataset
from app_backend.task_types import TaskType, detect_task_type, normalize_task_type


@pytest.mark.parametrize("value,expected", [("regression", TaskType.REGRESSION), ("Regression", TaskType.REGRESSION),
                                            ("REGRESSION", TaskType.REGRESSION),
                                            ("classification", TaskType.CLASSIFICATION),
                                            ("Classification", TaskType.CLASSIFICATION),
                                            (TaskType.REGRESSION, TaskType.REGRESSION), ("auto", None), (None, None)])
def test_normalize_task_type(value, expected):
    assert normalize_task_type(value) == expected


def test_single_cardinality_threshold():
    """The statistical engine and the preprocessor used < 10 vs <= 10; both now use detect_task_type."""
    ten = pd.Series(np.arange(100) % 10)
    eleven = pd.Series(np.arange(110) % 11)
    assert detect_task_type(ten) == TaskType.CLASSIFICATION
    assert detect_task_type(eleven) == TaskType.REGRESSION
    assert analyze_dataset(pd.DataFrame({"x": np.arange(100), "y": ten}), "y")["task_type"] == "Classification"


RESULTS = pd.DataFrame([{"Model": "Ridge", "RMSE": 0.70, "R²": 0.60},
                        {"Model": "XGBoost", "RMSE": 0.46, "R²": 0.84},
                        {"Model": "SVM", "Error": "boom"}])


@pytest.mark.parametrize("casing", ["regression", "Regression"])
def test_best_model_recorded_for_regression_in_any_casing(casing):
    """B-5: the UI compared with lower-case 'regression' while the workspace held 'Regression'."""
    best = select_best_model(RESULTS, casing)
    assert best["model"] == "XGBoost" and best["metric"] == "RMSE" and best["score"] == pytest.approx(0.46)


def test_imbalanced_classification_uses_f1():
    assert choose_primary_metric("Classification", 0.02)["metric"] == "F1 Score"
    assert choose_primary_metric("classification", 0.30)["metric"] == "Accuracy"


def test_workspace_with_none_shape_round_trips(temp_workspace):
    """B-6: create_workspace(dataset_shape=None) used to write JSON that could not be loaded."""
    wm = temp_workspace.WorkspaceManager()
    ws = wm.create_workspace(dataset_name="x.csv", dataset_shape=None)
    loaded = wm.load_workspace(ws.workspace_id)
    assert loaded is not None and loaded.dataset_shape == ()
    assert temp_workspace.Workspace.from_dict({"workspace_id": "a", "dataset_shape": None}).dataset_shape == ()


def test_regression_workspace_stores_best_model(temp_workspace):
    wm = temp_workspace.WorkspaceManager()
    ws = wm.create_workspace("cal.csv", (100, 9))
    ws.task_type = "regression"            # any casing is normalised on load
    best = select_best_model(RESULTS, ws.task_type)
    ws.best_model, ws.best_score, ws.best_metric = best["model"], best["score"], best["metric"]
    wm.save_workspace(ws)
    again = wm.load_workspace(ws.workspace_id)
    assert again.best_model == "XGBoost" and again.task_type == "Regression"


def _prior(wm, stats, best="XGBoost"):
    ws = wm.create_workspace("prior.csv", (stats["rows"], stats["columns"]))
    ws.task_type, ws.profile_summary = stats["task_type"], stats
    ws.best_model, ws.best_score, ws.best_metric = best, 0.5, "RMSE"
    ws.status = "completed"
    wm.save_workspace(ws)
    return ws


@pytest.mark.parametrize("fixture,target", [("california_df", "MedHouseVal"), ("adult_df", "class")])
def test_meta_learning_retrieves_same_dataset_prior(request, temp_workspace, fixture, target):
    """B-11: a prior workspace on the same dataset matches (similarity 1.0), for regression too."""
    df = request.getfixturevalue(fixture)
    stats = analyze_dataset(df, target)
    wm = temp_workspace.WorkspaceManager()
    prior = _prior(wm, stats)
    matches = wm.find_similar_workspaces(stats)
    assert [m["workspace_id"] for m in matches] == [prior.workspace_id]
    assert matches[0]["similarity"] == pytest.approx(1.0) and "similarity" in matches[0]["reason"]
    assert wm.find_similar_workspaces(stats, exclude_id=prior.workspace_id) == []


def test_meta_learning_rejects_dissimilar_and_other_task(temp_workspace, adult_df, credit_like_df, california_df):
    wm = temp_workspace.WorkspaceManager()
    _prior(wm, analyze_dataset(credit_like_df, "Class"))
    _prior(wm, analyze_dataset(california_df, "MedHouseVal"))
    adult = analyze_dataset(adult_df, "class")
    assert wm.find_similar_workspaces(adult) == []
    sim = meta_learning.similarity(adult["profile"], analyze_dataset(credit_like_df, "Class")["profile"])
    assert sim < meta_learning.SIMILARITY_THRESHOLD


def test_meta_learning_uses_index_only(temp_workspace, california_df):
    stats = analyze_dataset(california_df, "MedHouseVal")
    wm = temp_workspace.WorkspaceManager()
    prior = _prior(wm, stats)
    os.remove(os.path.join(temp_workspace.WORKSPACE_DIR, f"{prior.workspace_id}.json"))
    assert len(temp_workspace.WorkspaceManager().find_similar_workspaces(stats)) == 1


def test_artifact_checksum_detects_tampering(temp_workspace):
    wm = temp_workspace.WorkspaceManager()
    ws = wm.create_workspace("x.csv", (1, 1))
    wm.save_trained_models(ws.workspace_id, {"m": {"weights": [1, 2, 3]}})
    assert wm.load_trained_models(ws.workspace_id) == {"m": {"weights": [1, 2, 3]}}
    with open(wm.artifact_path(ws.workspace_id, "trained_models"), "ab") as f:
        f.write(b"tampered")
    with pytest.raises(temp_workspace.ArtifactIntegrityError):
        wm.load_trained_models(ws.workspace_id)


def test_delete_removes_files(temp_workspace):
    wm = temp_workspace.WorkspaceManager()
    ws = wm.create_workspace("x.csv", (1, 1))
    wm.save_session_state(ws.workspace_id, {"a": 1})
    assert wm.delete_workspace(ws.workspace_id) == []
    assert wm.load_workspace(ws.workspace_id) is None
    assert not any(f.startswith(ws.workspace_id) for f in os.listdir(temp_workspace.WORKSPACE_DIR))
    with open(temp_workspace.WORKSPACE_INDEX) as f:
        assert json.load(f)["workspaces"] == []


def test_analyze_dataset_does_not_mutate_input():
    """B-17: a date column used to be converted in place."""
    df = pd.DataFrame({"order_date": pd.date_range("2024-01-01", periods=60).astype(str),
                       "value": np.arange(60.0), "y": np.arange(60) * 2.0})
    before = df.copy()
    stats = analyze_dataset(df, "y")
    pd.testing.assert_frame_equal(df, before)
    assert stats["is_time_series"] and stats["time_column"] == "order_date"
