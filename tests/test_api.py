"""B-8: /predict and /predict/csv through FastAPI's TestClient on a workspace written by WorkspaceManager."""
import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def served(adult_df, tmp_path_factory):
    """A trained Adult workspace in a temporary store, plus the offline predictions for its test rows."""
    import app_backend.main_api as api
    import app_backend.workspace_manager as wsm
    from app_backend.model_trainer import ModelTrainer
    from app_backend.preprocessing_engine.engine import AutoPreprocessor

    root = tmp_path_factory.mktemp("store")
    saved = (wsm.WORKSPACE_DIR, wsm.DATA_DIR, wsm.UPLOADS_DIR, wsm.WORKSPACE_INDEX)
    wsm.WORKSPACE_DIR, wsm.DATA_DIR = str(root / "ws"), str(root / "data")
    wsm.UPLOADS_DIR, wsm.WORKSPACE_INDEX = str(root / "data" / "up"), str(root / "ws" / "index.json")
    api.CACHE.clear()

    p = AutoPreprocessor(target_col="class", verbose=False, random_state=0)
    out = p.fit_transform(df=adult_df)
    tr = ModelTrainer(adult_df, "class", p.task_type, random_state=0)
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    tr.run_selected_models(["XGBoost", "Logistic Regression"])
    wm = wsm.WorkspaceManager()
    ws = wm.create_workspace("adult.csv", adult_df.shape)
    ws.target_col, ws.task_type = "class", p.task_type
    wm.save_workspace(ws)
    wm.save_preprocessor(ws.workspace_id, p)
    wm.save_trained_models(ws.workspace_id, tr.trained_models)
    raw_test = adult_df.loc[out["X_test"].index]
    offline = {m: p.decode_target(tr.trained_models[m].predict(out["X_test"])) for m in tr.trained_models}
    yield TestClient(api.app), ws.workspace_id, raw_test, offline, api
    wsm.WORKSPACE_DIR, wsm.DATA_DIR, wsm.UPLOADS_DIR, wsm.WORKSPACE_INDEX = saved
    api.CACHE.clear()


def _rows(df):
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
            for r in df.to_dict(orient="records")]


def _post(client, ws_id, rows, model="XGBoost"):
    return client.post("/predict", json={"workspace_id": ws_id, "model_name": model, "data": _rows(rows)})


def test_unperturbed_rows_match_offline_predictions(served):
    client, ws_id, raw, offline, _ = served
    rows = raw.drop(columns=["class"]).head(64)
    r = _post(client, ws_id, rows)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predictions"] == list(offline["XGBoost"][:64])
    assert set(body["predictions"]) <= {"<=50K", ">50K"}          # original labels, not codes
    assert body["class_names"] == ["<=50K", ">50K"]
    assert abs(sum(body["probabilities"][0].values()) - 1) < 1e-6


def test_target_column_in_request_is_ignored(served):
    client, ws_id, raw, offline, _ = served
    r = _post(client, ws_id, raw.head(8))
    assert r.status_code == 200 and r.json()["predictions"] == list(offline["XGBoost"][:8])


def test_reordered_columns(served):
    client, ws_id, raw, offline, _ = served
    rows = raw.drop(columns=["class"]).head(16)
    r = _post(client, ws_id, rows[rows.columns[::-1]], model="Logistic Regression")
    assert r.status_code == 200 and r.json()["predictions"] == list(offline["Logistic Regression"][:16])


def test_unseen_category_returns_200_with_warning(served):
    client, ws_id, raw, _, _ = served
    rows = raw.drop(columns=["class"]).head(4).assign(workclass="Space-agency")
    r = _post(client, ws_id, rows)
    assert r.status_code == 200 and any(w.startswith("workclass") for w in r.json()["warnings"])


def test_missing_column_returns_422(served):
    client, ws_id, raw, _, _ = served
    r = _post(client, ws_id, raw.drop(columns=["class", "age"]).head(4))
    assert r.status_code == 422
    assert r.json()["detail"]["missing_columns"] == ["age"]


def test_invalid_numeric_value_returns_422(served):
    client, ws_id, raw, _, _ = served
    r = _post(client, ws_id, raw.drop(columns=["class"]).head(2).assign(age="forty"))
    assert r.status_code == 422 and "age" in r.json()["detail"]["invalid_columns"]


def test_extra_column_is_ignored_with_warning(served):
    client, ws_id, raw, offline, _ = served
    r = _post(client, ws_id, raw.drop(columns=["class"]).head(4).assign(customer_id=123))
    assert r.status_code == 200 and r.json()["predictions"] == list(offline["XGBoost"][:4])
    assert any("customer_id" in w for w in r.json()["warnings"])


def test_empty_body_and_unknown_names(served):
    client, ws_id, raw, _, _ = served
    assert client.post("/predict", json={"workspace_id": ws_id, "model_name": "XGBoost", "data": []}).status_code == 422
    assert _post(client, "nope", raw.head(1)).status_code == 404
    r = _post(client, ws_id, raw.head(1), model="SVM")
    assert r.status_code == 404 and "Available" in r.json()["detail"]


def test_single_row_and_missing_values(served):
    client, ws_id, raw, _, _ = served
    row = raw.drop(columns=["class"]).head(1).assign(occupation=None, age=None)
    r = _post(client, ws_id, row)
    assert r.status_code == 200 and len(r.json()["predictions"]) == 1
    assert any("filled" in w for w in r.json()["warnings"])


def test_schema_endpoint(served):
    client, ws_id, _, _, _ = served
    s = client.get(f"/workspaces/{ws_id}/schema").json()
    assert "age" in s["required_columns"] and s["columns"]["age"]["kind"] == "numeric"
    assert s["class_names"] == ["<=50K", ">50K"]


def test_predict_csv(served):
    client, ws_id, raw, offline, _ = served
    buf = io.StringIO()
    raw.drop(columns=["class"]).head(10).to_csv(buf, index=False)
    r = client.post(f"/predict/csv/{ws_id}/XGBoost", files={"file": ("x.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    res = pd.DataFrame(r.json()["results"])
    assert list(res["prediction"]) == list(offline["XGBoost"][:10])
    assert {"proba_<=50K", "proba_>50K"} <= set(res.columns)
    bad = raw.drop(columns=["class", "education"]).head(3).to_csv(index=False)
    assert client.post(f"/predict/csv/{ws_id}/XGBoost", files={"file": ("x.csv", bad, "text/csv")}).status_code == 422


def test_artifacts_are_cached(served):
    client, ws_id, raw, _, api = served
    rows = raw.drop(columns=["class"]).head(1)
    _post(client, ws_id, rows)
    hits = api.CACHE.hits
    _post(client, ws_id, rows)
    assert api.CACHE.hits == hits + 1


def test_warm_single_row_latency(served, record_property):
    """Measured and reported; the target is < 15 ms median for XGBoost in-process (warm cache)."""
    client, ws_id, raw, _, _ = served
    rows = [raw.drop(columns=["class"]).iloc[[i]] for i in range(40)]
    _post(client, ws_id, rows[0])
    times, server = [], []
    for r in rows:
        t0 = time.perf_counter()
        resp = _post(client, ws_id, r)
        times.append((time.perf_counter() - t0) * 1000)
        server.append(resp.json()["timing_ms"]["total"])
    median, server_median = float(np.median(times)), float(np.median(server))
    record_property("predict_latency_median_ms", median)
    record_property("predict_server_latency_median_ms", server_median)
    print(f"\nwarm /predict single row: client median {median:.2f} ms (includes TestClient overhead), "
          f"server-side median {server_median:.2f} ms, client p95 {np.percentile(times, 95):.2f} ms")
    assert server_median < 15.0   # in-process: request parsing, transform, predict, response building
    assert median < 30.0          # generous guard on the full TestClient round trip
