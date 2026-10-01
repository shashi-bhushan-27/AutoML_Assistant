"""
FastAPI inference server for trained workspaces.

* Requests are validated against the pipeline's training schema: a missing required
  column or a non-numeric value in a numeric column returns HTTP 422 naming the columns.
  Extra columns are ignored; unseen categories are handled and reported in ``warnings``.
* There is no fallback to raw, untransformed input (it used to be silently passed to the
  model after a transform error).
* Loaded pipelines/models are cached per workspace and model, keyed on the files'
  modification times, instead of being unpickled on every request.
* Classification responses contain the original class labels and per-class probabilities
  keyed by class name.

Workspaces are pickles: only serve workspaces you trust (see workspace_manager.py).
"""
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_backend import model_registry  # noqa: E402
from app_backend.preprocessing_engine.engine import PIPELINE_VERSION, SchemaError  # noqa: E402
from app_backend.task_types import is_classification  # noqa: E402
import app_backend.workspace_manager as wsm  # noqa: E402

logger = logging.getLogger(__name__)
API_VERSION = "3.0.0"
MAX_ROWS_PER_REQUEST = 10_000
NON_SERVABLE = set(model_registry.TIME_SERIES_MODELS + ["LSTM"])

app = FastAPI(title="AutoML Assistant API", version=API_VERSION,
              description="Serves models trained in AutoML Assistant workspaces. Interactive docs: /docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── schemas ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    workspace_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    data: List[Dict[str, Any]] = Field(min_length=1, max_length=MAX_ROWS_PER_REQUEST,
                                       description="Rows as {column: value}; see GET /workspaces/{id}/schema")


class PredictResponse(BaseModel):
    workspace_id: str
    model_name: str
    task_type: Optional[str]
    n_rows: int
    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None
    class_names: Optional[List[str]] = None
    warnings: List[str] = []
    timing_ms: Dict[str, float] = {}


class WorkspaceSummaryResponse(BaseModel):
    workspace_id: str
    dataset_name: Optional[str]
    task_type: Optional[str]
    best_model: Optional[str]
    best_score: Optional[float]
    status: str
    available_models: List[str]


# ── artefact cache ──────────────────────────────────────────────────────────
class _Loaded:
    def __init__(self, ws, pipeline, model, model_names):
        self.ws, self.pipeline, self.model, self.model_names = ws, pipeline, model, model_names


class ArtifactCache:
    """LRU of (workspace, model) -> loaded objects, invalidated when a file's mtime changes."""

    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self._data: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def _stamp(workspace_id: str) -> tuple:
        paths = [os.path.join(wsm.WORKSPACE_DIR, f"{workspace_id}.json"),
                 os.path.join(wsm.WORKSPACE_DIR, f"{workspace_id}_pipeline.pkl"),
                 os.path.join(wsm.WORKSPACE_DIR, f"{workspace_id}_trained_models.pkl")]
        return tuple(os.path.getmtime(p) if os.path.exists(p) else None for p in paths)

    def get(self, workspace_id: str, model_name: str) -> _Loaded:
        key = (workspace_id, model_name)
        stamp = self._stamp(workspace_id)
        with self._lock:
            hit = self._data.get(key)
            if hit and hit[0] == stamp:
                self._data.move_to_end(key)
                self.hits += 1
                return hit[1]
        self.misses += 1
        loaded = _load(workspace_id, model_name)
        with self._lock:
            self._data[key] = (stamp, loaded)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
        return loaded

    def clear(self):
        with self._lock:
            self._data.clear()


def _load(workspace_id: str, model_name: str) -> _Loaded:
    wm = wsm.WorkspaceManager()
    ws = wm.load_workspace(workspace_id)
    if ws is None:
        raise HTTPException(404, f"Workspace '{workspace_id}' not found.")
    try:
        models = wm.load_trained_models(workspace_id) or {}
        pipeline = wm.load_preprocessor(workspace_id)
    except wsm.ArtifactIntegrityError as exc:
        raise HTTPException(409, str(exc))
    if model_name not in models:
        raise HTTPException(404, f"Model '{model_name}' not found in workspace. Available: {sorted(models)}")
    if model_name in NON_SERVABLE:
        raise HTTPException(400, f"'{model_name}' is a time-series model and is not served by /predict.")
    if pipeline is None:
        raise HTTPException(409, "This workspace has no fitted preprocessing pipeline; run the Prepare step.")
    if getattr(pipeline, "pipeline_version", 1) != PIPELINE_VERSION or not getattr(pipeline, "feature_names_", None):
        raise HTTPException(409, "This workspace was prepared by an older version of the app; "
                                 "re-run Prepare and Train to serve it.")
    model = models[model_name]
    if type(model).__module__.startswith("xgboost"):
        # Request handlers run in a thread pool; an all-cores OpenMP pool per worker thread made single-row
        # predictions ~3x slower. One thread per request gives the same predictions.
        model.set_params(n_jobs=1)
    return _Loaded(ws, pipeline, model, sorted(models))


CACHE = ArtifactCache()


def _jsonable(values) -> List[Any]:
    return [v.item() if isinstance(v, np.generic) else v for v in np.asarray(values, dtype=object).tolist()]


def _run(loaded: _Loaded, df: pd.DataFrame) -> Dict[str, Any]:
    pipe, model = loaded.pipeline, loaded.model
    t0 = time.perf_counter()
    try:
        X, warnings = pipe.transform(df, return_warnings=True)
    except SchemaError as exc:
        raise HTTPException(422, {"message": str(exc), "missing_columns": exc.missing,
                                  "invalid_columns": exc.invalid,
                                  "required_columns": pipe.required_columns_})
    t1 = time.perf_counter()
    # XGBoost spends ~4 ms validating a DataFrame per call; the pipeline already guarantees the training
    # column order, so it gets the plain array (identical predictions).
    X_in = X.to_numpy() if type(model).__module__.startswith("xgboost") else X
    try:
        raw = model.predict(X_in)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction failed: {type(exc).__name__}: {exc}")
    t2 = time.perf_counter()
    task = pipe.task_type
    out: Dict[str, Any] = {"warnings": warnings, "class_names": None, "probabilities": None}
    if is_classification(task) and not pipe.is_multi_output:
        out["predictions"] = _jsonable(pipe.decode_target(raw))
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_in)
                names = [str(c) for c in pipe.decode_target(getattr(model, "classes_", np.arange(proba.shape[1])))]
                out["class_names"] = names
                out["probabilities"] = [{n: float(p) for n, p in zip(names, row)} for row in proba]
            except (AttributeError, NotImplementedError):
                pass
    elif is_classification(task):  # multi-output classification: decode each target column
        arr = np.asarray(raw)
        cols = [pipe.decode_target(arr[:, i], column=c) for i, c in enumerate(pipe.target_cols)]
        out["predictions"] = [dict(zip(pipe.target_cols, _jsonable(r))) for r in zip(*cols)]
    else:
        arr = np.asarray(raw, dtype=float)
        out["predictions"] = ([dict(zip(pipe.target_cols, r)) for r in arr.tolist()] if arr.ndim > 1
                              else arr.tolist())
    out["timing_ms"] = {"transform": round((t1 - t0) * 1000, 3), "predict": round((t2 - t1) * 1000, 3)}
    return out


# ── routes ──────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "version": API_VERSION, "service": "AutoML Assistant API",
            "cache": {"entries": len(CACHE._data), "hits": CACHE.hits, "misses": CACHE.misses}}


@app.get("/workspaces", response_model=List[WorkspaceSummaryResponse], tags=["Workspaces"])
def list_workspaces():
    wm = wsm.WorkspaceManager()
    out = []
    for w in wm.list_workspaces():
        try:
            models = wm.load_trained_models(w["workspace_id"]) or {}
        except wsm.ArtifactIntegrityError:
            models = {}
        out.append(WorkspaceSummaryResponse(workspace_id=w["workspace_id"], dataset_name=w.get("dataset_name"),
                                            task_type=w.get("task_type"), best_model=w.get("best_model"),
                                            best_score=w.get("best_score"), status=w.get("status", "unknown"),
                                            available_models=sorted(models)))
    return out


@app.get("/workspaces/{workspace_id}/models", tags=["Workspaces"])
def get_workspace_models(workspace_id: str):
    wm = wsm.WorkspaceManager()
    ws = wm.load_workspace(workspace_id)
    if ws is None:
        raise HTTPException(404, f"Workspace '{workspace_id}' not found.")
    models = wm.load_trained_models(workspace_id) or {}
    return {"workspace_id": workspace_id, "dataset_name": ws.dataset_name, "task_type": ws.task_type,
            "best_model": ws.best_model, "available_models": sorted(models)}


@app.get("/workspaces/{workspace_id}/schema", tags=["Workspaces"])
def get_workspace_schema(workspace_id: str):
    """Columns the /predict request needs, with kinds, dtypes and example values."""
    wm = wsm.WorkspaceManager()
    if wm.load_workspace(workspace_id) is None:
        raise HTTPException(404, f"Workspace '{workspace_id}' not found.")
    pipe = wm.load_preprocessor(workspace_id)
    if pipe is None or not getattr(pipe, "input_schema_", None):
        raise HTTPException(409, "No fitted pipeline for this workspace; run the Prepare step.")
    return {"workspace_id": workspace_id, "task_type": pipe.task_type, "target_columns": pipe.target_cols,
            "required_columns": pipe.required_columns_, "columns": pipe.input_schema_,
            "class_names": [str(c) for c in pipe.classes_] if pipe.classes_ is not None else None}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """Predict for a list of rows. See GET /workspaces/{workspace_id}/schema for the expected columns."""
    t0 = time.perf_counter()
    loaded = CACHE.get(request.workspace_id, request.model_name)
    df = pd.DataFrame(request.data)
    out = _run(loaded, df)
    out["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 3)
    return PredictResponse(workspace_id=request.workspace_id, model_name=request.model_name,
                           task_type=loaded.pipeline.task_type, n_rows=len(df), **out)


@app.post("/predict/csv/{workspace_id}/{model_name}", tags=["Inference"])
async def predict_csv(workspace_id: str, model_name: str, file: UploadFile = File(...)):
    """Batch prediction from an uploaded CSV; returns the rows with prediction (and probability) columns."""
    from app_backend.preprocessing_engine.ingestion import DataIngestor

    loaded = CACHE.get(workspace_id, model_name)
    try:
        df, info = DataIngestor().read_csv_bytes(await file.read())
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(400, f"Failed to read CSV: {exc}")
    if len(df) == 0:
        raise HTTPException(422, "The CSV has no rows.")
    out = _run(loaded, df)
    result = df.drop(columns=[c for c in loaded.pipeline.target_cols if c in df.columns]).copy()
    preds = out["predictions"]
    if preds and isinstance(preds[0], dict):
        for key in preds[0]:
            result[f"prediction_{key}"] = [p[key] for p in preds]
    else:
        result["prediction"] = preds
    if out["probabilities"]:
        for name in out["class_names"]:
            result[f"proba_{name}"] = [p[name] for p in out["probabilities"]]
    result = result.astype(object).where(result.notna(), None)
    return {"workspace_id": workspace_id, "model_name": model_name, "rows_processed": len(result),
            "warnings": out["warnings"] + info.get("warnings", []), "results": result.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_backend.main_api:app", host="0.0.0.0", port=8000)
