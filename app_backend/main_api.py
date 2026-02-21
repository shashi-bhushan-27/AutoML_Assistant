"""
FastAPI Backend — AutoML Assistant
Exposes a /predict endpoint so trained models can be used for inference from any client.
Also exposes a /health endpoint for monitoring.
"""
import os
import sys
import io
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Path setup so we can import from app_backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_backend.workspace_manager import WorkspaceManager

# ---------------------------------------------------------------------------
# App Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AutoML Assistant API",
    description="REST API for serving trained AutoML models and predictions.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for single-row or batch prediction."""
    workspace_id: str
    model_name: str
    data: List[Dict[str, Any]]  # List of rows as dicts (column_name -> value)


class PredictResponse(BaseModel):
    workspace_id: str
    model_name: str
    predictions: List[Any]
    prediction_labels: Optional[List[Any]] = None  # For classification
    probabilities: Optional[List[List[float]]] = None  # For classification models


class WorkspaceSummaryResponse(BaseModel):
    workspace_id: str
    dataset_name: str
    task_type: Optional[str]
    best_model: Optional[str]
    best_score: Optional[float]
    status: str
    available_models: List[str]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_model_and_preprocessor(workspace_id: str, model_name: str):
    """Load trained model + preprocessor for a given workspace."""
    wm = WorkspaceManager()

    # Load workspace
    ws = wm.load_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    # Load trained models dict
    trained_models = wm.load_trained_models(workspace_id)
    if trained_models is None or model_name not in trained_models:
        available = list((trained_models or {}).keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in workspace. Available: {available}"
        )

    model = trained_models[model_name]

    # Load preprocessor (optional — may not exist for all workspaces)
    preprocessor = wm.load_preprocessor(workspace_id)

    return model, preprocessor, ws


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Returns API health status."""
    return {"status": "ok", "version": "2.0.0", "service": "AutoML Assistant API"}


@app.get("/workspaces", response_model=List[WorkspaceSummaryResponse], tags=["Workspaces"])
def list_workspaces():
    """List all available workspaces and their trained models."""
    wm = WorkspaceManager()
    all_ws = wm.list_workspaces()

    result = []
    for ws_data in all_ws:
        ws_id = ws_data.get("workspace_id")
        trained_models = wm.load_trained_models(ws_id) or {}
        result.append(WorkspaceSummaryResponse(
            workspace_id=ws_id,
            dataset_name=ws_data.get("dataset_name", "Unknown"),
            task_type=ws_data.get("task_type"),
            best_model=ws_data.get("best_model"),
            best_score=ws_data.get("best_score"),
            status=ws_data.get("status", "unknown"),
            available_models=list(trained_models.keys()),
        ))
    return result


@app.get("/workspaces/{workspace_id}/models", tags=["Workspaces"])
def get_workspace_models(workspace_id: str):
    """Get all available trained model names for a workspace."""
    wm = WorkspaceManager()
    trained_models = wm.load_trained_models(workspace_id) or {}
    ws = wm.load_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    return {
        "workspace_id": workspace_id,
        "dataset_name": ws.dataset_name,
        "task_type": ws.task_type,
        "best_model": ws.best_model,
        "available_models": list(trained_models.keys()),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Run inference using a trained model from a workspace.

    Send a list of rows as dicts. The API will apply the saved preprocessor
    (if available) and return predictions.

    Example request body:
    {
      "workspace_id": "abc123",
      "model_name": "XGBoost",
      "data": [
        {"feature1": 1.2, "feature2": "cat_A", "feature3": 5},
        {"feature1": 3.4, "feature2": "cat_B", "feature3": 2}
      ]
    }
    """
    model, preprocessor, ws = _load_model_and_preprocessor(
        request.workspace_id, request.model_name
    )

    # Convert input data to DataFrame
    try:
        input_df = pd.DataFrame(request.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data format: {e}")

    # Remove target column if accidentally included
    if ws.target_col and ws.target_col in input_df.columns:
        input_df = input_df.drop(columns=[ws.target_col])

    # Apply preprocessor transform (not fit!) if available
    X = input_df
    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(input_df)
            if isinstance(X_transformed, dict):
                # AutoPreprocessor returns a dict; extract the right split or just use X_test
                # For inference, we only use transform on input
                X = input_df  # Fall back to raw if transform returns unexpected type
            else:
                X = X_transformed
        except Exception as e:
            # Preprocessor transform failed — try raw data (some models handle it)
            print(f"[API] Preprocessor transform failed: {e}. Using raw data.")
            X = input_df

    # Run prediction
    try:
        raw_preds = model.predict(X)
        predictions = raw_preds.tolist() if hasattr(raw_preds, 'tolist') else list(raw_preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Classification: also return class labels and probabilities
    probabilities = None
    prediction_labels = None

    if ws.task_type and ws.task_type.lower() == "classification":
        prediction_labels = predictions  # Already class labels
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(X)
                probabilities = probas.tolist()
            except Exception:
                pass

    return PredictResponse(
        workspace_id=request.workspace_id,
        model_name=request.model_name,
        predictions=predictions,
        prediction_labels=prediction_labels,
        probabilities=probabilities,
    )


@app.post("/predict/csv/{workspace_id}/{model_name}", tags=["Inference"])
async def predict_csv(workspace_id: str, model_name: str, file: UploadFile = File(...)):
    """
    Run batch inference by uploading a CSV file.
    Returns predictions appended to the original data as JSON.
    """
    model, preprocessor, ws = _load_model_and_preprocessor(workspace_id, model_name)

    # Read uploaded CSV
    try:
        contents = await file.read()
        input_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Remove target column if present
    if ws.target_col and ws.target_col in input_df.columns:
        input_df = input_df.drop(columns=[ws.target_col])

    X = input_df
    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(input_df)
            if not isinstance(X_transformed, dict):
                X = X_transformed
        except Exception as e:
            print(f"[API] Preprocessor transform failed: {e}. Using raw data.")

    try:
        raw_preds = model.predict(X)
        predictions = raw_preds.tolist() if hasattr(raw_preds, 'tolist') else list(raw_preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Build result
    result_df = input_df.copy()
    result_df["prediction"] = predictions

    if ws.task_type and ws.task_type.lower() == "classification" and hasattr(model, 'predict_proba'):
        try:
            probas = model.predict_proba(X)
            classes = model.classes_ if hasattr(model, 'classes_') else list(range(probas.shape[1]))
            for i, cls in enumerate(classes):
                result_df[f"proba_{cls}"] = probas[:, i]
        except Exception:
            pass

    return JSONResponse(content={
        "workspace_id": workspace_id,
        "model_name": model_name,
        "rows_processed": len(result_df),
        "results": result_df.to_dict(orient="records"),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
