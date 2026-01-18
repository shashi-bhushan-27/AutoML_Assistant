"""
Workspace Manager Module
Handles saving, loading, and managing analysis workspaces.
"""
import os
import json
import uuid
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# Storage paths
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspaces")
WORKSPACE_INDEX = os.path.join(WORKSPACE_DIR, "index.json")


def ensure_workspace_dir():
    """Ensure workspace directory exists."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    if not os.path.exists(WORKSPACE_INDEX):
        with open(WORKSPACE_INDEX, 'w') as f:
            json.dump({"workspaces": []}, f)


class Workspace:
    """Represents a single analysis workspace."""
    
    def __init__(self, workspace_id: str = None):
        self.workspace_id = workspace_id or str(uuid.uuid4())[:8]
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.status = "in-progress"
        
        # Dataset metadata
        self.dataset_name: str = None
        self.dataset_shape: tuple = None
        self.target_col: str = None
        self.task_type: str = None
        
        # Timeline of actions
        self.timeline: List[Dict[str, Any]] = []
        
        # Analysis artifacts
        self.profile_summary: Dict = {}
        self.preprocessing_steps: List[Dict] = []
        self.recommendations: List[str] = []
        self.model_results: Dict = {}
        self.best_model: str = None
        self.best_score: float = None
        
        # User selections
        self.user_config: Dict = {}
    
    def add_event(self, event_type: str, description: str, metadata: Dict = None):
        """Add event to timeline."""
        self.timeline.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "description": description,
            "metadata": metadata or {}
        })
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "dataset_name": self.dataset_name,
            "dataset_shape": self.dataset_shape,
            "target_col": self.target_col,
            "task_type": self.task_type,
            "timeline": self.timeline,
            "profile_summary": self.profile_summary,
            "preprocessing_steps": self.preprocessing_steps,
            "recommendations": self.recommendations,
            "model_results": self.model_results,
            "best_model": self.best_model,
            "best_score": self.best_score,
            "user_config": self.user_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Workspace':
        """Create workspace from dictionary."""
        ws = cls(workspace_id=data.get("workspace_id"))
        ws.created_at = data.get("created_at")
        ws.updated_at = data.get("updated_at")
        ws.status = data.get("status", "in-progress")
        ws.dataset_name = data.get("dataset_name")
        ws.dataset_shape = tuple(data.get("dataset_shape", []))
        ws.target_col = data.get("target_col")
        ws.task_type = data.get("task_type")
        ws.timeline = data.get("timeline", [])
        ws.profile_summary = data.get("profile_summary", {})
        ws.preprocessing_steps = data.get("preprocessing_steps", [])
        ws.recommendations = data.get("recommendations", [])
        ws.model_results = data.get("model_results", {})
        ws.best_model = data.get("best_model")
        ws.best_score = data.get("best_score")
        ws.user_config = data.get("user_config", {})
        return ws
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workspace summary for list display."""
        return {
            "workspace_id": self.workspace_id,
            "dataset_name": self.dataset_name or "Untitled",
            "created_at": self.created_at,
            "task_type": self.task_type or "Unknown",
            "status": self.status,
            "best_model": self.best_model,
            "best_score": self.best_score
        }


class WorkspaceManager:
    """Manages all workspaces."""
    
    def __init__(self):
        ensure_workspace_dir()
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load workspace index."""
        try:
            with open(WORKSPACE_INDEX, 'r') as f:
                return json.load(f)
        except:
            return {"workspaces": []}
    
    def _save_index(self):
        """Save workspace index."""
        with open(WORKSPACE_INDEX, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def create_workspace(self, dataset_name: str = None, dataset_shape: tuple = None) -> Workspace:
        """Create new workspace."""
        ws = Workspace()
        ws.dataset_name = dataset_name
        ws.dataset_shape = dataset_shape
        ws.add_event("workspace_created", "New workspace initialized")
        
        # Add to index
        self.index["workspaces"].append({
            "workspace_id": ws.workspace_id,
            "created_at": ws.created_at,
            "dataset_name": dataset_name
        })
        self._save_index()
        
        # Save workspace
        self.save_workspace(ws)
        return ws
    
    def save_workspace(self, workspace: Workspace):
        """Save workspace to disk."""
        workspace.updated_at = datetime.now().isoformat()
        ws_path = os.path.join(WORKSPACE_DIR, f"{workspace.workspace_id}.json")
        with open(ws_path, 'w') as f:
            json.dump(workspace.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        # Update index
        for entry in self.index["workspaces"]:
            if entry["workspace_id"] == workspace.workspace_id:
                entry["dataset_name"] = workspace.dataset_name
                entry["status"] = workspace.status
                entry["task_type"] = workspace.task_type
                entry["best_model"] = workspace.best_model
                break
        self._save_index()
    
    def load_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Load workspace from disk."""
        ws_path = os.path.join(WORKSPACE_DIR, f"{workspace_id}.json")
        if os.path.exists(ws_path):
            with open(ws_path, 'r') as f:
                data = json.load(f)
            return Workspace.from_dict(data)
        return None
    
    def list_workspaces(self) -> List[Dict]:
        """List all workspaces with summaries."""
        workspaces = []
        for entry in self.index.get("workspaces", []):
            ws = self.load_workspace(entry["workspace_id"])
            if ws:
                workspaces.append(ws.get_summary())
        # Sort by created_at descending
        workspaces.sort(key=lambda x: x["created_at"], reverse=True)
        return workspaces
    
    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace."""
        ws_path = os.path.join(WORKSPACE_DIR, f"{workspace_id}.json")
        if os.path.exists(ws_path):
            os.remove(ws_path)
        
        self.index["workspaces"] = [
            e for e in self.index["workspaces"] 
            if e["workspace_id"] != workspace_id
        ]
        self._save_index()
        return True
    
    def save_preprocessor(self, workspace_id: str, preprocessor):
        """Save preprocessor pipeline for a workspace."""
        pipeline_path = os.path.join(WORKSPACE_DIR, f"{workspace_id}_pipeline.pkl")
        with open(pipeline_path, 'wb') as f:
            pickle.dump(preprocessor, f)
    
    def load_preprocessor(self, workspace_id: str):
        """Load preprocessor pipeline for a workspace."""
        pipeline_path = os.path.join(WORKSPACE_DIR, f"{workspace_id}_pipeline.pkl")
        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_dataset(self, workspace_id: str, df: pd.DataFrame):
        """Save dataset for a workspace."""
        if df is not None:
            data_path = os.path.join(WORKSPACE_DIR, f"{workspace_id}_data.csv")
            df.to_csv(data_path, index=False)
    
    def load_dataset(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Load dataset for a workspace."""
        data_path = os.path.join(WORKSPACE_DIR, f"{workspace_id}_data.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        return None
    
    def save_session_state(self, workspace_id: str, state_data: Dict):
        """Save session state (stats, recommendations, etc.)."""
        state_path = os.path.join(WORKSPACE_DIR, f"{workspace_id}_state.pkl")
        with open(state_path, 'wb') as f:
            pickle.dump(state_data, f)
    
    def load_session_state(self, workspace_id: str) -> Optional[Dict]:
        """Load session state for a workspace."""
        state_path = os.path.join(WORKSPACE_DIR, f"{workspace_id}_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                return pickle.load(f)
        return None
