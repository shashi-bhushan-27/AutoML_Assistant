"""
Workspace Manager: workspace metadata (JSON), datasets (CSV) and artefacts (pickles).

Security note: model and pipeline artefacts are Python pickles. Loading a pickle can
execute code, so only load workspaces you created yourself or otherwise trust. Every
artefact is written with a SHA-256 checksum in ``<id>_manifest.json`` and verified on
load; a mismatch (file modified or corrupted outside the app) raises
``ArtifactIntegrityError`` instead of unpickling. The checksum detects accidental or
out-of-band modification, it is not a defence against someone who can rewrite both files.
"""
import hashlib
import json
import logging
import os
import pickle
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app_backend import meta_learning
from app_backend.task_types import normalize_task_type

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if not np.isfinite(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return str(obj)


class ArtifactIntegrityError(RuntimeError):
    pass


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# AUTOML_DATA_ROOT moves workspaces/ and data/uploads/ elsewhere (e.g. a Docker volume); default: the repo root
DATA_ROOT = os.environ.get("AUTOML_DATA_ROOT") or ROOT_DIR
WORKSPACE_DIR = os.path.join(DATA_ROOT, "workspaces")
DATA_DIR = os.path.join(DATA_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
WORKSPACE_INDEX = os.path.join(WORKSPACE_DIR, "index.json")

ARTIFACTS = ("state", "pipeline", "training_data", "trained_models")


def ensure_workspace_dir():
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    if not os.path.exists(WORKSPACE_INDEX):
        with open(WORKSPACE_INDEX, "w") as f:
            json.dump({"workspaces": []}, f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Workspace:
    def __init__(self, workspace_id: str = None):
        self.workspace_id = workspace_id or str(uuid.uuid4())[:8]
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.status = "in-progress"
        self.dataset_name: Optional[str] = None
        self.dataset_shape: tuple = ()
        self.dataset_hash: Optional[str] = None
        self.target_col = None
        self.task_type: Optional[str] = None
        self.timeline: List[Dict[str, Any]] = []
        self.profile_summary: Dict = {}
        self.preprocessing_steps: List[Dict] = []
        self.recommendations: List[str] = []
        self.model_results: Dict = {}
        self.best_model: Optional[str] = None
        self.best_score: Optional[float] = None
        self.best_metric: Optional[str] = None
        self.user_config: Dict = {}
        self.steps: Dict[str, Dict[str, Any]] = {}

    def add_event(self, event_type: str, description: str, metadata: Dict = None):
        self.timeline.append({"timestamp": datetime.now().isoformat(), "event": event_type,
                              "description": description, "metadata": metadata or {}})
        self.updated_at = datetime.now().isoformat()

    @property
    def seed(self) -> int:
        return int(self.user_config.get("seed", 42))

    @property
    def profile(self) -> Optional[Dict[str, float]]:
        p = (self.profile_summary or {}).get("profile")
        return p if meta_learning.valid_profile(p) else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id, "created_at": self.created_at, "updated_at": self.updated_at,
            "status": self.status, "dataset_name": self.dataset_name,
            "dataset_shape": list(self.dataset_shape) if self.dataset_shape else [],
            "dataset_hash": self.dataset_hash, "target_col": self.target_col, "task_type": self.task_type,
            "timeline": self.timeline, "profile_summary": self.profile_summary,
            "preprocessing_steps": self.preprocessing_steps, "recommendations": self.recommendations,
            "model_results": self.model_results, "best_model": self.best_model, "best_score": self.best_score,
            "best_metric": self.best_metric, "user_config": self.user_config, "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Workspace":
        ws = cls(workspace_id=data.get("workspace_id"))
        ws.created_at = data.get("created_at") or ws.created_at
        ws.updated_at = data.get("updated_at") or ws.created_at
        ws.status = data.get("status", "in-progress")
        ws.dataset_name = data.get("dataset_name")
        ws.dataset_shape = tuple(data.get("dataset_shape") or ())
        ws.dataset_hash = data.get("dataset_hash")
        ws.target_col = data.get("target_col")
        tt = normalize_task_type(data.get("task_type")) if data.get("task_type") else None
        ws.task_type = tt.value if tt else None
        ws.timeline = data.get("timeline") or []
        ws.profile_summary = data.get("profile_summary") or {}
        ws.preprocessing_steps = data.get("preprocessing_steps") or []
        ws.recommendations = data.get("recommendations") or []
        ws.model_results = data.get("model_results") or {}
        ws.best_model = data.get("best_model")
        ws.best_score = data.get("best_score")
        ws.best_metric = data.get("best_metric")
        ws.user_config = data.get("user_config") or {}
        ws.steps = data.get("steps") or {}
        return ws

    def get_summary(self) -> Dict[str, Any]:
        return {"workspace_id": self.workspace_id, "dataset_name": self.dataset_name or "Untitled",
                "created_at": self.created_at, "updated_at": self.updated_at,
                "task_type": self.task_type or "Unknown", "status": self.status,
                "best_model": self.best_model, "best_score": self.best_score, "best_metric": self.best_metric,
                "dataset_shape": list(self.dataset_shape) if self.dataset_shape else [],
                "target_col": self.target_col, "steps": self.steps}


class WorkspaceManager:
    def __init__(self):
        ensure_workspace_dir()
        self.index = self._load_index()

    # ── index ────────────────────────────────────────────────────────────────
    def _load_index(self) -> Dict:
        try:
            with open(WORKSPACE_INDEX) as f:
                data = json.load(f)
            return data if isinstance(data, dict) and "workspaces" in data else {"workspaces": []}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Workspace index unreadable (%s); starting empty", exc)
            return {"workspaces": []}

    def _save_index(self):
        tmp = WORKSPACE_INDEX + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.index, f, indent=2, cls=NumpyEncoder)
        os.replace(tmp, WORKSPACE_INDEX)

    def _index_entry(self, ws: Workspace) -> Dict[str, Any]:
        return {"workspace_id": ws.workspace_id, "created_at": ws.created_at, "updated_at": ws.updated_at,
                "dataset_name": ws.dataset_name, "status": ws.status, "task_type": ws.task_type,
                "best_model": ws.best_model, "best_score": ws.best_score, "best_metric": ws.best_metric,
                "profile": ws.profile}

    # ── workspaces ───────────────────────────────────────────────────────────
    def create_workspace(self, dataset_name: str = None, dataset_shape: tuple = None) -> Workspace:
        ws = Workspace()
        ws.dataset_name = dataset_name
        ws.dataset_shape = tuple(dataset_shape or ())
        ws.add_event("workspace_created", "New workspace initialized")
        self.index["workspaces"].append(self._index_entry(ws))
        self.save_workspace(ws)
        return ws

    def save_workspace(self, workspace: Workspace):
        workspace.updated_at = datetime.now().isoformat()
        path = os.path.join(WORKSPACE_DIR, f"{workspace.workspace_id}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(workspace.to_dict(), f, indent=2, cls=NumpyEncoder)
        os.replace(tmp, path)
        entry = self._index_entry(workspace)
        for i, e in enumerate(self.index["workspaces"]):
            if e.get("workspace_id") == workspace.workspace_id:
                self.index["workspaces"][i] = {**e, **entry}
                break
        else:
            self.index["workspaces"].append(entry)
        self._save_index()

    def load_workspace(self, workspace_id: str) -> Optional[Workspace]:
        path = os.path.join(WORKSPACE_DIR, f"{workspace_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return Workspace.from_dict(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Workspace %s unreadable: %s", workspace_id, exc)
            return None

    def list_workspaces(self) -> List[Dict]:
        out = []
        for entry in self.index.get("workspaces", []):
            ws = self.load_workspace(entry["workspace_id"])
            if ws:
                out.append(ws.get_summary())
        out.sort(key=lambda x: x.get("updated_at") or x["created_at"], reverse=True)
        return out

    def delete_workspace(self, workspace_id: str) -> List[str]:
        """Delete a workspace and its files; returns the files that could not be removed."""
        paths = [os.path.join(WORKSPACE_DIR, f"{workspace_id}.json"),
                 os.path.join(WORKSPACE_DIR, f"{workspace_id}_manifest.json"),
                 os.path.join(UPLOADS_DIR, f"{workspace_id}_data.csv"),
                 os.path.join(WORKSPACE_DIR, f"{workspace_id}_data.csv")]
        paths += [os.path.join(WORKSPACE_DIR, f"{workspace_id}_{a}.pkl") for a in ARTIFACTS]
        failed = []
        for p in paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError as exc:
                    logger.error("Could not delete %s: %s", p, exc)
                    failed.append(p)
        self.index["workspaces"] = [e for e in self.index["workspaces"] if e["workspace_id"] != workspace_id]
        self._save_index()
        return failed

    # ── artefacts with checksums ─────────────────────────────────────────────
    def _manifest_path(self, workspace_id):
        return os.path.join(WORKSPACE_DIR, f"{workspace_id}_manifest.json")

    def _read_manifest(self, workspace_id) -> Dict[str, str]:
        p = self._manifest_path(workspace_id)
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    def artifact_path(self, workspace_id: str, kind: str) -> str:
        return os.path.join(WORKSPACE_DIR, f"{workspace_id}_{kind}.pkl")

    def _save_artifact(self, workspace_id: str, kind: str, obj):
        path = self.artifact_path(workspace_id, kind)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        manifest = self._read_manifest(workspace_id)
        manifest[kind] = _sha256(path)
        with open(self._manifest_path(workspace_id), "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_artifact(self, workspace_id: str, kind: str, verify: bool = True):
        path = self.artifact_path(workspace_id, kind)
        if not os.path.exists(path):
            return None
        expected = self._read_manifest(workspace_id).get(kind)
        if verify and expected and _sha256(path) != expected:
            raise ArtifactIntegrityError(f"{os.path.basename(path)} does not match its recorded checksum; "
                                         "refusing to unpickle it")
        if verify and not expected:
            logger.warning("%s has no recorded checksum (created by an older version)", path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_preprocessor(self, workspace_id: str, preprocessor):
        self._save_artifact(workspace_id, "pipeline", preprocessor)

    def load_preprocessor(self, workspace_id: str):
        return self._load_artifact(workspace_id, "pipeline")

    def save_session_state(self, workspace_id: str, state_data: Dict):
        self._save_artifact(workspace_id, "state", state_data)

    def load_session_state(self, workspace_id: str) -> Optional[Dict]:
        return self._load_artifact(workspace_id, "state")

    def save_training_data(self, workspace_id: str, X_train, X_test, y_train, y_test):
        self._save_artifact(workspace_id, "training_data",
                            {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test})

    def load_training_data(self, workspace_id: str) -> Optional[Dict]:
        return self._load_artifact(workspace_id, "training_data")

    def save_trained_models(self, workspace_id: str, trained_models: Dict) -> List[str]:
        """Pickle the trained models; returns the names that could not be serialised (not saved)."""
        ok, skipped = {}, []
        for name, model in trained_models.items():
            try:
                pickle.dumps(model)
                ok[name] = model
            except (pickle.PicklingError, TypeError, AttributeError) as exc:
                logger.warning("Model %s is not serialisable: %s", name, exc)
                skipped.append(name)
        self._save_artifact(workspace_id, "trained_models", ok)
        return skipped

    def load_trained_models(self, workspace_id: str) -> Optional[Dict]:
        return self._load_artifact(workspace_id, "trained_models")

    # ── datasets ─────────────────────────────────────────────────────────────
    def save_dataset(self, workspace_id: str, df: pd.DataFrame):
        if df is not None:
            df.to_csv(os.path.join(UPLOADS_DIR, f"{workspace_id}_data.csv"), index=False)

    def load_dataset(self, workspace_id: str) -> Optional[pd.DataFrame]:
        for path in (os.path.join(UPLOADS_DIR, f"{workspace_id}_data.csv"),
                     os.path.join(WORKSPACE_DIR, f"{workspace_id}_data.csv")):
            if os.path.exists(path):
                return pd.read_csv(path)
        return None

    # ── meta-learning ────────────────────────────────────────────────────────
    def find_similar_workspaces(self, current_stats: Dict, exclude_id: str = None,
                                threshold: float = meta_learning.SIMILARITY_THRESHOLD) -> List[Dict]:
        """Past workspaces of the same task with a recorded best model and profile similarity >= threshold.

        Uses the profiles stored in the index (no workspace JSON is opened). Sorted by similarity.
        """
        if not current_stats:
            return []
        profile = current_stats.get("profile")
        task = normalize_task_type(current_stats.get("task_type"))
        if not meta_learning.valid_profile(profile) or task is None:
            return []
        matches = []
        for entry in self.index.get("workspaces", []):
            if entry.get("workspace_id") == exclude_id or not entry.get("best_model"):
                continue
            entry_task = normalize_task_type(entry.get("task_type")) if entry.get("task_type") else None
            if entry_task != task or not meta_learning.valid_profile(entry.get("profile")):
                continue
            sim = meta_learning.similarity(profile, entry["profile"])
            if sim < threshold:
                continue
            close, far = meta_learning.explain(profile, entry["profile"])
            reason = (f"same task ({task.value}); profile similarity {sim:.3f} >= {threshold:.2f}; "
                      f"closest on {', '.join(close)}" + (f"; differs most on {', '.join(far)}" if far else ""))
            matches.append({"workspace_id": entry["workspace_id"], "dataset": entry.get("dataset_name"),
                            "best_model": entry["best_model"], "best_score": entry.get("best_score"),
                            "metric": entry.get("best_metric"), "similarity": round(sim, 4), "reason": reason})
        matches.sort(key=lambda m: -m["similarity"])
        return matches
