"""
Session and workspace state for the UI.

The seven workflow steps each record, in ``Workspace.steps``, the fingerprint of the
upstream result they were built from. A step is **stale** when that upstream
fingerprint has changed (new data, new target or split, retrained models), **error**
when its last run failed, **done** otherwise; downstream steps of a stale step are stale
too. A step is enabled only when its prerequisite has been run.
"""
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app_backend.model_trainer import ModelTrainer
from app_backend.workspace_manager import ArtifactIntegrityError, Workspace, WorkspaceManager

logger = logging.getLogger(__name__)

STEPS = [("data", "Data"), ("prepare", "Prepare"), ("recommend", "Recommend"), ("train", "Train & Compare"),
         ("explain", "Explain"), ("tune", "Tune"), ("deploy", "Deploy")]
STEP_TITLES = dict(STEPS)
PREREQ = {"data": None, "prepare": "data", "recommend": "prepare", "train": "prepare",
          "explain": "train", "tune": "train", "deploy": "train"}
UPSTREAM = PREREQ  # the step whose fingerprint a step records
STATE_LABELS = {"not_started": "Not started", "done": "Done", "stale": "Stale", "error": "Error",
                "running": "Running", "disabled": "Locked"}


# ── fingerprints ─────────────────────────────────────────────────────────────
def fingerprint(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def dataset_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    return h.hexdigest()[:16]


# ── managers / session ───────────────────────────────────────────────────────
def wm() -> WorkspaceManager:
    if "wm" not in st.session_state:
        st.session_state.wm = WorkspaceManager()
    manager = st.session_state.wm
    manager.index = manager._load_index()  # other sessions may have written the index
    return manager


def ctx() -> Dict[str, Any]:
    """Objects of the open workspace kept in this browser session."""
    if "ctx" not in st.session_state:
        st.session_state.ctx = {}
    return st.session_state.ctx


def current_ws() -> Optional[Workspace]:
    return ctx().get("ws")


def state() -> Dict[str, Any]:
    return ctx().setdefault("state", {})


def close_workspace():
    st.session_state.ctx = {}
    for key in list(st.session_state.keys()):
        if key.startswith(("w_", "job_")):
            del st.session_state[key]


def new_workspace(name: str = "Untitled workspace") -> Workspace:
    ws = wm().create_workspace(dataset_name=name, dataset_shape=None)
    ws.user_config.setdefault("seed", 42)
    wm().save_workspace(ws)
    close_workspace()
    ctx()["ws"] = ws
    ctx()["state"] = {}
    return ws


def open_workspace(workspace_id: str) -> List[str]:
    """Load a workspace and its artefacts into the session. Returns a list of problems (shown in the UI)."""
    manager = wm()
    ws = manager.load_workspace(workspace_id)
    if ws is None:
        return [f"Workspace {workspace_id} not found."]
    close_workspace()
    c = ctx()
    c["ws"] = ws
    problems = []
    c["df"] = manager.load_dataset(workspace_id)
    for kind, key in (("state", "state"), ("pipeline", "pipeline"), ("training_data", "splits"),
                      ("trained_models", "models")):
        try:
            loader = {"state": manager.load_session_state, "pipeline": manager.load_preprocessor,
                      "training_data": manager.load_training_data, "trained_models": manager.load_trained_models}[kind]
            c[key] = loader(workspace_id)
        except ArtifactIntegrityError as exc:
            problems.append(str(exc))
            c[key] = None
        except Exception as exc:  # corrupted or incompatible pickle: say so, keep going
            problems.append(f"Could not load the saved {kind.replace('_', ' ')}: {type(exc).__name__}: {exc}")
            c[key] = None
    c["state"] = c.get("state") or {}
    pipe = c.get("pipeline")
    if pipe is not None and getattr(pipe, "pipeline_version", 1) != 2:
        problems.append("This workspace was prepared by an older version of the app; re-run Prepare "
                        "(its pipeline cannot be served).")
        c["pipeline"] = None
    if c.get("splits") and pipe is not None:
        try:
            rebuild_trainer()
        except Exception as exc:
            problems.append(f"Could not rebuild the trained models: {type(exc).__name__}: {exc}")
    return problems


def rebuild_trainer() -> Optional[ModelTrainer]:
    c, s = ctx(), state()
    ws, pipe, splits = c.get("ws"), c.get("pipeline"), c.get("splits")
    if ws is None or pipe is None or not splits:
        c["trainer"] = None
        return None
    cfg = s.get("prepare_config", {})
    tr = ModelTrainer(c.get("df"), pipe.target_cols, pipe.task_type, cfg.get("is_time_series", False),
                      cfg.get("date_col"), random_state=ws.seed,
                      class_weight=s.get("train_config", {}).get("class_weight"))
    tr.set_preprocessed_data(splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"])
    # models are only valid for the preparation they were trained on
    train_rec, prep_rec = (ws.steps or {}).get("train") or {}, (ws.steps or {}).get("prepare") or {}
    if train_rec.get("state") == "done" and train_rec.get("upstream") == prep_rec.get("fp"):
        tr.trained_models = dict(c.get("models") or {})
        tr.predictions = dict(s.get("predictions", {}))
        tr.prediction_probas = dict(s.get("probas", {}))
        tr.prediction_scores = dict(s.get("scores", {}))
    c["trainer"] = tr
    return tr


def save_ws():
    ws = current_ws()
    if ws is not None:
        wm().save_workspace(ws)


def save_state():
    ws = current_ws()
    if ws is not None:
        wm().save_session_state(ws.workspace_id, state())


# ── steps ────────────────────────────────────────────────────────────────────
def record_step(name: str, ok: bool = True, fp: str = None, error: str = None, **extra):
    ws = current_ws()
    upstream = UPSTREAM[name]
    rec = {"state": "done" if ok else "error", "at": datetime.now().isoformat(timespec="seconds"),
           "upstream": (ws.steps.get(upstream) or {}).get("fp") if upstream else None,
           "fp": fp or fingerprint(name, time.time()), "error": error, **extra}
    ws.steps[name] = rec
    ws.add_event(f"{name}_{'done' if ok else 'error'}", error or f"{STEP_TITLES[name]} completed")
    save_ws()


def step_statuses(ws: Workspace = None) -> Dict[str, Dict[str, Any]]:
    ws = ws or current_ws()
    out: Dict[str, Dict[str, Any]] = {}
    running = {job.step for job in st.session_state.get("jobs", {}).values()
               if getattr(job, "status", None) == "running" and getattr(job, "step", None)} if st.runtime.exists() else set()
    for name, _ in STEPS:
        rec = (ws.steps or {}).get(name) if ws else None
        pre = PREREQ[name]
        pre_state = out.get(pre, {}).get("state") if pre else "done"
        enabled = pre is None or (pre_state in ("done", "stale") and (ws.steps or {}).get(pre, {}).get("state") == "done")
        note = ""
        if name in running:
            st_ = "running"
        elif not rec:
            st_ = "not_started" if enabled else "disabled"
            if not enabled:
                note = f"Needs {STEP_TITLES[pre]}"
        elif rec.get("state") == "error":
            st_, note = "error", (rec.get("error") or "")[:120]
        elif pre and (rec.get("upstream") != (ws.steps.get(pre) or {}).get("fp") or pre_state == "stale"):
            st_, note = "stale", f"{STEP_TITLES[pre]} changed since this ran"
        else:
            st_ = "done"
        out[name] = {"state": st_, "enabled": enabled, "note": note}
    return out


def prerequisite_problem(step: str) -> Optional[str]:
    """Why the action of ``step`` cannot run (None if it can)."""
    pre = PREREQ[step]
    if pre is None:
        return None
    s = step_statuses()
    if s[pre]["state"] in ("not_started", "disabled", "error"):
        return f"Run **{STEP_TITLES[pre]}** first."
    if s[pre]["state"] == "stale":
        return f"**{STEP_TITLES[pre]}** is stale ({s[pre]['note'].lower()}); re-run it first."
    return None


def task_type() -> Optional[str]:
    pipe = ctx().get("pipeline")
    return pipe.task_type if pipe is not None else (current_ws().task_type if current_ws() else None)


def target_cols() -> List[str]:
    pipe = ctx().get("pipeline")
    if pipe is not None:
        return list(pipe.target_cols)
    t = current_ws().target_col if current_ws() else None
    return list(t) if isinstance(t, list) else ([t] if t else [])


def raw_holdout(n: int = None) -> Optional[pd.DataFrame]:
    """Raw rows of the held-out test split (features only), as uploaded."""
    c = ctx()
    df, splits, pipe = c.get("df"), c.get("splits"), c.get("pipeline")
    if df is None or not splits or pipe is None:
        return None
    idx = [i for i in splits["X_test"].index if i in df.index]
    rows = df.loc[idx].drop(columns=[t for t in pipe.target_cols if t in df.columns])
    return rows.head(n) if n else rows
