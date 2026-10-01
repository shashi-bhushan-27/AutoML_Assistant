"""Workspaces dashboard: search, sort, open, delete (with confirmation), new workspace, samples."""
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

from app_frontend.ui import components as ui
from app_frontend.ui import nav
from app_frontend.ui import state as S
from app_frontend.ui.services import available_samples, sample_shape
from app_frontend.views.data import set_dataset


def ago(iso: str) -> str:
    try:
        delta = datetime.now() - datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return "unknown"
    s = delta.total_seconds()
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if s >= size:
            n = int(s // size)
            return f"{n} {unit}{'s' if n > 1 else ''} ago"
    return "just now"


def progress_text(steps: dict) -> tuple:
    statuses = [(steps or {}).get(n, {}).get("state") for n, _ in S.STEPS]
    done = sum(1 for x in statuses if x == "done")
    errors = sum(1 for x in statuses if x == "error")
    return done, errors


@st.dialog("Delete workspace?")
def confirm_delete(ws_id: str, name: str):
    st.write(f"This permanently deletes **{name}** (`{ws_id}`): its dataset copy, preprocessing pipeline, "
             "trained models and results. It cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Delete permanently", type="primary", icon=":material/delete:", width="stretch"):
        failed = S.wm().delete_workspace(ws_id)
        if S.current_ws() is not None and S.current_ws().workspace_id == ws_id:
            S.close_workspace()
        if failed:
            st.error(f"Some files could not be removed: {failed}")
        else:
            st.toast(f"Deleted {name}")
            st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()


def _open(ws_id: str):
    problems = S.open_workspace(ws_id)
    st.session_state["open_problems"] = problems
    statuses = S.step_statuses()
    target = next((n for n, _ in reversed(S.STEPS) if statuses[n]["state"] in ("done", "stale", "error")), "data")
    nav.go(target)


def render():
    st.title("Workspaces", anchor=False)
    st.caption("A workspace holds one dataset, its fitted preprocessing pipeline, the trained models and the "
               "results of every step. Everything is stored locally under workspaces/ and data/uploads/.")
    for p in st.session_state.pop("open_problems", []) or []:
        ui.banner("critical", p)

    c1, c2, c3 = st.columns([1.1, 1.6, 1.0], vertical_alignment="bottom")
    if c1.button("New workspace", type="primary", icon=":material/add:", width="stretch"):
        S.new_workspace()
        nav.go("data")
    samples = available_samples()
    if samples:
        labels = {f"{s['name']} ({sample_shape(s['path'])[0]:,} rows × {sample_shape(s['path'])[1]} cols)": s
                  for s in samples}
        choice = c2.selectbox("Or start from a sample dataset", list(labels), key="w_sample")
        if c3.button("Load sample", icon=":material/dataset:", width="stretch"):
            s = labels[choice]
            S.new_workspace(s["name"])
            raw = open(s["path"], "rb").read()
            df = pd.read_csv(s["path"])
            set_dataset(df, f"{s['name']} (sample).csv",
                        {"encoding": "utf-8", "size_mb": len(raw) / 1e6, "warnings": [], "sample": s["about"]},
                        hashlib.sha256(raw).hexdigest()[:16])
            if s["target"]:
                S.state()["suggested_target"] = s["target"]
                S.save_state()
            nav.go("data")
        if choice:
            st.caption(labels[choice]["about"])

    workspaces = S.wm().list_workspaces()
    if not workspaces:
        ui.banner("info", "No workspaces yet. Create one, or load a sample dataset to try the full workflow.")
        return

    f1, f2, f3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    query = f1.text_input("Search", placeholder="Dataset, model, task or workspace id", key="w_search").lower()
    sort = f2.selectbox("Sort by", ["Last updated", "Name", "Created", "Best score"], key="w_sort")
    view = f3.segmented_control("View", ["Cards", "Table"], default="Cards", key="w_view") or "Cards"

    rows = []
    for w in workspaces:
        text = " ".join(str(w.get(k) or "") for k in ("dataset_name", "best_model", "task_type", "workspace_id")).lower()
        if query and query not in text:
            continue
        rows.append(w)
    key = {"Last updated": lambda w: w.get("updated_at") or "", "Name": lambda w: (w.get("dataset_name") or "").lower(),
           "Created": lambda w: w.get("created_at") or "", "Best score": lambda w: w.get("best_score") or float("-inf")}[sort]
    rows.sort(key=key, reverse=sort in ("Last updated", "Created", "Best score"))
    st.caption(f"{len(rows)} of {len(workspaces)} workspaces")

    if view == "Table":
        table = pd.DataFrame([{
            "Dataset": w["dataset_name"], "Task": w.get("task_type"),
            "Shape": " × ".join(map(str, w.get("dataset_shape") or [])) or "-",
            "Best model": w.get("best_model") or "-",
            "Metric": f"{w.get('best_metric')} {w['best_score']:.4f}" if w.get("best_score") is not None else "-",
            "Steps done": f"{progress_text(w.get('steps'))[0]}/7", "Updated": ago(w.get("updated_at")),
            "ID": w["workspace_id"]} for w in rows])
        event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key="w_table")
        sel = event.selection.rows if event and event.selection else []
        if sel:
            w = rows[sel[0]]
            b1, b2 = st.columns(2)
            if b1.button(f"Open {w['dataset_name']}", type="primary", width="stretch"):
                _open(w["workspace_id"])
            if b2.button("Delete…", width="stretch"):
                confirm_delete(w["workspace_id"], w["dataset_name"])
        else:
            st.caption("Select a row to open or delete it.")
        return

    cols = st.columns(3)
    for i, w in enumerate(rows):
        done, errors = progress_text(w.get("steps"))
        shape = w.get("dataset_shape") or []
        task = w.get("task_type") if w.get("task_type") not in (None, "Unknown") else None
        chips = [ui.chip(task or "No task yet", "info" if task else "neutral")]
        if w.get("best_model"):
            score = f"{w['best_metric']} {w['best_score']:.4f}" if w.get("best_score") is not None else ""
            chips.append(ui.chip(f"Best: {w['best_model']} {score}".strip(), "neutral", dot=False))
        chips.append(ui.chip(f"{done}/7 steps done", "good" if done == 7 else "neutral"))
        if errors:
            chips.append(ui.chip(f"{errors} step error", "critical"))
        with cols[i % 3]:
            st.markdown(
                f'<div class="card"><h4>{ui.esc(w["dataset_name"])}</h4>'
                f'<p class="meta">{ui.esc(" × ".join(map(str, shape)) + " · " if len(shape) == 2 and shape[0] else "")}'
                f'updated {ui.esc(ago(w.get("updated_at")))} · id {ui.esc(w["workspace_id"])}</p>'
                f'<div class="row">{"".join(chips)}</div></div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            if b1.button("Open", key=f"open_{w['workspace_id']}", icon=":material/folder_open:",
                         width="stretch"):
                _open(w["workspace_id"])
            if b2.button("Delete", key=f"del_{w['workspace_id']}", icon=":material/delete:",
                         width="stretch", help="Asks for confirmation"):
                confirm_delete(w["workspace_id"], w["dataset_name"])
