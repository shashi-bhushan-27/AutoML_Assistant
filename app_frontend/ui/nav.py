"""Page registry shared by the entry point and the views (filled in by main_ui.py on every run)."""
import os
from typing import Dict

import streamlit as st

from app_frontend.ui import components as ui
from app_frontend.ui import state as S

PAGES: Dict[str, "st.Page"] = {}
# API_BASE_URL: where this server reaches the API (readiness check); API_PUBLIC_URL: what users' browsers and
# the generated examples should use (differs in docker-compose, where the API is http://backend:8000 inside).
# 127.0.0.1 rather than localhost: on Windows "localhost" tries IPv6 first and each request waits ~2 s for the
# fallback when the API listens on IPv4 only.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", "http://localhost:8000").rstrip("/")


def go(name: str):
    st.switch_page(PAGES[name])


def link(name: str, label: str = None, icon: str = None, disabled: bool = False):
    st.page_link(PAGES[name], label=label or S.STEP_TITLES.get(name, name.title()), icon=icon, disabled=disabled)


def require_step(step: str):
    """Header + gating for a workspace page. Stops the page if no workspace is open or the step is locked."""
    ws = S.current_ws()
    if ws is None:
        ui.banner("info", "No workspace is open. Open or create one on the Workspaces page.")
        link("home", "Go to Workspaces", ":material/folder_open:")
        st.stop()
    statuses = S.step_statuses()
    pipe = S.ctx().get("pipeline")
    split = None
    if pipe is not None:
        split = f"{int(round((1 - pipe.test_size) * 100))}/{int(round(pipe.test_size * 100))} {pipe.splitter.strategy}"
    ui.ws_header(ws, statuses, step, task=S.task_type(), targets=S.target_cols() or None,
                 seed=ws.seed if pipe is not None else None, split=split)
    if not statuses[step]["enabled"]:
        pre = S.PREREQ[step]
        ui.banner("info", f"This step is locked: {statuses[step]['note'] or 'a previous step is missing'}.",
                  title="i Locked")
        link(pre, f"Go to {S.STEP_TITLES[pre]}", ":material/arrow_back:")
        st.stop()
    if statuses[step]["state"] == "stale":
        ui.banner("warning", f"Results on this page are stale: {statuses[step]['note'].lower()}. "
                             "Re-run this step to update them.", title="! Stale")
    return ws, statuses
