"""
AutoML Assistant - Streamlit entry point.

    streamlit run app_frontend/main_ui.py

Multipage app (st.navigation) with a persistent workflow stepper in the sidebar:
Data -> Prepare -> Recommend -> Train & Compare -> Explain -> Tune -> Deploy.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app_backend  # noqa: E402,F401  (loads torch before scikit-learn on Windows)
import streamlit as st  # noqa: E402

st.set_page_config(page_title="AutoML Assistant", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="auto")

from app_backend.llm_rag_core import get_llm_model  # noqa: E402
from app_frontend.ui import components as ui  # noqa: E402
from app_frontend.ui import nav  # noqa: E402
from app_frontend.ui import state as S  # noqa: E402
from app_frontend.ui.services import llm_health  # noqa: E402
from app_frontend.views import data, deploy, docs, explain, home, prepare, recommend, train, tune  # noqa: E402

ui.inject_css()

STATE_ICONS = {"done": ":material/check_circle:", "stale": ":material/warning:", "error": ":material/error:",
               "running": ":material/progress_activity:", "not_started": ":material/radio_button_unchecked:",
               "disabled": ":material/lock:"}
VIEWS = {"data": data, "prepare": prepare, "recommend": recommend, "train": train, "explain": explain,
         "tune": tune, "deploy": deploy}

nav.PAGES = {"home": st.Page(home.render, title="Workspaces", icon=":material/folder_open:", url_path="home",
                             default=True)}
for name, title in S.STEPS:
    nav.PAGES[name] = st.Page(VIEWS[name].render, title=title, url_path=name)
nav.PAGES["docs"] = st.Page(docs.render, title="Docs", icon=":material/menu_book:", url_path="docs")
page = st.navigation(list(nav.PAGES.values()), position="hidden")

with st.sidebar:
    ui.brand()
    st.page_link(nav.PAGES["home"], label="All workspaces", icon=":material/folder_open:")
    ws = S.current_ws()
    if ws is not None:
        statuses = S.step_statuses()
        st.markdown(f"**{ui.esc(ws.dataset_name or 'Untitled')}**")
        for i, (name, title) in enumerate(S.STEPS, 1):
            s = statuses[name]
            st.page_link(nav.PAGES[name], label=f"{i}\\. {title}", icon=STATE_ICONS[s["state"]],
                         disabled=not s["enabled"])
            kind = ui.STATE_KIND[s["state"]]
            note = f" · {s['note']}" if s["note"] and s["state"] != "done" else ""
            st.markdown(f'<div class="step-state {kind}">{ui.esc(S.STATE_LABELS[s["state"]] + note)}</div>',
                        unsafe_allow_html=True)
    st.divider()
    health = llm_health(get_llm_model())
    st.markdown(ui.llm_chip(health), unsafe_allow_html=True)
    if not health.get("ok"):
        st.caption(f"{health.get('error')}. Recommendations and reports fall back to defaults/templates.")
    st.page_link(nav.PAGES["docs"], label="Docs", icon=":material/menu_book:")
    st.markdown(f"[API reference (FastAPI /docs)]({nav.API_PUBLIC_URL}/docs)")
    st.caption("Theme: follows your system; switch Light/Dark in the ⋮ menu → Settings.")

page.run()
