"""Reusable UI pieces. Status is always shown as icon + word + colour, never colour alone."""
import html
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from app_frontend.ui import theme
from app_frontend.ui.state import STATE_LABELS, STEPS

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
STATE_KIND = {"done": "good", "stale": "warning", "error": "critical", "running": "info",
              "not_started": "neutral", "disabled": "neutral"}
STATE_ICON = {"done": "✓", "stale": "!", "error": "✕", "running": "…", "not_started": "○", "disabled": "·"}
BANNER_ICON = {"good": "✓ OK", "warning": "! Warning", "critical": "✕ Error", "info": "i Note"}

LOGO_SVG = """<svg width="28" height="28" viewBox="0 0 28 28" role="img" aria-label="AutoML Assistant logo">
<rect x="1" y="1" width="26" height="26" rx="7" fill="none" stroke="currentColor" stroke-width="2"/>
<rect x="7" y="15" width="3" height="7" rx="1.5" fill="#2a78d6"/><rect x="12.5" y="10" width="3" height="12" rx="1.5"
fill="#2a78d6"/><rect x="18" y="6" width="3" height="16" rx="1.5" fill="#2a78d6"/></svg>"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def inject_css():
    with open(os.path.join(ASSETS, "style.css"), encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{theme.css_variables()}\n{css}</style>", unsafe_allow_html=True)


def brand(subtitle: str = "Tabular AutoML workspace"):
    st.markdown(f'<div class="brand">{LOGO_SVG}<div>AutoML Assistant<small>{esc(subtitle)}</small></div></div>',
                unsafe_allow_html=True)


def chip(text: str, kind: str = "neutral", dot: bool = True) -> str:
    return f'<span class="chip {kind}">{"<span class=dot></span>" if dot else ""}{esc(text)}</span>'


def chips(items: Iterable[str]):
    st.markdown('<div class="card-row" style="display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 12px">'
                + "".join(items) + "</div>", unsafe_allow_html=True)


def banner(kind: str, text: str, title: str = None):
    """kind: good | warning | critical | info. ``text`` may contain simple markdown-free HTML-escaped text."""
    label = title or BANNER_ICON[kind]
    st.markdown(f'<div class="banner {kind}" role="status"><span class="icon">{esc(label)}</span>'
                f'<p>{esc(text)}</p></div>', unsafe_allow_html=True)


def tiles(items: Sequence[Tuple[str, str, Optional[str]]]):
    cells = "".join(f'<div class="tile"><div class="label">{esc(l)}</div><div class="value">{esc(v)}</div>'
                    + (f'<div class="note">{esc(n)}</div>' if n else "") + "</div>" for l, v, n in items)
    st.markdown(f'<div class="tiles">{cells}</div>', unsafe_allow_html=True)


def llm_chip(health: Dict) -> str:
    if health.get("ok"):
        return chip(f"LLM: {health['model']} · OK", "good")
    return chip(f"LLM: {health.get('model')} · unavailable", "critical")


def ws_header(ws, statuses: Dict, current: str, task: str = None, targets: List[str] = None, seed: int = None,
              split: str = None):
    shape = ws.dataset_shape
    shape_txt = f"{shape[0]:,} rows × {shape[1]} columns" if shape and len(shape) == 2 and shape[0] else "no data yet"
    sub = [shape_txt]
    if targets:
        sub.append("target: " + ", ".join(targets))
    items = []
    if task:
        items.append(chip(task, "info"))
    if seed is not None:
        items.append(chip(f"split seed {seed}" + (f" · {split}" if split else ""), "neutral"))
    items.append(chip(f"workspace {ws.workspace_id}", "neutral", dot=False))
    st.markdown(f'<div class="ws-header"><div><p class="title">{esc(ws.dataset_name or "Untitled")}</p>'
                f'<div class="sub">{esc(" · ".join(sub))}</div></div>'
                f'<div class="chips">{"".join(items)}</div></div>', unsafe_allow_html=True)
    compact = []
    for i, (name, title) in enumerate(STEPS, 1):
        s = statuses[name]["state"]
        cls = "current" if name == current else {"done": "done", "stale": "stale", "error": "error"}.get(s, "")
        compact.append(f'<span class="{cls}" title="{esc(STATE_LABELS[s])}">{i}. {esc(title)} '
                       f'{STATE_ICON[s]}</span>')
    st.markdown(f'<nav class="stepper-compact" aria-label="Workflow progress">{"".join(compact)}</nav>',
                unsafe_allow_html=True)


def download_df(df: pd.DataFrame, filename: str, key: str, label: str = "Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv",
                       key=key, icon=":material/download:")


def chart(fig, data: pd.DataFrame, key: str, filename: str, caption: str = None, height: int = None):
    """A Plotly chart with a Table tab (the chart's data) and PNG/CSV downloads."""
    if height:
        fig.update_layout(height=height)
    # Plotly clips tick labels that do not fit the margins unless automargin is on.
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    tab_chart, tab_table = st.tabs(["Chart", "Table"])
    with tab_chart:
        # theme="streamlit": backgrounds, text and grid follow the active light/dark theme client-side (also
        # after a switch in Settings, which does not rerun the script); series colours stay as set.
        st.plotly_chart(fig, theme="streamlit", width="stretch", key=f"{key}_fig",
                        config={"displaylogo": False, "responsive": True,
                                "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2},
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        if caption:
            st.caption(caption)
    with tab_table:
        st.dataframe(data, width="stretch", hide_index=True)
        download_df(data, f"{filename}.csv", key=f"{key}_csv")


def status_label(state: str) -> str:
    return f"{STATE_ICON[state]} {STATE_LABELS[state]}"


def section(title: str, help_text: str = None):
    st.subheader(title, anchor=False)
    if help_text:
        st.caption(help_text)
