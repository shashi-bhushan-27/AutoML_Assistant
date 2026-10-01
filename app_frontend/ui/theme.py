"""
Design tokens: the single source of truth for colour, spacing, radius and type.

* ``TOKENS[mode]`` holds the light and dark values. ``.streamlit/config.toml`` is generated
  from them (``python -m app_frontend.ui.theme`` rewrites it; a test fails if it drifts),
  so Streamlit's own widgets, our CSS and the Plotly charts all use the same values.
* Categorical chart colours: a colour-blind-safe order validated with the dataviz
  palette validator on both surfaces (worst adjacent CVD dE 9.1 light / 8.4 dark,
  normal-vision dE >= 19). Three light slots are below 3:1 against the surface, so every
  chart also has a table view and direct labels.
* Status colours (good / warning / critical) are reserved for status and always shipped
  with an icon and a word, never colour alone.
* Text/background pairs are checked for WCAG contrast in tests/test_theme.py.
"""
import os
from typing import Dict

import plotly.graph_objects as go

FONT_STACK = '"Source Sans", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", sans-serif'

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
DIVERGING = {"negative": "#2a78d6", "neutral": {"light": "#f0efec", "dark": "#383835"}, "positive": "#e34948"}

TOKENS: Dict[str, Dict[str, str]] = {
    "light": {
        "bg": "#f4f4f1", "surface": "#fcfcfb", "surface-2": "#ecebe6", "border": "#e1e0d9",
        "border-strong": "#c3c2b7", "text": "#0b0b0b", "text-2": "#52514e", "muted": "#5f5e5a",
        "accent": "#1c5cab", "on-accent": "#ffffff", "accent-soft": "#e6eefa", "link": "#1c5cab",
        "focus": "#1c5cab", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "good": "#0ca30c", "good-text": "#006300", "good-soft": "#e7f5e7",
        "warning": "#fab219", "warning-text": "#8a5a00", "warning-soft": "#fdf3dc",
        "critical": "#d03b3b", "critical-text": "#b42318", "critical-soft": "#fbe9e8",
        "info-soft": "#e6eefa",
    },
    "dark": {
        "bg": "#141413", "surface": "#1a1a19", "surface-2": "#242422", "border": "#2c2c2a",
        "border-strong": "#383835", "text": "#f2f2ef", "text-2": "#c3c2b7", "muted": "#a3a29b",
        "accent": "#6da7ec", "on-accent": "#0b0b0b", "accent-soft": "#1b2a3d", "link": "#6da7ec",
        "focus": "#86b6ef", "grid": "#2c2c2a", "axis": "#383835",
        "good": "#0ca30c", "good-text": "#3fbf3f", "good-soft": "#16261a",
        "warning": "#fab219", "warning-text": "#fab219", "warning-soft": "#2e2612",
        "critical": "#d03b3b", "critical-text": "#f07a7a", "critical-soft": "#311b1b",
        "info-soft": "#1b2a3d",
    },
}
SPACE = {"1": "4px", "2": "8px", "3": "12px", "4": "16px", "5": "24px", "6": "32px"}
RADIUS = {"sm": "6px", "md": "10px", "lg": "14px", "pill": "999px"}
TYPE = {"xs": "0.75rem", "sm": "0.875rem", "md": "1rem", "lg": "1.25rem", "xl": "1.75rem"}


def current_mode() -> str:
    """'light' or 'dark' as reported by the viewer's browser/Streamlit theme (light if unknown)."""
    try:
        import streamlit as st

        mode = getattr(getattr(st.context, "theme", None), "type", None)
        return mode if mode in ("light", "dark") else "light"
    except Exception:  # outside a Streamlit run
        return "light"


def css_variables(mode: str = None) -> str:
    """CSS custom properties for both themes.

    Each colour token is ``light-dark(<light>, <dark>)``, resolved against the ``color-scheme`` Streamlit sets
    on ``.stApp`` - so the tokens follow the active theme instantly, including a switch in the Settings menu
    (which does not rerun the script). Browsers without ``light-dark()`` get the light values.
    """
    light, dark = TOKENS["light"], TOKENS["dark"]
    fixed = [f"--space-{k}: {v};" for k, v in SPACE.items()]
    fixed += [f"--radius-{k}: {v};" for k, v in RADIUS.items()]
    fixed += [f"--font-{k}: {v};" for k, v in TYPE.items()]
    fallback = [f"--{k}: {v};" for k, v in light.items()]
    fallback += [f"--series-{i + 1}: {c};" for i, c in enumerate(CATEGORICAL["light"])]
    adaptive = [f"--{k}: light-dark({light[k]}, {dark[k]});" for k in light]
    adaptive += [f"--series-{i + 1}: light-dark({a}, {b});"
                 for i, (a, b) in enumerate(zip(CATEGORICAL["light"], CATEGORICAL["dark"]))]
    return (":root, .stApp {\n  %s\n  %s\n}\n@supports (color: light-dark(#000, #fff)) {\n  .stApp {\n    %s\n  }\n}"
            % ("\n  ".join(fixed), "\n  ".join(fallback), "\n    ".join(adaptive)))


def plotly_template(mode: str) -> go.layout.Template:
    t = TOKENS[mode]
    axis = dict(gridcolor=t["grid"], linecolor=t["axis"], zerolinecolor=t["axis"], tickcolor=t["axis"],
                ticks="outside", ticklen=4, showline=True, gridwidth=1, zerolinewidth=1, automargin=True,
                title=dict(font=dict(color=t["text-2"], size=13)), tickfont=dict(color=t["muted"], size=12))
    return go.layout.Template(layout=dict(
        font=dict(family=FONT_STACK, color=t["text"], size=13),
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        colorway=CATEGORICAL[mode],
        title=dict(font=dict(size=15, color=t["text"]), x=0, xanchor="left"),
        xaxis=axis, yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t["text-2"], size=12), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border-strong"], font=dict(color=t["text"], family=FONT_STACK)),
        margin=dict(l=8, r=8, t=48, b=8),
        bargap=0.35, barcornerradius=4,
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=t["muted"]))),
    ))


def diverging_scale(mode: str):
    return [[0.0, DIVERGING["negative"]], [0.5, DIVERGING["neutral"][mode]], [1.0, DIVERGING["positive"]]]


def sequential_scale(mode: str):
    steps = SEQUENTIAL_BLUE if mode == "light" else list(reversed(SEQUENTIAL_BLUE))
    n = len(steps) - 1
    return [[i / n, c] for i, c in enumerate(steps)]


def streamlit_config() -> str:
    """Contents of .streamlit/config.toml, generated from the tokens."""
    def section(mode):
        t = TOKENS[mode]
        return "\n".join([
            f"[theme.{mode}]",
            f'primaryColor = "{t["accent"]}"',
            f'backgroundColor = "{t["bg"]}"',
            f'secondaryBackgroundColor = "{t["surface-2"]}"',
            f'textColor = "{t["text"]}"',
            f'linkColor = "{t["link"]}"',
            f'borderColor = "{t["border"]}"',
            f'dataframeBorderColor = "{t["border"]}"',
            f'greenColor = "{t["good-text"]}"',
            f'yellowColor = "{t["warning-text"]}"',
            f'redColor = "{t["critical-text"]}"',
            f'blueColor = "{t["accent"]}"',
            "",
            f"[theme.{mode}.sidebar]",
            f'backgroundColor = "{t["surface"]}"',
            f'secondaryBackgroundColor = "{t["surface-2"]}"',
            "",
        ])
    return "\n".join([
        "# Generated from app_frontend/ui/theme.py - edit the tokens there and run",
        "#   python -m app_frontend.ui.theme",
        "[server]",
        "maxUploadSize = 200",
        "",
        "[browser]",
        "gatherUsageStats = false",
        "",
        "[client]",
        'toolbarMode = "viewer"',
        "",
        "[theme]",
        'baseRadius = "0.5rem"',
        'buttonRadius = "0.5rem"',
        "showWidgetBorder = true",
        "showSidebarBorder = true",
        "",
        section("light"),
        section("dark"),
    ])


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           ".streamlit", "config.toml")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(streamlit_config())
    print("wrote", CONFIG_PATH)
