"""
Plotly figure builders. Every figure uses the theme template for the viewer's mode,
one y-axis, thin marks, single-hue bars for one series, the categorical order for
identity (with legend + marker/dash as secondary encoding), a one-hue sequential scale
for magnitude and blue<->gray<->red for polarity. Each figure is returned with the
DataFrame behind it so the page can show a table view.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_frontend.ui import theme

# Neutral ink readable on both surfaces (4.85:1 on dark, 3.4:1 on light): reference lines and value labels,
# which are rendered once and must survive a live theme switch.
NEUTRAL = "#898781"
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "cross", "star", "hexagon"]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot", "solid", "dash"]


def _mode(mode):
    return mode or theme.current_mode()


def label(value) -> str:
    """Plotly renders a subset of HTML in text; escape '<' / '>' so labels like '<=50K' survive."""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _labels(values):
    return [label(v) for v in values]


def _legend_on_top(fig, n_rows_of_legend: int = 1):
    """Title at the top, legend on its own row between title and plot (they used to overlap)."""
    fig.update_layout(margin=dict(t=70 + 22 * n_rows_of_legend),
                      title=dict(y=0.98, yanchor="top", yref="container"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))


def _base(mode, title=None, height=380, **layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=theme.plotly_template(_mode(mode)), title=title, height=height, **layout)
    return fig


def _series(mode, i):
    return theme.CATEGORICAL[_mode(mode)][i % 8]


def bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, mode=None, fmt: str = ".3f",
        horizontal: bool = True, ref: Optional[Tuple[float, str]] = None, value_title: str = None) -> go.Figure:
    """One series -> one colour (slot 1); values as direct labels; optional reference line (e.g. a baseline)."""
    height = max(260, 38 * len(df) + 110) if horizontal else 380
    fig = _base(mode, title, height=height)
    kw = dict(marker_color=_series(mode, 0), texttemplate=f"%{{{'x' if horizontal else 'y'}:{fmt}}}",
              textposition="outside", textfont=dict(color=NEUTRAL), cliponaxis=False,
              hovertemplate=f"%{{{'y' if horizontal else 'x'}}}: %{{{'x' if horizontal else 'y'}:{fmt}}}<extra></extra>")
    if horizontal:
        fig.add_bar(y=_labels(df[label_col]), x=df[value_col], orientation="h", **kw)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(title=value_title or value_col)
    else:
        fig.add_bar(x=_labels(df[label_col]), y=df[value_col], **kw)
        fig.update_yaxes(title=value_title or value_col)
    if ref is not None:
        value, text = ref
        line = dict(color=NEUTRAL, width=1.5, dash="dash")
        if horizontal:
            fig.add_vline(x=value, line=line, annotation_text=text, annotation_position="top",
                          annotation_font_color=NEUTRAL)
        else:
            fig.add_hline(y=value, line=line, annotation_text=text, annotation_font_color=NEUTRAL)
    fig.update_layout(showlegend=False, margin=dict(r=40))
    return fig


def histograms(df: pd.DataFrame, columns: List[str], mode=None, bins: int = 30) -> go.Figure:
    n = len(columns)
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(c) for c in columns],
                        horizontal_spacing=0.08, vertical_spacing=0.16)
    for i, c in enumerate(columns):
        fig.add_histogram(x=df[c], nbinsx=bins, marker_color=_series(mode, 0), row=i // cols + 1, col=i % cols + 1,
                          name=str(c), hovertemplate=f"{c}: %{{x}}<br>rows: %{{y}}<extra></extra>")
    fig.update_layout(template=theme.plotly_template(_mode(mode)), showlegend=False, bargap=0.08,
                      height=230 * rows + 40, margin=dict(t=40))
    fig.update_annotations(font_size=13)
    return fig


def correlation_heatmap(corr: pd.DataFrame, mode=None) -> go.Figure:
    m = _mode(mode)
    n = len(corr)
    fig = _base(mode, "Correlation (Pearson, numeric columns)", height=max(360, 26 * n + 140))
    fig.add_heatmap(z=corr.values, x=_labels(corr.columns), y=_labels(corr.index),
                    zmin=-1, zmax=1, zmid=0, colorscale=theme.diverging_scale(m), xgap=2, ygap=2,
                    text=np.round(corr.values, 2) if n <= 15 else None,
                    texttemplate="%{text}" if n <= 15 else None, textfont=dict(size=11),
                    colorbar=dict(title="r"), hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-40)
    return fig


def confusion_matrix(cm: np.ndarray, labels: List[str], title: str, mode=None) -> go.Figure:
    m = _mode(mode)
    fig = _base(mode, title, height=max(320, 70 * len(labels) + 140))
    fig.add_heatmap(z=cm, x=_labels(labels), y=_labels(labels), colorscale=theme.sequential_scale(m), xgap=2, ygap=2,
                    text=cm, texttemplate="%{text:,}", textfont=dict(size=14),
                    colorbar=dict(title="rows"), hovertemplate="actual %{y}, predicted %{x}: %{z:,}<extra></extra>")
    fig.update_xaxes(title="Predicted", showgrid=False, type="category")
    fig.update_yaxes(title="Actual", showgrid=False, autorange="reversed", type="category")
    return fig


def roc_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]], title: str, mode=None) -> go.Figure:
    fig = _base(mode, label(title), height=460)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, width=1), name="chance",
                    hoverinfo="skip")
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        fig.add_scatter(x=fpr, y=tpr, mode="lines+markers", name=label(f"{name} (AUC {auc:.3f})"),
                        line=dict(color=_series(mode, i), width=2, dash=DASHES[i % 8]),
                        marker=dict(symbol=SYMBOLS[i % 8], size=8, maxdisplayed=10),
                        hovertemplate=f"{name}<br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>")
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    _legend_on_top(fig, n_rows_of_legend=1 + len(curves) // 3)
    return fig


def pred_vs_actual(y_true, y_pred, title: str, mode=None) -> go.Figure:
    fig = _base(mode, title, height=400)
    lo, hi = float(np.nanmin([np.min(y_true), np.min(y_pred)])), float(np.nanmax([np.max(y_true), np.max(y_pred)]))
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=NEUTRAL, width=1, dash="dash"),
                    name="perfect prediction", hoverinfo="skip")
    fig.add_scattergl(x=y_true, y=y_pred, mode="markers", name="test rows",
                      marker=dict(color=_series(mode, 0), size=6, opacity=0.5, line=dict(width=0)),
                      hovertemplate="actual %{x:.3f}<br>predicted %{y:.3f}<extra></extra>")
    fig.update_xaxes(title="Actual")
    fig.update_yaxes(title="Predicted")
    return fig


def residual_hist(residuals, title: str, mode=None) -> go.Figure:
    fig = _base(mode, title, height=400, bargap=0.05)
    fig.add_histogram(x=residuals, nbinsx=40, marker_color=_series(mode, 0), name="residuals",
                      hovertemplate="residual %{x}<br>rows %{y}<extra></extra>")
    fig.add_vline(x=0, line=dict(color=NEUTRAL, width=1))
    fig.update_xaxes(title="Residual (actual − predicted)")
    fig.update_yaxes(title="Rows")
    fig.update_layout(showlegend=False)
    return fig


def grouped_bars(df: pd.DataFrame, category: str, series: List[str], title: str, mode=None, fmt=".3f") -> go.Figure:
    """Few series on one axis (e.g. before/after tuning); legend + direct labels."""
    fig = _base(mode, title, height=360, bargap=0.25, bargroupgap=0.08)
    for i, s in enumerate(series):
        fig.add_bar(x=df[category], y=df[s], name=s, marker_color=_series(mode, i),
                    texttemplate=f"%{{y:{fmt}}}", textposition="outside", textfont=dict(color=NEUTRAL),
                    cliponaxis=False, hovertemplate=f"{s} · %{{x}}: %{{y:{fmt}}}<extra></extra>")
    fig.update_layout(barmode="group")
    _legend_on_top(fig)
    return fig


def beeswarm(df: pd.DataFrame, title: str, mode=None) -> go.Figure:
    m = _mode(mode)
    order = list(dict.fromkeys(df["Feature"]))
    rng = np.random.default_rng(0)
    y = df["Feature"].map({f: i for i, f in enumerate(order)}).to_numpy() + rng.uniform(-0.3, 0.3, len(df))
    fig = _base(mode, title, height=max(360, 34 * len(order) + 120))
    fig.add_vline(x=0, line=dict(color=NEUTRAL, width=1))
    fig.add_scattergl(x=df["SHAP Value"], y=y, mode="markers",
                      marker=dict(size=6, color=df["Feature Value (norm)"], colorscale=theme.diverging_scale(m),
                                  cmin=0, cmax=1, opacity=0.8, line=dict(width=0),
                                  colorbar=dict(title="feature value", tickvals=[0, 1], ticktext=["low", "high"])),
                      customdata=np.stack([df["Feature"], df["Feature Value"]], axis=1),
                      hovertemplate="%{customdata[0]} = %{customdata[1]:.3g}<br>SHAP %{x:.4f}<extra></extra>")
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=order, autorange="reversed", showgrid=False)
    fig.update_xaxes(title="SHAP value (impact on the explained output)")
    return fig


def waterfall(data: Dict, title: str, mode=None) -> go.Figure:
    feats, vals = data["features"], data["shap_contributions"]
    fig = _base(mode, title, height=max(360, 30 * len(feats) + 160))
    fig.add_trace(go.Waterfall(
        orientation="h", measure=["relative"] * len(vals) + ["total"],
        y=[str(f) for f in feats] + ["prediction f(x)"], x=list(vals) + [None], base=data["base_value"],
        text=[f"{v:+.3f}" for v in vals] + [f"{data['prediction']:.3f}"], textposition="outside",
        textfont=dict(color=NEUTRAL), cliponaxis=False,
        increasing=dict(marker=dict(color=theme.DIVERGING["positive"])),
        decreasing=dict(marker=dict(color=theme.DIVERGING["negative"])),
        totals=dict(marker=dict(color=NEUTRAL)), connector=dict(line=dict(color=NEUTRAL, width=1))))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    # pad the x range so the value labels outside the bars never run into the feature names
    running = data["base_value"] + np.concatenate([[0], np.cumsum(vals)])
    lo, hi = float(min(running.min(), data["prediction"])), float(max(running.max(), data["prediction"]))
    pad = 0.3 * (hi - lo or 1.0)
    fig.update_xaxes(title=f"output (base value {data['base_value']:.3f})", range=[lo - pad, hi + pad])
    fig.update_layout(showlegend=False, margin=dict(r=60))
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, title: str, mode=None) -> go.Figure:
    fig = _base(mode, title, height=380)
    fig.add_scattergl(x=df[x], y=df[y], mode="markers",
                      marker=dict(color=_series(mode, 0), size=7, opacity=0.6, line=dict(width=0)),
                      hovertemplate=f"{x} %{{x:.3g}}<br>{y} %{{y:.4f}}<extra></extra>")
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return fig
