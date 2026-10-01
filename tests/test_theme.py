"""Design tokens: WCAG contrast in both themes, config.toml in sync, charts with one axis and a table view."""
import pytest

from app_frontend.ui import theme


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted([_lum(a), _lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


TEXT_PAIRS = [("text", "bg"), ("text", "surface"), ("text", "surface-2"), ("text-2", "bg"), ("text-2", "surface"),
              ("muted", "bg"), ("muted", "surface"), ("link", "bg"), ("link", "surface"), ("on-accent", "accent"),
              ("good-text", "good-soft"), ("warning-text", "warning-soft"), ("critical-text", "critical-soft"),
              ("accent", "accent-soft"), ("good-text", "surface"), ("warning-text", "surface"),
              ("critical-text", "surface")]


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("fg,bg", TEXT_PAIRS)
def test_text_contrast_meets_wcag_aa(mode, fg, bg):
    t = theme.TOKENS[mode]
    assert contrast(t[fg], t[bg]) >= 4.5, f"{mode}: {fg} {t[fg]} on {bg} {t[bg]} = {contrast(t[fg], t[bg]):.2f}"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_ui_component_contrast(mode):
    t = theme.TOKENS[mode]
    assert contrast(t["focus"], t["bg"]) >= 3.0            # focus ring (WCAG 2.4.11 non-text contrast)
    assert contrast(t["accent"], t["bg"]) >= 3.0            # primary button against the page


def test_both_themes_define_the_same_tokens():
    assert set(theme.TOKENS["light"]) == set(theme.TOKENS["dark"])
    assert len(theme.CATEGORICAL["light"]) == len(theme.CATEGORICAL["dark"]) == 8


def test_css_uses_light_dark_for_every_token():
    css = theme.css_variables()
    for key, light in theme.TOKENS["light"].items():
        assert f"--{key}: light-dark({light}, {theme.TOKENS['dark'][key]});" in css


def test_streamlit_config_is_generated_from_tokens():
    with open(theme.CONFIG_PATH, encoding="utf-8") as f:
        assert f.read() == theme.streamlit_config(), "run: python -m app_frontend.ui.theme"


def test_charts_have_one_y_axis():
    import numpy as np
    import pandas as pd

    from app_frontend.ui import charts

    df = pd.DataFrame({"m": ["<=50K", ">50K", "c"], "v": [0.1, 0.2, 0.3], "w": [0.2, 0.1, 0.4]})
    figs = [charts.bar(df, "m", "v", "t", mode="light"), charts.grouped_bars(df, "m", ["v", "w"], "t", mode="dark"),
            charts.roc_curves({"a": (np.array([0, 1]), np.array([0, 1]), 0.5)}, "roc", mode="light"),
            charts.confusion_matrix(np.eye(2, dtype=int), ["<=50K", ">50K"], "cm", mode="light")]
    for fig in figs:
        layout = fig.to_dict()["layout"]
        assert "yaxis2" not in layout
    # '<' is escaped so Plotly does not swallow '<=50K' as an HTML tag
    assert list(figs[0].data[0].y) == ["&lt;=50K", "&gt;50K", "c"]
