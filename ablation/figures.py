"""
Figures for the ablation subsection (PDF + PNG at 300 dpi).

Colour: categorical slots validated with the dataviz palette validator
(adjacent CVD dE >= 9.1 on white; the 4-series line set passes all-pairs).
Identity is never colour-only: x-axis labels, markers and line styles carry it too.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DATASET_META, DATASETS, FIG_DIR, RESULTS_DIR  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"ok": "#0ca30c", "timeout": "#fab219", "error": "#d03b3b", "nan_values": "#ec835a",
          "crashed_process": "#d03b3b"}

RAG_CFGS = [("full", "Full"), ("no_rag", "w/o RAG"), ("no_meta", "w/o Meta"), ("no_fuzzy", "w/o Fuzzy"),
            ("as_shipped", "As shipped"), ("brute_force", "Brute force")]
RAG_COLORS = {"full": SLOT[0], "no_rag": SLOT[1], "no_meta": SLOT[2], "no_fuzzy": SLOT[3],
              "as_shipped": SLOT[4], "brute_force": MUTED}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _read(name):
    p = os.path.join(RESULTS_DIR, name)
    try:
        return pd.read_csv(p)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


# ───────────────────────────── Figure: RAG ablation ─────────────────────────────

def fig_rag():
    ev = _read("rag_eval_per_call_seed.csv")
    if ev.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    for j, ds in enumerate(DATASETS):
        task = DATASET_META[ds]["task"]
        qcol, qlab = ("best_R²", "$R^2$ of best selected model") if task == "Regression" else \
            ("best_F1 Score", "F1 of best selected model")
        cfgs = [(c, l) for c, l in RAG_CFGS if ((ev.dataset == ds) & (ev.config == c)).any()]
        x = np.arange(len(cfgs))
        qm, qs, tm, ts = [], [], [], []
        for c, _ in cfgs:
            g = ev[(ev.dataset == ds) & (ev.config == c)].groupby("seed")
            q = g[qcol].mean()
            t = g["search_time_s"].mean()
            qm.append(q.mean()); qs.append(q.std(ddof=1) if len(q) > 1 else 0)
            tm.append(t.mean()); ts.append(t.std(ddof=1) if len(t) > 1 else 0)
        colors = [RAG_COLORS[c] for c, _ in cfgs]
        labels = [l for _, l in cfgs]

        ax = axes[0, j]
        for xi, m, s, col in zip(x, qm, qs, colors):
            ax.errorbar(xi, m, yerr=s, fmt="o", ms=5, color=col, ecolor=INK2, elinewidth=0.8, capsize=2,
                        markeredgecolor="white", markeredgewidth=0.8)
        lo, hi = np.nanmin(np.array(qm) - np.array(qs)), np.nanmax(np.array(qm) + np.array(qs))
        pad = max(0.01, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(DATASET_META[ds]["label"], color=INK, loc="left", fontweight="bold")
        ax.set_ylabel(qlab if j == 0 else qlab.split(" of")[0])
        ax.grid(axis="x", visible=False)

        ax = axes[1, j]
        ax.bar(x, tm, yerr=ts, color=colors, width=0.62, edgecolor="white", linewidth=1.0,
               error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 2})
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Search time (s, log)" if j == 0 else "")
        ax.grid(axis="x", visible=False)
        for xi, m in zip(x, tm):
            ax.text(xi, m * 1.12, f"{m:.0f}" if m >= 10 else f"{m:.1f}", ha="center", va="bottom",
                    fontsize=5.8, color=INK2)
        ax.set_ylim(min(tm) / 3, max(tm) * 4)  # explicit: log-scale bars need a visible floor
    _save(fig, "fig_ablation_rag")


# ───────────────────────── Figure: preprocessing / serving ─────────────────────────

SERVE_LINES = [("stateful", "app", "Stateful, as shipped", SLOT[0], "o", "-"),
               ("stateful", "aligned", "Stateful + column alignment*", SLOT[6], "D", "--"),
               ("stateless", "app", "Stateless (refit per batch)", SLOT[1], "s", "-"),
               ("stateless", "aligned", "Stateless + column alignment*", SLOT[2], "^", "--")]


def fig_preproc(model="XGBoost"):
    sv = _read("preproc_serving.csv")
    if sv.empty:
        return
    bss = ["1", "16", "256", "full"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.7), gridspec_kw={"hspace": 0.5, "wspace": 0.32})
    for j, ds in enumerate(DATASETS):
        task = DATASET_META[ds]["task"]
        d = sv[(sv.dataset == ds) & (sv.model == model)].copy()
        d["batch_size"] = d["batch_size"].astype(str)
        d["identical"] = d["row_coverage"] * d["agreement_vs_offline"].fillna(0)
        qa = "served_R2" if task == "Regression" else "served_F1"
        qo = "offline_R2" if task == "Regression" else "offline_F1"
        d["delta"] = d[qa] - d[qo]
        for k, (prep, mode, label, col, mk, ls) in enumerate(SERVE_LINES):
            g = d[(d.preprocessing == prep) & (d["mode"] == mode)]
            if g.empty:
                continue
            agg = g.groupby("batch_size").agg(im=("identical", "mean"), isd=("identical", "std"),
                                              dm=("delta", "mean"), dsd=("delta", "std"),
                                              crash=("crash_rate", "mean")).reindex(bss)
            xs = np.arange(len(bss)) + (k - 1.5) * 0.08   # dodge so coincident series stay visible
            axes[0, j].errorbar(xs, 100 * agg.im, yerr=100 * agg.isd.fillna(0), color=col, marker=mk, ms=4,
                                ls=ls, lw=1.4, capsize=2, elinewidth=0.7, label=label,
                                markeredgecolor="white", markeredgewidth=0.6)
            crashed = (agg.crash >= 1).values
            if crashed.any():  # every batch crashed: mark explicitly instead of a silent zero
                axes[0, j].scatter(xs[crashed], np.zeros(crashed.sum()), marker="x", s=22, color=col,
                                   linewidths=1.2, zorder=5)
            ok = agg.crash < 1
            axes[1, j].errorbar(xs[ok.values], agg.dm[ok], yerr=agg.dsd[ok].fillna(0), color=col, marker=mk, ms=4,
                                ls=ls, lw=1.4, capsize=2, elinewidth=0.7, markeredgecolor="white",
                                markeredgewidth=0.6)
        ax = axes[0, j]
        ax.set_title(DATASET_META[ds]["label"], color=INK, loc="left", fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.set_xticks(range(len(bss)))
        ax.set_xticklabels(bss)
        ax.set_ylabel("Rows served with the\noffline prediction (%)" if j == 0 else "")
        ax = axes[1, j]
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks(range(len(bss)))
        ax.set_xticklabels(bss)
        ax.set_xlabel("Serving batch size")
        ax.set_ylabel(("$\\Delta R^2$" if task == "Regression" else "$\\Delta$F1") + " vs offline")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.1, "* diagnostic only (not app behaviour): output re-indexed to the training columns, "
             "missing columns filled with 0.   \u00d7 = every batch crashed (no row served).", ha="center",
             fontsize=6, color=INK2)
    _save(fig, "fig_ablation_preprocessing")


# ─────────────────────────────── Figure: SHAP matrix ───────────────────────────────

SHAP_CFGS = [("full", "Full\n(routing+f32)"), ("forced_tree", "Forced\nTree"), ("forced_kernel", "Forced\nKernel"),
             ("no_cast", "No f32\ncast")]


def _tint(hex_color, a=0.38):
    c = matplotlib.colors.to_rgb(hex_color)
    return tuple(1 - a * (1 - v) for v in c)


def fig_shap():
    runs = _read("shap_runs.csv")
    if runs.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.42})
    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        d = runs[runs.dataset == ds]
        models = list(dict.fromkeys(d.model))
        ax.set_xlim(-0.5, len(SHAP_CFGS) - 0.5)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.grid(False)
        for i, m in enumerate(models):
            for k, (cfg, _) in enumerate(SHAP_CFGS):
                g = d[(d.model == m) & (d.config == cfg)]
                if g.empty:
                    continue
                n, n_ok = len(g), int((g.status == "ok").sum())
                dominant = g.status.value_counts().index[0]
                face = _tint(STATUS.get(dominant, MUTED), 0.45 if dominant != "ok" else 0.35)
                ax.add_patch(plt.Rectangle((k - 0.47, i - 0.45), 0.94, 0.9, facecolor=face, edgecolor="white", lw=1.5))
                if dominant == "ok":
                    t = g[g.status == "ok"].runtime_s.median()
                    txt = f"ok {t:.1f}s" if t < 100 else f"ok {t:.0f}s"
                elif dominant == "timeout":
                    txt = "timeout"
                    if "extrapolated_runtime_s" in g and g.extrapolated_runtime_s.notna().any():
                        txt += f"\n~{g.extrapolated_runtime_s.median() / 60:.0f} min"
                elif dominant == "nan_values":
                    txt = "NaN out"
                else:
                    txt = "error"
                if n_ok not in (0, n):
                    txt += f" ({n_ok}/{n})"
                ax.text(k, i, txt, ha="center", va="center", fontsize=5.6, color=INK, linespacing=0.95)
        ax.set_xticks(range(len(SHAP_CFGS)))
        ax.set_xticklabels([l for _, l in SHAP_CFGS], fontsize=5.8)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(DATASET_META[ds]["label"], color=INK, loc="left", fontweight="bold", pad=22)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=_tint(STATUS[s], 0.35 if s == "ok" else 0.45), edgecolor=AXIS,
                             lw=0.4) for s in ("ok", "timeout", "error")]
    fig.legend(handles, ["ok (median runtime, 200 rows)", "timeout (> 300 s; ~ = extrapolated)",
                         "error (explainer raised)"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    _save(fig, "fig_ablation_shap")


# ─────────────────────── Figure: measured vs claimed performance ───────────────────────

def fig_claims():
    bf = _read("brute_force_models.csv")
    if bf.empty:
        return
    panels = [("adult", "Accuracy", 0.904, "Paper: 0.904"), ("credit", "F1 Score", None, None),
              ("california", "R²", 0.884, "Paper: 0.884")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"wspace": 0.55})
    for ax, (ds, metric, claim, claim_label) in zip(axes, panels):
        d = bf[(bf.dataset == ds) & bf[metric].notna()]
        agg = d.groupby("Model")[metric].agg(["mean", "std"]).sort_values("mean")
        y = np.arange(len(agg))
        ax.errorbar(agg["mean"], y, xerr=agg["std"].fillna(0), fmt="o", ms=4, color=SLOT[0], ecolor=INK2,
                    elinewidth=0.7, capsize=1.5, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(agg.index)
        ax.grid(axis="y", visible=False)
        lab = {"Accuracy": "Accuracy", "F1 Score": "F1 (fraud class)", "R²": "$R^2$"}[metric]
        ax.set_xlabel(lab)
        ax.set_title(DATASET_META[ds]["label"], color=INK, loc="left", fontweight="bold")
        if claim is not None:
            ax.axvline(claim, color=SLOT[7], lw=1.1)
            ax.text(claim, len(agg) - 0.4, claim_label, color=SLOT[7], fontsize=6, ha="right", va="bottom")
            lo = min(agg["mean"].min(), claim)
            ax.set_xlim(max(lo - 0.08, agg["mean"].min() - 0.1), max(claim, agg["mean"].max()) + 0.02)
    _save(fig, "fig_measured_vs_claimed")


def make_all_figures():
    _style()
    fig_rag()
    fig_preproc()
    fig_shap()
    fig_claims()
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    make_all_figures()
