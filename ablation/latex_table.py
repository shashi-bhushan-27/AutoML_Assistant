"""
Auto-generate the ablation table* (results/table_ablation.tex) from aggregated
results so that no number in the paper is transcribed by hand.

Layout follows the ablation guide's template (one row per system configuration;
metric columns, search time, SHAP runtime, notes), with one block per dataset.
"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from common import DATASET_META, DATASETS, RESULTS_DIR, primary_metrics

PM = r"$\pm$"


def _f(m, s, fmt):
    if m is None or (isinstance(m, float) and not np.isfinite(m)):
        return "---"
    if s is None or (isinstance(s, float) and not np.isfinite(s)):
        return fmt.format(m)
    return fmt.format(m) + PM + fmt.format(s)


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.0f}"


def _shap_cell(sh_cfg, ds, cfg):
    g = sh_cfg[(sh_cfg.dataset == ds) & (sh_cfg.config == cfg)] if sh_cfg is not None and len(sh_cfg) else []
    if len(g) == 0:
        return "n/a"
    r = g.iloc[0]
    return f"{r.models_ok_mean:.0f}/{r.n_models} ok"


def _shap_time(sh_summ, ds, cfg, models=None):
    if sh_summ is None or not len(sh_summ):
        return None
    g = sh_summ[(sh_summ.dataset == ds) & (sh_summ.config == cfg) & sh_summ.runtime_s_mean.notna()]
    if models is not None:
        g = g[g.model.isin(models)]
    if not len(g):
        return None
    return float(np.median(g.runtime_s_mean))


def _shap_fail_note(sh_summ, ds, cfg):
    g = sh_summ[(sh_summ.dataset == ds) & (sh_summ.config == cfg)]
    bad = []
    for _, r in g.iterrows():
        sc = json.loads(r.status_counts)
        if sc.get("ok", 0) < r.n_seeds:
            kinds = [k for k in ("timeout", "error", "nan_values", "crashed_process") if sc.get(k)]
            short = {"timeout": "t/o", "error": "err", "nan_values": "NaN", "crashed_process": "crash"}
            bad.append(f"{_abbr(r.model)} {'/'.join(short[k] for k in kinds)}")
    return ", ".join(bad) if bad else "all succeed"


def _abbr(m):
    return {"Random Forest": "RF", "XGBoost": "XGB", "Gradient Boosting": "GB", "Logistic Regression": "LR",
            "Ridge": "Ridge", "SVM": "SVM", "KNN": "KNN", "Extra Trees": "ET", "AdaBoost": "Ada",
            "Decision Tree": "DT", "Linear Regression": "LinReg", "Lasso": "Lasso", "ElasticNet": "EN"}.get(m, m)


def write_table(units, rag, pre, sh, bf_summ, path=None):
    path = path or os.path.join(RESULTS_DIR, "table_ablation.tex")
    rs = rag[2] if rag is not None else pd.DataFrame()
    psum = pre.get("summary", pd.DataFrame())
    sh_runs, sh_summ, sh_cfg = sh

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study of RAG-AutoML (mean$\pm$std over five 80/20 splits; LLM rows also "
                 r"average over 10 calls per configuration). Metric columns report the best model of the selected "
                 r"set (Accuracy/F1 for Adult and Credit Card Fraud, RMSE in \$100k/$R^2$ for California Housing). "
                 r"Search time = LLM latency + training/scoring time of the selected models (single thread). "
                 r"Valid = share of recommended names that map to an executable model. "
                 r"SHAP = model families explained within 300\,s (median runtime of the successful ones). "
                 r"Serving rows report XGBoost on the raw test rows (batch size 256). "
                 r"--- : could not complete; n/a : component not involved.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{lccccp{1.35cm}p{3.6cm}}")
    lines.append(r"\hline")
    lines.append(r"\textbf{System Configuration} & \textbf{Metric 1} & \textbf{Metric 2} & "
                 r"\textbf{Search (s)} & \textbf{Valid (\%)} & \textbf{SHAP} & \textbf{Notes} \\")
    lines.append(r"\hline")

    for ds in DATASETS:
        task = DATASET_META[ds]["task"]
        sel, sec, higher = primary_metrics(task)
        m1, m2 = ("RMSE", "$R^2$") if task == "Regression" else ("Acc.", "F1")
        f1 = "{:.4f}" if ds == "credit" else "{:.3f}"   # Credit accuracies differ only in the 4th decimal
        n_models = len([r for r in next(iter(units[ds].values()))["brute_force"]]) if units.get(ds) else 0
        lines.append(r"\multicolumn{7}{l}{\textit{" + DATASET_META[ds]["label"] + f" ({m1} / {m2})" + r"}} \\")

        def rag_row(label, cfg, note_fn=None):
            r = rs[(rs.dataset == ds) & (rs.config == cfg)] if len(rs) else []
            if len(r) == 0:
                lines.append(f"{label} & --- & --- & --- & --- & --- & not run \\\\")
                return None
            r = r.iloc[0]
            valid = r.get("valid_name_rate_after_matcher") if cfg not in ("no_fuzzy", "brute_force") else (
                r.get("valid_name_rate_raw") if cfg == "no_fuzzy" else None)
            valid_s = "n/a" if valid is None or (isinstance(valid, float) and np.isnan(valid)) else _pct(valid)
            note = note_fn(r) if note_fn else ""
            shap_s = _shap_cell(sh_cfg, ds, "full")
            lines.append(f"{label} & {_f(r.sel_mean, r.sel_std, f1)} & {_f(r.sec_mean, r.sec_std, '{:.3f}')} & "
                         f"{_f(r.search_time_s_mean, r.search_time_s_std, '{:.1f}')} & {valid_s} & {shap_s} & {note} \\\\")
            return r

        def best_note(r):
            bc = r.get("best_model_counts")
            if isinstance(bc, dict) and bc:
                top, cnt = max(bc.items(), key=lambda kv: kv[1])
                tot = sum(bc.values())
                return f"best: {_abbr(top)} ({100 * cnt / tot:.0f}\\%)"
            return ""

        full = rag_row(r"Full RAG-AutoML$^\dagger$", "full",
                       lambda r: best_note(r) + (f"; {r.n_trained_mean:.1f} models" if r.n_trained_mean else ""))
        rag_row(r"\quad w/o RAG (empty context)", "no_rag",
                lambda r: best_note(r) + f"; {r.n_trained_mean:.1f} models")
        rag_row(r"\quad w/o Meta-learning", "no_meta",
                lambda r: "prompt identical to Full" if (full is not None and full.history_in_prompt_rate == 0)
                else best_note(r))
        rag_row(r"\quad w/o Fuzzy matcher", "no_fuzzy",
                lambda r: f"run failures {100 * r.runtime_failure_rate_mean:.0f}\\%; {r.n_trained_mean:.1f} models")
        bf = rag_row(f"Brute force (all {n_models})", "brute_force",
                     lambda r: best_note(r))
        if full is not None and bf is not None:
            lines[-1] = lines[-1].replace(r" \\", "") + \
                f"; RAG saves {full.train_time_reduction_pct_mean:.0f}\\% train time \\\\"
        rag_row(r"As shipped (Llama-3.3 404)$^\ddagger$", "as_shipped",
                lambda r: f"fallback in {100 * r.advisor_fallback_rate:.0f}\\% of calls")

        # serving rows (ablation B), XGBoost, batch 256
        def serve_row(label, prep_mode, mode, bs="256"):
            g = psum[(psum.dataset == ds) & (psum.preprocessing == prep_mode) & (psum["mode"] == mode) &
                     (psum.batch_size == bs) & (psum.model == "XGBoost")] if len(psum) else []
            if len(g) == 0:
                lines.append(f"{label} & --- & --- & n/a & n/a & n/a & not run \\\\")
                return
            r = g.iloc[0]
            crash = r.crash_rate_mean
            agree = r.get("agreement_vs_offline_mean")
            if task == "Regression":
                a, b = ("served_RMSE", "served_R2")
            else:
                a, b = ("served_Accuracy", "served_F1")
            v1 = _f(r.get(a + "_mean"), r.get(a + "_std"), f1) if crash < 1 else "---"
            v2 = _f(r.get(b + "_mean"), r.get(b + "_std"), "{:.3f}") if crash < 1 else "---"
            note = f"crash {100 * crash:.0f}\\%"
            if agree is not None and np.isfinite(agree):
                note += f"; identical to offline {100 * agree:.0f}\\%"
            lines.append(f"{label} & {v1} & {v2} & n/a & n/a & n/a & {note} \\\\")

        serve_row(r"Full, served (stateful \texttt{transform})", "stateful", "app")
        serve_row(r"\quad w/o Stateful (refit per batch)", "stateless", "app")

        # SHAP rows (ablation C)
        for label, cfg in [(r"\quad w/o SHAP routing (Tree)", "forced_tree"),
                           (r"\quad w/o SHAP routing (Kernel)", "forced_kernel"),
                           (r"\quad w/o float32 pre-cast", "no_cast")]:
            if sh_summ is None or not len(sh_summ):
                lines.append(f"{label} & n/a & n/a & n/a & n/a & --- & not run \\\\")
                continue
            t = _shap_time(sh_summ, ds, cfg)
            cell = _shap_cell(sh_cfg, ds, cfg) + (f" ({t:.1f}s)" if t is not None else "")
            lines.append(f"{label} & n/a & n/a & n/a & n/a & {cell} & {_shap_fail_note(sh_summ, ds, cfg)} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\\[2pt]")
    lines.append(r"\parbox{\textwidth}{\scriptsize $^\dagger$The Groq model hard-coded in the app "
                 r"(\texttt{llama-3.3-70b-versatile}) is no longer served; the recommendation rows use "
                 r"\texttt{openai/gpt-oss-120b} at the app's temperature (0.3) with the app's prompt, retriever and "
                 r"parser unchanged. $^\ddagger$The unmodified app: every call raises \texttt{model\_not\_found} and "
                 r"\texttt{ModelAdvisor} returns its hard-coded fallback.}")
    lines.append(r"\end{table*}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
