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
    to, err = [], []
    for _, r in g.iterrows():
        sc = json.loads(r.status_counts)
        if sc.get("ok", 0) == r.n_seeds:
            continue
        (to if sc.get("timeout", 0) >= sc.get("error", 0) else err).append(_abbr(r.model))
    parts = ([("t/o " + " ".join(to))] if to else []) + ([("err " + " ".join(err))] if err else [])
    return "; ".join(parts) if parts else "all succeed"


def _abbr(m):
    return {"Random Forest": "RF", "XGBoost": "XGB", "Gradient Boosting": "GB", "Logistic Regression": "LR",
            "Ridge": "Ridge", "SVM": "SVM", "KNN": "KNN", "Extra Trees": "ET", "AdaBoost": "Ada",
            "Decision Tree": "DT", "Linear Regression": "LinReg", "Lasso": "Lasso", "ElasticNet": "EN"}.get(m, m)


def write_table(units, rag, pre, sh, bf_summ, path=None):
    path = path or os.path.join(RESULTS_DIR, "table_ablation.tex")
    rs = rag[2] if rag is not None else pd.DataFrame()
    psum = pre.get("summary", pd.DataFrame())
    sh_runs, sh_summ, sh_cfg = sh

    def shap_cell(ds, cfg):
        g = sh_cfg[(sh_cfg.dataset == ds) & (sh_cfg.config == cfg)] if sh_cfg is not None and len(sh_cfg) else []
        if len(g) == 0:
            return "n/a"
        r = g.iloc[0]
        ok = f"{r.models_ok_mean:.1f}".rstrip("0").rstrip(".")
        t = _shap_time(sh_summ, ds, cfg)
        if t is None:
            return f"{ok}/{r.n_models}"
        ts = f"{t:.2f}" if t < 1 else (f"{t:.1f}" if t < 10 else f"{t:.0f}")
        return f"{ok}/{r.n_models} ({ts}\\,s)"

    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Ablation study of RAG-AutoML (mean$\pm$std over five splits; LLM rows also average 10 calls "
             r"each). Metric~1/2: Accuracy/F1 (Adult, Credit) or RMSE in \$100k/$R^2$ (California) of the best model "
             r"in the selected set; serving rows ($\Delta$) give the change against the offline predictions on the same "
             r"rows (XGBoost, batches of 256). Search: LLM latency plus training of the selected models on one thread. "
             r"SHAP: model families explained within 300\,s, out of six (median runtime). All LLM-returned names were "
             r"valid. ---: could not complete; n/a: not involved.}")
    L.append(r"\label{tab:ablation}")
    L.append(r"\scriptsize")
    L.append(r"\setlength{\tabcolsep}{1.8pt}")
    L.append(r"\begin{tabular}{@{}lcccl>{\raggedright\arraybackslash}p{2.0cm}@{}}")
    L.append(r"\hline")
    L.append(r"\textbf{System Configuration} & \textbf{Metric 1} & \textbf{Metric 2} & \textbf{Search (s)} & "
             r"\textbf{SHAP} & \textbf{Notes} \\")
    L.append(r"\hline")

    for ds in DATASETS:
        task = DATASET_META[ds]["task"]
        m1, m2 = ("RMSE", "$R^2$") if task == "Regression" else ("Acc.", "F1")
        f1 = "{:.4f}" if ds == "credit" else "{:.3f}"   # Credit accuracies differ only in the 4th decimal
        n_models = len(next(iter(units[ds].values()))["brute_force"]) if units.get(ds) else 0
        L.append(r"\multicolumn{6}{@{}l}{\textit{" + DATASET_META[ds]["label"] + f" ({m1} / {m2})" + r"}} \\")
        full_shap = shap_cell(ds, "full")

        def rag_row(label, cfg, note):
            r = rs[(rs.dataset == ds) & (rs.config == cfg)] if len(rs) else []
            if len(r) == 0:
                L.append(f"{label} & --- & --- & --- & --- & not run \\\\")
                return None
            r = r.iloc[0]
            L.append(f"{label} & {_f(r.sel_mean, r.sel_std, f1)} & {_f(r.sec_mean, r.sec_std, '{:.3f}')} & "
                     f"{_f(r.search_time_s_mean, r.search_time_s_std, '{:.1f}')} & {full_shap} & {note(r)} \\\\")
            return r

        def best(r):
            bc = r.get("best_model_counts")
            if isinstance(bc, dict) and bc:
                top, cnt = max(bc.items(), key=lambda kv: kv[1])
                return f"best {_abbr(top)} ({100 * cnt / sum(bc.values()):.0f}\\%)"
            return ""

        full = rag_row(r"Full RAG-AutoML$^\dagger$", "full", lambda r: f"{r.n_trained_mean:.1f} models")
        rag_row(r"\quad w/o RAG", "no_rag", lambda r: f"{r.n_trained_mean:.1f} models")
        rag_row(r"\quad w/o Meta-learning", "no_meta",
                lambda r: "same prompt" if (full is not None and full.history_in_prompt_rate == 0)
                else f"{r.n_trained_mean:.1f} models")
        rag_row(r"\quad w/o Fuzzy match", "no_fuzzy",
                lambda r: f"{100 * r.runtime_failure_rate_mean:.0f}\\% fail, {r.n_trained_mean:.1f} models")
        bf = rag_row(f"Brute force (all {n_models})", "brute_force", lambda r: "")
        if full is not None and bf is not None:
            L[-1] = L[-1][:-3] + f"Full: $-${full.train_time_reduction_pct_mean:.0f}\\% time \\\\"
        rag_row(r"As shipped (no LLM)$^\ddagger$", "as_shipped",
                lambda r: f"fallback {100 * r.advisor_fallback_rate:.0f}\\%")

        def serve_row(label, prep_mode, mode="app", bs="256"):
            g = psum[(psum.dataset == ds) & (psum.preprocessing == prep_mode) & (psum["mode"] == mode) &
                     (psum.batch_size.astype(str) == bs) & (psum.model == "XGBoost")] if len(psum) else []
            if len(g) == 0:
                L.append(f"{label} & --- & --- & n/a & n/a & not run \\\\")
                return
            r = g.iloc[0]
            k1, k2 = ("RMSE", "R2") if task == "Regression" else ("Accuracy", "F1")
            crash = r.crash_rate_mean

            def d(k):
                m, sd = r.get(k + "_change_mean"), r.get(k + "_change_std")
                if crash >= 1 or m is None or not np.isfinite(m):
                    return "---"
                return ("$+$" if m >= 0 else "$-$") + f"{abs(m):.3f}" + PM + f"{sd:.3f}"
            note = f"crash {100 * crash:.0f}\\%"
            agree = r.get("agreement_vs_offline_mean")
            if crash < 1 and agree is not None and np.isfinite(agree):
                note += f"; {100 * agree:.0f}\\% same"
            L.append(f"{label} & {d(k1)} & {d(k2)} & n/a & n/a & {note} \\\\")

        serve_row(r"Served, stateful ($\Delta$)", "stateful")
        serve_row(r"\quad w/o Stateful ($\Delta$)", "stateless")

        for label, cfg in [(r"\quad w/o routing (Tree)", "forced_tree"),
                           (r"\quad w/o routing (Kernel)", "forced_kernel"),
                           (r"\quad w/o float32 cast", "no_cast")]:
            if sh_summ is None or not len(sh_summ):
                L.append(f"{label} & n/a & n/a & n/a & --- & not run \\\\")
                continue
            L.append(f"{label} & n/a & n/a & n/a & {shap_cell(ds, cfg)} & {_shap_fail_note(sh_summ, ds, cfg)} \\\\")
        L.append(r"\hline")

    L.append(r"\end{tabular}")
    L.append(r"\par\smallskip")
    L.append(r"\parbox{\textwidth}{\scriptsize $^\dagger$The Groq model hard-coded in the application "
             r"(\texttt{llama-3.3-70b-versatile}) is no longer served; the recommendation rows use "
             r"\texttt{openai/\allowbreak gpt-oss-120b} at the application's temperature (0.3) with its prompt, "
             r"retriever and parser unchanged. $^\ddagger$The unmodified advisor: every call fails with "
             r"\texttt{model\_not\_found} and the hard-coded fallback (XGBoost, Random Forest; plus Linear "
             r"Regression for regression) is trained.}")
    L.append(r"\end{table*}")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path
