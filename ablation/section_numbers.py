"""
Compute every number quoted in the ablation subsection from the result files
and render ablation_section.tex from ablation_section_template.tex, so that the
prose, the table and the merged paper cannot drift from the data.

    python ablation/section_numbers.py      # writes results/paper_numbers.json + ablation_section.tex
"""
import json
import os
import re

import numpy as np
import pandas as pd

from common import DATASETS, RESULTS_DIR, load_json

HERE = os.path.dirname(os.path.abspath(__file__))


def _csv(name):
    return pd.read_csv(os.path.join(RESULTS_DIR, name))


def f(x, nd=3):
    return "---" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pct(x, nd=0):
    return "---" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}"


def pm(m, s, nd=3):
    return f"{m:.{nd}f}$\\pm${s:.{nd}f}"


def compute():
    N = {}
    rs = _csv("rag_summary.csv")
    calls = _csv("rag_llm_calls.csv")
    bf = _csv("brute_force_summary.csv")
    bfm = _csv("brute_force_models.csv")
    ps = _csv("preproc_summary.csv")
    par = _csv("preproc_parity.csv")
    ep = _csv("preproc_endpoint.csv")
    lat = _csv("preproc_endpoint_latency.csv")
    shc = _csv("shap_config_summary.csv")
    shs = _csv("shap_summary.csv")
    shr = _csv("shap_runs.csv")
    checks = load_json(os.path.join(RESULTS_DIR, "code_checks.json"))
    summ = load_json(os.path.join(RESULTS_DIR, "summary.json"))

    def rr(ds, cfg):
        g = rs[(rs.dataset == ds) & (rs.config == cfg)]
        return g.iloc[0] if len(g) else None

    # ── ablation A ──
    sub = calls[calls.config != "as_shipped"]
    N["n_llm_calls_substitute"] = int(len(sub))
    N["n_llm_calls_total"] = int(len(calls))
    N["n_calls_per_cfg"] = int(calls.groupby(["dataset", "config"]).size().min())
    N["valid_raw_all"] = pct(sub.n_exact_valid.sum() / sub.n_recommended.sum())
    N["json_rate"] = pct((sub.advisor_branch == "json").mean())
    N["as_shipped_fallback"] = pct(calls[calls.config == "as_shipped"].advisor_branch.eq("exception_fallback").mean())
    for ds in DATASETS:
        full, b, ship, norag = rr(ds, "full"), rr(ds, "brute_force"), rr(ds, "as_shipped"), rr(ds, "no_rag")
        nofz = rr(ds, "no_fuzzy")
        N[f"{ds}_full_sel"] = pm(full.sel_mean, full.sel_std, 4 if ds == "credit" else 3)
        N[f"{ds}_full_sec"] = pm(full.sec_mean, full.sec_std)
        N[f"{ds}_bf_sel"] = pm(b.sel_mean, b.sel_std, 4 if ds == "credit" else 3)
        N[f"{ds}_bf_sec"] = pm(b.sec_mean, b.sec_std)
        N[f"{ds}_full_search"] = f(full.search_time_s_mean, 1)
        N[f"{ds}_bf_search"] = f(b.search_time_s_mean, 1)
        N[f"{ds}_ship_search"] = f(ship.search_time_s_mean, 1)
        N[f"{ds}_red_train"] = f(full.train_time_reduction_pct_mean, 0)
        N[f"{ds}_red_search"] = f(full.search_time_reduction_pct_mean, 0)
        N[f"{ds}_ship_red_train"] = f(ship.train_time_reduction_pct_mean, 0)
        N[f"{ds}_full_models"] = f(full.n_trained_mean, 1)
        N[f"{ds}_nofuzzy_fail"] = pct(nofz.runtime_failure_rate_mean)
        N[f"{ds}_ship_sel"] = pm(ship.sel_mean, ship.sel_std, 4 if ds == "credit" else 3)
        N[f"{ds}_ship_sec"] = pm(ship.sec_mean, ship.sec_std)
        c_full = calls[(calls.dataset == ds) & (calls.config == "full")]
        c_norag = calls[(calls.dataset == ds) & (calls.config == "no_rag")]
        N[f"{ds}_lat_full"] = f(c_full.latency_s.mean(), 2)
        N[f"{ds}_lat_norag"] = f(c_norag.latency_s.mean(), 2)
        N[f"{ds}_tok_full"] = f(c_full.total_tokens.mean(), 0)
        N[f"{ds}_tok_norag"] = f(c_norag.total_tokens.mean(), 0)
    red = [float(rr(ds, "full").train_time_reduction_pct_mean) for ds in DATASETS]
    N["red_train_min"], N["red_train_max"] = f(min(red), 0), f(max(red), 0)
    ship_red = [float(rr(ds, "as_shipped").train_time_reduction_pct_mean) for ds in DATASETS]
    N["ship_red_min"], N["ship_red_max"] = f(min(ship_red), 0), f(max(ship_red), 0)
    lat_full = calls[calls.config == "full"].latency_s.mean()
    lat_norag = calls[calls.config == "no_rag"].latency_s.mean()
    N["rag_latency_delta"] = f(lat_full - lat_norag, 1)
    N["rag_token_delta"] = f(calls[calls.config == "full"].total_tokens.mean()
                             - calls[calls.config == "no_rag"].total_tokens.mean(), 0)
    N["adult_ship_acc_drop"] = f(rr("adult", "full").sel_mean - rr("adult", "as_shipped").sel_mean, 3)

    # the brute-force winner is inside every selected set?
    ev = _csv("rag_eval_per_call_seed.csv")
    same = []
    for ds in DATASETS:
        bfw = ev[(ev.dataset == ds) & (ev.config == "brute_force")].set_index("seed").best_model
        e = ev[(ev.dataset == ds) & ev.config.isin(["full", "no_rag", "no_meta", "no_fuzzy", "full_sameprior"])]
        same.append(float(np.mean([bfw.loc[r.seed] in str(r.selected).split("|") for r in e.itertuples()])))
    N["winner_in_set_rate"] = pct(min(same))
    gaps = []
    for ds in DATASETS:
        sel = "best_RMSE" if ds == "california" else "best_Accuracy"
        bfv = ev[(ev.dataset == ds) & (ev.config == "brute_force")].set_index("seed")[sel]
        e = ev[(ev.dataset == ds) & ev.config.isin(["full", "no_rag", "no_meta", "no_fuzzy", "full_sameprior"])]
        gaps += list((e[sel].values - bfv.loc[e.seed].values) * (1 if ds == "california" else -1))
    worst = max(gaps)
    worst = 0.0 if abs(worst) < 5e-5 else worst  # avoid printing a negative zero
    N["max_metric_gap_vs_bf"] = f"{worst:.4f}"  # worst (call, split) shortfall vs brute force

    # credit: GB always recommended, its time and AUC
    gb = bf[(bf.dataset == "credit") & (bf.model == "Gradient Boosting")].iloc[0]
    N["credit_gb_time"] = f(gb["wall_s_mean"], 0)
    N["credit_gb_auc"] = f(gb["AUC-ROC_mean"], 2)
    c_cred = calls[(calls.dataset == "credit") & calls.config.isin(["full", "no_rag", "no_meta", "no_fuzzy", "full_sameprior"])]
    N["credit_gb_rate"] = pct(c_cred.recommendations.str.contains("Gradient Boosting").mean())

    probes = [p for p in checks["fuzzy_matcher_probes"] if p["task"] == "Classification"]
    N["probe_n"] = len(probes)
    N["probe_mapped"] = sum(p["match_type"] != "unmatched" for p in probes)
    N["probe_exact"] = sum(p["match_type"] == "exact" for p in probes)

    meta = checks["meta_learning_similarity"]
    N["meta_pairs_matching"] = sum(m["match"] for m in meta)
    reg = checks["regression_best_model_bookkeeping"]
    N["repo_reg_ws"] = reg.get("repo_workspaces_regression")
    N["repo_reg_ws_best"] = reg.get("repo_workspaces_regression_with_best_model")

    # ── ablation B ──
    def srow(ds, prep, mode, bs, model="XGBoost"):
        g = ps[(ps.dataset == ds) & (ps.preprocessing == prep) & (ps["mode"] == mode) &
               (ps.batch_size.astype(str) == bs) & (ps.model == model)]
        return g.iloc[0] if len(g) else None

    for ds in DATASETS:
        p0 = par[par.dataset == ds]
        N[f"{ds}_extra_cols"] = int(p0.extra_at_serving.apply(lambda s: len(eval(s)) if isinstance(s, str) else 0).max())
        N[f"{ds}_mismatch_feats"] = int(p0.n_common_features_mismatched.max())
        N[f"{ds}_cell_mismatch"] = f(100 * p0.cell_mismatch_frac.mean(), 1)
        st = srow(ds, "stateful", "app", "256")
        N[f"{ds}_stateful_crash"] = pct(st.crash_rate_mean)
        sl = {bs: srow(ds, "stateless", "app", bs) for bs in ["1", "16", "256", "full"]}
        N[f"{ds}_stateless_crash"] = {bs: pct(r.crash_rate_mean) for bs, r in sl.items()}
    # credit
    cs = srow("credit", "stateful", "app", "256")
    N["credit_stream_stateful_agree"] = pct(cs.agreement_vs_offline_mean, 1)
    N["credit_stream_f1"] = f(cs.served_F1_mean, 2)
    cf = srow("credit", "stateful", "app", "full")
    N["credit_full_stateful_agree"] = pct(cf.agreement_vs_offline_mean, 2)
    N["credit_full_off_f1"] = f(cf.offline_F1_mean, 3)
    N["credit_full_srv_f1"] = f(cf.served_F1_mean, 3)
    sl = {bs: srow("credit", "stateless", "app", bs) for bs in ["1", "16", "256", "full"]}
    N["credit_stateless_f1_small_max"] = f(max(sl[b].served_F1_mean for b in ["1", "16", "256"]), 2)
    N["credit_stateless_agree_small"] = (pct(min(sl[b].agreement_vs_stateful_app_mean for b in ["1", "16", "256"]), 0) + "--" +
                                         pct(max(sl[b].agreement_vs_stateful_app_mean for b in ["1", "16", "256"]), 0))
    N["credit_stateless_full_f1"] = f(sl["full"].served_F1_mean, 2)
    N["credit_stateless_crash16"] = pct(sl["16"].crash_rate_mean)
    # adult (aligned diagnostic)
    al1 = srow("adult", "stateful", "aligned", "1")
    N["adult_aligned1_agree"] = pct(al1.agreement_vs_offline_mean)
    N["adult_aligned1_f1"] = f(al1.served_F1_mean, 2)
    N["adult_off_f1"] = f(al1.offline_F1_mean, 2)
    sla = {bs: srow("adult", "stateless", "aligned", bs) for bs in ["1", "16", "256", "full"]}
    N["adult_stateless_aligned_agree"] = (pct(min(r.agreement_vs_stateful_aligned_mean for r in sla.values())) + "--" +
                                         pct(max(r.agreement_vs_stateful_aligned_mean for r in sla.values())))
    N["adult_stateless_aligned_f1_max"] = f(max(r.served_F1_mean for r in sla.values()), 2)
    # stateful + alignment, tree models, batches >= 256 (diagnostic)
    tree_al = ps[(ps.preprocessing == "stateful") & (ps["mode"] == "aligned") &
                 ps.batch_size.astype(str).isin(["256", "full"]) & ps.model.isin(["XGBoost", "Random Forest"])]
    N["tree_aligned_min_agree"] = pct(tree_al.agreement_vs_offline_mean.min(), 2)
    # california
    c = {bs: srow("california", "stateless", "app", bs) for bs in ["1", "16", "256", "full"]}
    N["ca_stateless_crash16"] = pct(c["16"].crash_rate_mean)
    N["ca_stateless_r2_256"] = f(c["256"].served_R2_mean, 2)
    N["ca_stateless_r2_full"] = f(c["full"].served_R2_mean, 2)
    N["ca_off_r2"] = f(c["256"].offline_R2_mean, 2)
    N["ca_stateless_mad_256"] = f(c["256"].mean_abs_pred_diff_mean * 100, 0)  # $ thousands (target in $100k)
    # endpoint
    codes = ep.groupby("dataset").status_code.agg(lambda s: dict(s.value_counts()))
    N["api_codes"] = {ds: {str(k): int(v) for k, v in codes.get(ds, {}).items()} for ds in DATASETS}
    N["api_adult_500_rate"] = pct((ep[ep.dataset == "adult"].status_code == 500).mean())
    N["api_ca_500_rate"] = pct((ep[ep.dataset == "california"].status_code == 500).mean())
    cr = ep[(ep.dataset == "credit")]
    N["api_credit_baseline_200"] = pct((cr[cr.variant == "baseline"].status_code == 200).mean())
    N["api_credit_perturbed_500"] = pct((cr[cr.variant != "baseline"].status_code == 500).mean())
    lc = lat[lat.dataset == "credit"].groupby("model").median_ms.mean()
    N["lat_credit_min"], N["lat_credit_max"] = f(lc.min(), 0), f(lc.max(), 0)
    lf = lat[lat.dataset != "credit"].groupby(["dataset", "model"]).median_ms.mean()
    N["lat_fail_min"], N["lat_fail_max"] = f(lf.min(), 0), f(lf.max(), 0)
    N["pipeline_pkl_credit_mb"] = f(lat[lat.dataset == "credit"].pipeline_pkl_mb.mean(), 0)
    N["models_pkl_ca_mb"] = f(lat[lat.dataset == "california"].trained_models_pkl_mb.mean(), 0)
    N["model_only_ms"] = f(lat.model_only_median_ms.median(), 1)

    # ── ablation C ──
    for ds in DATASETS:
        for cfg in ["full", "forced_tree", "forced_kernel", "no_cast"]:
            g = shc[(shc.dataset == ds) & (shc.config == cfg)]
            N[f"shap_{ds}_{cfg}_ok"] = f(g.models_ok_mean.iloc[0], 1).rstrip("0").rstrip(".") if len(g) else "---"
    full_rates = shc[shc.config == "full"].set_index("dataset").success_rate
    N["shap_full_rate_min"], N["shap_full_rate_max"] = pct(full_rates.min()), pct(full_rates.max())

    def srt(ds, model, cfg):
        g = shs[(shs.dataset == ds) & (shs.model == model) & (shs.config == cfg)]
        return g.iloc[0] if len(g) else None

    N["svm_extrap_adult_h"] = f(srt("adult", "SVM", "full").extrapolated_runtime_s_mean / 3600, 1)
    N["svm_extrap_credit_min"] = f(srt("credit", "SVM", "full").extrapolated_runtime_s_mean / 60, 0)
    N["svm_extrap_ca_min"] = f(srt("california", "SVM", "full").extrapolated_runtime_s_mean / 60, 1)
    gbc = srt("credit", "Gradient Boosting", "full")
    N["credit_gb_tree_fail"] = int(json.loads(gbc.status_counts).get("error", 0))
    N["credit_gb_tree_n"] = int(gbc.n_seeds)
    N["adult_rf_tree_s"] = f(srt("adult", "Random Forest", "full").runtime_s_mean, 0)
    N["adult_gb_tree_s"] = f(srt("adult", "Gradient Boosting", "full").runtime_s_mean, 2)
    N["adult_gb_kernel_s"] = f(srt("adult", "Gradient Boosting", "forced_kernel").runtime_s_mean, 0)
    N["adult_lr_lin_s"] = f(srt("adult", "Logistic Regression", "full").runtime_s_mean, 1)
    N["adult_lr_kernel_s"] = f(srt("adult", "Logistic Regression", "forced_kernel").runtime_s_mean, 0)
    N["adult_rf_kernel_s"] = f(srt("adult", "Random Forest", "forced_kernel").runtime_s_mean, 0)
    N["ca_rf_tree_s"] = f(srt("california", "Random Forest", "full").runtime_s_mean, 0)
    N["ca_rf_kernel_s"] = f(srt("california", "Random Forest", "forced_kernel").runtime_s_mean, 0)
    N["ca_knn_kernel_s"] = f(srt("california", "KNN", "full").runtime_s_mean, 0)
    # float32 cast and KNN (rows finished within the limit)
    kn = shr[(shr.model == "KNN") & shr.config.isin(["full", "no_cast"]) & (shr.dataset == "adult")]
    rows = kn.groupby("config").kernel_rows_done.mean()
    N["knn_rows_cast"] = f(rows.get("full", np.nan), 0)
    N["knn_rows_nocast"] = f(rows.get("no_cast", np.nan), 0)
    ok = shr[shr.status == "ok"]
    tl = ok[ok.explainer.isin(["TreeExplainer", "LinearExplainer", "Tree", "Linear"])]
    N["mem_tree_linear_max_mb"] = f(tl.peak_mem_mb.max(), 0)
    kk = ok[ok.explainer.astype(str).str.contains("Kernel")]
    N["mem_kernel_ok_max_mb"] = f(kk.peak_mem_mb.max(), 0)
    mem = kn.groupby("config").peak_mem_mb.mean()
    N["knn_mem_cast_gb"] = f(mem.get("full", np.nan) / 1000, 1)
    N["knn_mem_nocast_gb"] = f(mem.get("no_cast", np.nan) / 1000, 2)
    msg = shr[(shr.dataset == "credit") & (shr.model == "Gradient Boosting") & (shr.config == "full") &
              (shr.status == "error")].message.dropna()
    m = re.search(r"sum of the SHAP values was (-?\d+(?:\.\d+)?), while the model output was (-?\d+(?:\.\d+)?)", " ".join(msg))
    if m:
        mant, exp = f"{float(m.group(1)):.1e}".split("e")
        N["gb_additivity_sum"] = f"{mant}\\times10^{{{int(exp)}}}"
    else:
        N["gb_additivity_sum"] = "---"
    N["gb_additivity_out"] = f(float(m.group(2)), 1) if m else "---"
    cap = checks.get("adult_capital_columns", {})
    N["adult_xgb_app_acc"] = f(cap.get("app_features_mean"), 3) if cap else "---"
    N["adult_xgb_app_acc_std"] = f(cap.get("app_features_std"), 3) if cap else "---"
    N["adult_xgb_rawcap_acc"] = f(cap.get("with_raw_capital_columns_mean"), 3) if cap else "---"
    N["adult_xgb_rawcap_acc_std"] = f(cap.get("with_raw_capital_columns_std"), 3) if cap else "---"
    nc = checks.get("shap_no_cast_linear", {})
    N["nocast_linear_error"] = (nc.get("no_cast") or {}).get("error", "---") if isinstance(nc.get("no_cast"), dict) else nc.get("no_cast")

    # ── claims ──
    def best(ds, metric, higher=True):
        g = bf[bf.dataset == ds].dropna(subset=[metric + "_mean"])
        r = g.sort_values(metric + "_mean", ascending=not higher).iloc[0]
        return r
    ba = best("adult", "Accuracy")
    N["adult_best_acc"], N["adult_best_acc_std"], N["adult_best_acc_model"] = f(ba["Accuracy_mean"]), f(ba["Accuracy_std"]), ba.model
    bc = best("credit", "Accuracy")
    N["credit_best_acc"], N["credit_best_acc_model"] = f(bc["Accuracy_mean"], 4), bc.model
    bcf = best("credit", "F1 Score")
    N["credit_best_f1"], N["credit_best_f1_model"] = f(bcf["F1 Score_mean"]), bcf.model
    cred = bf[bf.dataset == "credit"]
    N["credit_acc_min"], N["credit_acc_max"] = f(cred["Accuracy_mean"].min(), 4), f(cred["Accuracy_mean"].max(), 4)
    units = [load_json(p) for p in sorted(__import__("glob").glob(os.path.join(RESULTS_DIR, "raw", "unit_credit_s*.json")))]
    N["credit_majority_acc"] = f(1 - 492 / 284807, 4)
    bca = best("california", "R²")
    N["california_best_r2"], N["california_best_r2_std"], N["california_best_r2_model"] = f(bca["R²_mean"]), f(bca["R²_std"]), bca.model
    N["california_best_rmse"] = f(bca["RMSE_mean"], 3)
    N["california_best_rmse_usd"] = f"{bca['RMSE_mean'] * 100000:,.0f}"
    xg = bf[(bf.dataset == "california") & (bf.model == "XGBoost")].iloc[0]
    N["california_xgb_r2"] = f(xg["R²_mean"])
    xa = bf[(bf.dataset == "adult") & (bf.model == "XGBoost")].iloc[0]
    N["adult_xgb_acc"] = f(xa["Accuracy_mean"])
    rl = checks.get("retrieval_latency", {}).get("per_dataset", {})
    if rl:
        v = [x["median_ms"] for x in rl.values()]
        N["retrieval_ms_min"], N["retrieval_ms_max"] = f(min(v), 0), f(max(v), 0)
    pre = {ds: summ["headline"]["measured"].get(f"{ds}_predict_latency_ms_median") for ds in DATASETS}
    N["predict_latency_by_dataset"] = pre
    return N


def flag_values(N):
    """Short strings used by merge_paper.py for the comment-only flags."""
    return {
        "adult_best_acc": N["adult_best_acc"], "adult_best_acc_model": N["adult_best_acc_model"],
        "adult_capital_note": f"IQR capping sets capital-gain/capital-loss to 0 and the selector drops them; restoring "
                              f"them lifts XGBoost from {N['adult_xgb_app_acc']} to {N['adult_xgb_rawcap_acc']}",
        "credit_best_acc": N["credit_best_acc"], "credit_majority_acc": N["credit_majority_acc"],
        "california_best_r2": N["california_best_r2"], "california_best_r2_model": N["california_best_r2_model"],
        "california_best_rmse": N["california_best_rmse"],
        "reduction_by_dataset": f"Adult {N['adult_red_train']}%, Credit {N['credit_red_train']}%, California "
                                f"{N['california_red_train']}% (training time; brute force = all supported models, "
                                f"single thread)",
        "latency_by_dataset": f"Adult and California: HTTP 500 for every request (schema mismatch); Credit: HTTP 200, "
                              f"median {N['lat_credit_min']}-{N['lat_credit_max']} ms in-process",
        "parity_summary": f"transform() skips outlier caps, skew transforms and feature selection; Adult/California "
                          f"serve {N['adult_extra_cols']}/{N['california_extra_cols']} extra columns and crash on every "
                          f"batch, Credit serves all 29 features with different values for capped rows",
        "shap_full_summary": f"shipped routing explains {N['shap_adult_full_ok']}/6, {N['shap_california_full_ok']}/6 and "
                             f"{N['shap_credit_full_ok']}/6 model families (Adult/California/Credit) within 300 s; SVM "
                             f"and KNN time out, GB fails the additivity check on Credit",
        "meta_summary": "meta-learning never activated: no benchmark pair satisfies the workspace-similarity rule, and "
                        "regression workspaces never store a best model",
        "valid_summary": f"with a working LLM, {N['valid_raw_all']}% of names were valid before any matching; the "
                         f"shipped model is unavailable, so the shipped app always uses the hard-coded fallback",
        "test_rows": "9,758 (Adult), 55,133 (Credit) and 4,128 (California)",
        "retrieval_ms": f"{N.get('retrieval_ms_min', '?')}-{N.get('retrieval_ms_max', '?')} ms median",
    }


def render(N):
    tpl = open(os.path.join(HERE, "ablation_section_template.tex"), encoding="utf-8").read()
    table = open(os.path.join(RESULTS_DIR, "table_ablation.tex"), encoding="utf-8").read().rstrip()

    def sub(m):
        key = m.group(1)
        if key == "TABLE":
            return table
        path = key.split(".")
        v = N
        for p_ in path:
            v = v[p_]
        return str(v)

    out = re.sub(r"<<([A-Za-z0-9_.]+)>>", sub, tpl)
    missing = re.findall(r"<<[^>]+>>", out)
    assert not missing, missing
    with open(os.path.join(HERE, "ablation_section.tex"), "w", encoding="utf-8") as fh:
        fh.write(out)


def main():
    N = compute()
    with open(os.path.join(RESULTS_DIR, "paper_numbers.json"), "w") as fh:
        json.dump(N, fh, indent=2, default=str)
    summ_p = os.path.join(RESULTS_DIR, "summary.json")
    s = load_json(summ_p)
    s["flag_values"] = flag_values(N)
    with open(summ_p, "w") as fh:
        json.dump(s, fh, indent=2, default=str)
    if os.path.exists(os.path.join(HERE, "ablation_section_template.tex")):
        render(N)
        print("wrote ablation_section.tex")
    print("wrote results/paper_numbers.json")


if __name__ == "__main__":
    main()
