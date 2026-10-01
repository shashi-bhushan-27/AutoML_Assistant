"""
Aggregate raw unit/LLM results into results/*.csv, results/*.json and the
auto-generated LaTeX table results/table_ablation.tex.

Convention: every "mean ± std" is over the 5 split seeds (sample std, ddof=1).
For LLM configurations the per-seed value is first averaged over the LLM calls.
"""
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from common import DATASET_META, DATASETS, RAW_DIR, RESULTS_DIR, dump_json, load_json, primary_metrics

PAPER_CLAIMS = {
    "accuracy_ai_selected_ensemble": 0.904,
    "f1_ai_selected_ensemble": 0.889,
    "r2_ai_selected_ensemble": 0.884,
    "search_compute_reduction_pct": 86.4,
    "rag_search_time_s": 7.85,
    "brute_force_search_time_s": 57.8,
    "predict_latency_ms": 12.4,
    "shap_tree_runtime_s_2k_rows": 0.42,
    "shap_kernel_runtime_s_50_bg": 3.85,
    "validated_algorithm_rate_pct": 100.0,
    "shap_routing_success_pct": 100.0,
    "xgboost_r2_california": 0.870,
    "xgboost_accuracy": 0.896,
}


def ms(values, fmt="{:.3f}"):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None, None, "---"
    m = float(v.mean())
    s = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return m, s, (fmt.format(m) + r"$\pm$" + fmt.format(s))


def load_units():
    units = {ds: {} for ds in DATASETS}
    for p in sorted(glob.glob(os.path.join(RAW_DIR, "unit_*_s*.json"))):
        u = load_json(p)
        units[u["dataset"]][int(u["seed"])] = u
    return units


def load_llm():
    p = os.path.join(RAW_DIR, "llm_calls.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f]


# ─────────────────────────────── brute force ────────────────────────────────

def brute_force_tables(units):
    rows = []
    for ds, us in units.items():
        for seed, u in us.items():
            for r in u["brute_force"]:
                rows.append({"dataset": ds, "seed": seed, **{k: v for k, v in r.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "brute_force_models.csv"), index=False)
    summ = []
    for (ds, m), g in df.groupby(["dataset", "Model"]):
        rec = {"dataset": ds, "model": m, "n_seeds": len(g), "errors": int(g["Error"].notna().sum()) if "Error" in g else 0}
        for col in ["Accuracy", "F1 Score", "AUC-ROC", "MCC", "RMSE", "MAE", "R²", "Time (s)", "wall_s"]:
            if col in g and g[col].notna().any():
                rec[col + "_mean"], rec[col + "_std"], _ = ms(g[col].astype(float))
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RESULTS_DIR, "brute_force_summary.csv"), index=False)
    return df, summ


# ─────────────────────────────── ablation A ─────────────────────────────────

def rag_tables(units, llm):
    from rag_ablation import brute_force_rows, evaluate_calls, supported_models_for

    calls = []
    for r in llm:
        ds = r["dataset"]
        sup = supported_models_for(DATASET_META[ds]["task"])
        out = r["advisor_output"]
        names = out.get("recommendations", []) if isinstance(out, dict) else []
        names = names if isinstance(names, list) else [names]
        usage = next(iter((r.get("usage") or {}).values()), {}) if r.get("usage") else {}
        calls.append({
            "dataset": ds, "config": r["config"], "call_idx": r["call_idx"], "llm_model": r["llm_model"],
            "latency_s": r["latency_s"], "advisor_branch": r["advisor_branch"],
            "history_in_prompt": r["history_in_prompt"], "n_similar_workspaces": r["n_similar_workspaces"],
            "context_rules": ";".join("/".join(x) for x in (r.get("context_rules") or [])),
            "recommendations": "|".join(map(str, names)), "n_recommended": len(names),
            "n_exact_valid": sum(1 for n in names if n in sup),
            "llm_error": r.get("llm_error"), "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning"),
        })
    calls = pd.DataFrame(calls)
    calls.to_csv(os.path.join(RESULTS_DIR, "rag_llm_calls.csv"), index=False)

    ev = evaluate_calls(llm, units)
    bf = brute_force_rows(units)
    ev_all = pd.concat([ev, bf], ignore_index=True)
    ev_all.to_csv(os.path.join(RESULTS_DIR, "rag_eval_per_call_seed.csv"), index=False)

    summary = []
    for ds in DATASETS:
        task = DATASET_META[ds]["task"]
        sel, sec, _ = primary_metrics(task)
        bf_ds = bf[bf.dataset == ds].set_index("seed")
        for cfg in ["as_shipped", "full", "no_rag", "no_meta", "no_fuzzy", "no_fuzzy_paired", "full_sameprior",
                    "brute_force"]:
            g = ev_all[(ev_all.dataset == ds) & (ev_all.config == cfg)]
            if g.empty:
                continue
            per_seed = g.groupby("seed").agg({
                f"best_{sel}": "mean", f"best_{sec}": "mean", "best_F1_any": "mean",
                "search_time_s": "mean", "train_time_s": "mean", "n_trained": "mean",
                "runtime_failure": "mean"})
            red = [100 * (1 - per_seed.loc[s, "train_time_s"] / bf_ds.loc[s, "train_time_s"])
                   for s in per_seed.index if s in bf_ds.index]
            red_total = [100 * (1 - per_seed.loc[s, "search_time_s"] / bf_ds.loc[s, "search_time_s"])
                         for s in per_seed.index if s in bf_ds.index]
            rec = {"dataset": ds, "config": cfg, "n_seeds": per_seed.shape[0]}
            for col, key, fmt in [(f"best_{sel}", "sel", "{:.3f}"), (f"best_{sec}", "sec", "{:.3f}"),
                                  ("best_F1_any", "best_f1_any", "{:.3f}"),
                                  ("search_time_s", "search_time_s", "{:.1f}"),
                                  ("train_time_s", "train_time_s", "{:.1f}"),
                                  ("n_trained", "n_trained", "{:.2f}"),
                                  ("runtime_failure", "runtime_failure_rate", "{:.2f}")]:
                rec[key + "_mean"], rec[key + "_std"], rec[key + "_tex"] = ms(per_seed[col], fmt)
            rec["train_time_reduction_pct_mean"], rec["train_time_reduction_pct_std"], _ = ms(red)
            rec["search_time_reduction_pct_mean"], rec["search_time_reduction_pct_std"], _ = ms(red_total)
            rec["metric_names"] = f"{sel}/{sec}"
            if cfg != "brute_force":
                base_cfg = "full" if cfg in ("no_fuzzy_paired",) else cfg
                c = calls[(calls.dataset == ds) & (calls.config == base_cfg)]
                rec["n_llm_calls"] = int(len(c))
                rec["valid_name_rate_raw"] = float(c.n_exact_valid.sum() / max(c.n_recommended.sum(), 1))
                gg = g.drop_duplicates("call_idx")
                if cfg not in ("no_fuzzy", "no_fuzzy_paired"):
                    rec["valid_name_rate_after_matcher"] = float(gg.n_after_matcher.sum() / max(gg.n_recommended.sum(), 1))
                    rec["ui_default_fallback_rate"] = float(gg.ui_default_fallback.mean())
                rec["advisor_fallback_rate"] = float(c.advisor_branch.isin(["exception_fallback", "parse_fallback"]).mean())
                rec["advisor_branches"] = dict(Counter(c.advisor_branch))
                rec["llm_latency_s_mean"] = float(c.latency_s.mean())
                rec["llm_latency_s_std"] = float(c.latency_s.std(ddof=1)) if len(c) > 1 else 0.0
                rec["tokens_per_call_mean"] = float(c.total_tokens.mean()) if c.total_tokens.notna().any() else None
                rec["history_in_prompt_rate"] = float(c.history_in_prompt.mean())
                rec["llm_errors"] = dict(Counter(e.split(":")[0] for e in c.llm_error.dropna()))
                rec["n_distinct_selections"] = int(gg.selected.nunique())
                rec["selection_counts"] = dict(Counter(gg.selected).most_common(5))
                rec["recommended_name_counts"] = dict(Counter(
                    n for s in c.recommendations for n in (s.split("|") if s else [])).most_common(12))
                rec["best_model_counts"] = dict(Counter(g.best_model.dropna()).most_common(5))
            else:
                rec["best_model_counts"] = dict(Counter(g.best_model.dropna()).most_common(5))
            summary.append(rec)
    summary = pd.DataFrame(summary)
    flat = summary.drop(columns=[c for c in summary.columns if isinstance(summary[c].iloc[0], dict)], errors="ignore")
    flat.to_csv(os.path.join(RESULTS_DIR, "rag_summary.csv"), index=False)
    return calls, ev_all, summary


# ─────────────────────────────── ablation B ─────────────────────────────────

def preproc_tables(units):
    serving, parity, schema, endpoint, latency, ser = [], [], [], [], [], []
    for ds, us in units.items():
        for seed, u in us.items():
            p = u.get("preproc")
            if not p:
                continue
            for r in p["serving"]:
                serving.append({"dataset": ds, "seed": seed, **{k: v for k, v in r.items() if k != "errors"},
                                "top_error": next(iter(r.get("errors") or {}), None)})
            par = {k: v for k, v in p["parity"].items() if k != "mismatched_features"}
            par["mismatched_features"] = ";".join(sorted(p["parity"]["mismatched_features"]))
            parity.append({"dataset": ds, "seed": seed, **par,
                           "fit_transform_s": u["preprocessing"]["fit_transform_s"],
                           "fit_rows_per_s": u["preprocessing"]["fit_rows_per_s"]})
            for r in p["schema"]:
                schema.append({"dataset": ds, "seed": seed, **r})
            for r in p["endpoint"]:
                endpoint.append({"dataset": ds, "seed": seed, **r})
            for r in p["endpoint_latency"]:
                latency.append({"dataset": ds, "seed": seed, **{k: v for k, v in r.items() if k != "status_codes"},
                                "status_codes": json.dumps(r["status_codes"])})
            ser.append({"dataset": ds, "seed": seed, **p["serialization"]})
    out = {}
    for name, rows in [("preproc_serving", serving), ("preproc_parity", parity), ("preproc_schema", schema),
                       ("preproc_endpoint", endpoint), ("preproc_endpoint_latency", latency),
                       ("preproc_serialization", ser)]:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RESULTS_DIR, f"{name}.csv"), index=False)
        out[name] = df

    sv = out["preproc_serving"]
    summ = []
    if not sv.empty:
        for (ds, prep, mode, bs, model), g in sv.groupby(["dataset", "preprocessing", "mode", "batch_size", "model"]):
            task = DATASET_META[ds]["task"]
            rec = {"dataset": ds, "preprocessing": prep, "mode": mode, "batch_size": bs, "model": model,
                   "n_seeds": len(g)}
            for col in ["crash_rate", "row_coverage", "agreement_vs_offline", "mean_abs_pred_diff",
                        "agreement_vs_stateful_app", "agreement_vs_stateful_aligned",
                        "rows_served_by_both_app", "rows_served_by_both_aligned",
                        "served_Accuracy", "served_F1", "offline_Accuracy", "offline_F1",
                        "served_RMSE", "served_R2", "offline_RMSE", "offline_R2", "transform_rows_per_s"]:
                if col in g and g[col].notna().any():
                    rec[col + "_mean"], rec[col + "_std"], _ = ms(g[col].astype(float))
            for key in (("RMSE", "R2") if task == "Regression" else ("Accuracy", "F1")):
                if f"served_{key}" in g and g[f"served_{key}"].notna().any():
                    d = (g[f"served_{key}"] - g[f"offline_{key}"]).astype(float)
                    rec[f"{key}_change_mean"], rec[f"{key}_change_std"], _ = ms(d)
            rec["top_error"] = Counter(g.top_error.dropna()).most_common(1)[0][0] if g.top_error.notna().any() else None
            if "stream" in g:
                rec["stream"] = Counter(g["stream"].dropna()).most_common(1)[0][0] if g["stream"].notna().any() else None
            summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RESULTS_DIR, "preproc_summary.csv"), index=False)
    out["summary"] = summ
    return out


# ─────────────────────────────── ablation C ─────────────────────────────────

def shap_tables(units):
    rows = []
    for ds, us in units.items():
        for seed, u in us.items():
            for r in u.get("shap", []):
                rows.append({"dataset": ds, "seed": seed,
                             **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "shap_runs.csv"), index=False)
    summ = []
    if not df.empty:
        for (ds, model, cfg), g in df.groupby(["dataset", "model", "config"]):
            ok = g[g.status == "ok"]
            rec = {"dataset": ds, "model": model, "config": cfg, "n_seeds": len(g),
                   "success_rate": float((g.status == "ok").mean()),
                   "status_counts": json.dumps(dict(Counter(g.status))),
                   "explainer": Counter(g.explainer.dropna()).most_common(1)[0][0] if g.explainer.notna().any() else None}
            rec["runtime_s_mean"], rec["runtime_s_std"], _ = ms(ok.runtime_s) if len(ok) else (None, None, "---")
            rec["peak_mem_mb_mean"], rec["peak_mem_mb_std"], _ = ms(g.peak_mem_mb)
            if "extrapolated_runtime_s" in g and g.extrapolated_runtime_s.notna().any():
                rec["extrapolated_runtime_s_mean"], _, _ = ms(g.extrapolated_runtime_s.astype(float))
            msgs = g.message.dropna()
            rec["message"] = Counter(m.split("|")[0][:160] for m in msgs if m).most_common(1)[0][0] if len(msgs) and any(msgs) else None
            summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RESULTS_DIR, "shap_summary.csv"), index=False)
    per_cfg = []
    if not df.empty:
        for (ds, cfg), g in df.groupby(["dataset", "config"]):
            per_seed = g.groupby("seed").apply(lambda x: (x.status == "ok").sum())
            clean = g.groupby("seed").apply(lambda x: x.status.isin(["ok", "budget_exceeded"]).sum())
            per_cfg.append({"dataset": ds, "config": cfg, "n_models": g.model.nunique(),
                            "models_ok_mean": float(per_seed.mean()), "models_ok_min": int(per_seed.min()),
                            "models_ok_or_budget_mean": float(clean.mean()),
                            "success_rate": float((g.status == "ok").mean()),
                            "budget_exceeded": int((g.status == "budget_exceeded").sum()),
                            "timeouts": int((g.status == "timeout").sum()),
                            "errors": int((g.status == "error").sum()),
                            "nan_values": int((g.status == "nan_values").sum())})
    per_cfg = pd.DataFrame(per_cfg)
    per_cfg.to_csv(os.path.join(RESULTS_DIR, "shap_config_summary.csv"), index=False)
    return df, summ, per_cfg


# ─────────────────────────────── orchestration ──────────────────────────────

def aggregate_all():
    units = load_units()
    llm = load_llm()
    bf, bf_summ = brute_force_tables(units)
    out = {"n_units": {ds: sorted(us) for ds, us in units.items()}, "n_llm_calls": len(llm)}
    rag = rag_tables(units, llm) if llm else None
    pre = preproc_tables(units)
    sh = shap_tables(units)
    headline = build_headline(units, bf_summ, rag, pre, sh)
    out["headline"] = headline
    if rag is not None:
        out["rag_summary"] = rag[2].to_dict(orient="records")
    out["shap_config_summary"] = sh[2].to_dict(orient="records")
    env = next((u["environment"] for us in units.values() for u in us.values()), None)
    out["environment"] = env
    dump_json(out, os.path.join(RESULTS_DIR, "summary.json"))
    from latex_table import write_table

    write_table(units, rag, pre, sh, bf_summ)
    print("aggregate: wrote results/*.csv, summary.json, table_ablation.tex")
    return out


def build_headline(units, bf_summ, rag, pre, sh):
    h = {"paper_claims": PAPER_CLAIMS, "measured": {}, "contradictions": []}
    m = h["measured"]
    for ds in DATASETS:
        task = DATASET_META[ds]["task"]
        sel, sec, higher = primary_metrics(task)
        g = bf_summ[bf_summ.dataset == ds]
        if g.empty or f"{sel}_mean" not in g:
            continue
        best = g.sort_values(f"{sel}_mean", ascending=not higher).iloc[0]
        m[f"{ds}_best_single_model"] = {"model": best.model, sel: best[f"{sel}_mean"], sec: best[f"{sec}_mean"]}
    if rag is not None:
        rs = rag[2]
        for ds in DATASETS:
            r = rs[(rs.dataset == ds) & (rs.config == "full")]
            if len(r):
                r = r.iloc[0]
                m[f"{ds}_full_rag"] = {"sel": r.sel_mean, "sec": r.sec_mean,
                                       "train_time_reduction_pct": r.train_time_reduction_pct_mean,
                                       "search_time_reduction_pct": r.search_time_reduction_pct_mean,
                                       "search_time_s": r.search_time_s_mean,
                                       "valid_name_rate_raw": r.valid_name_rate_raw}
            b = rs[(rs.dataset == ds) & (rs.config == "brute_force")]
            if len(b):
                m[f"{ds}_brute_force_time_s"] = b.iloc[0].search_time_s_mean
    lat = pre.get("preproc_endpoint_latency")
    if lat is not None and not lat.empty:
        for ds, g in lat.groupby("dataset"):
            m[f"{ds}_predict_latency_ms_median"] = float(g.median_ms.median())
            m[f"{ds}_predict_status_codes"] = dict(Counter(k for s in g.status_codes for k, v in json.loads(s).items()
                                                           for _ in range(v)))
    shp = sh[2]
    if not shp.empty:
        for ds, g in shp.groupby("dataset"):
            f = g[g.config == "full"]
            if len(f):
                m[f"{ds}_shap_full_success_rate"] = float(f.success_rate.iloc[0])
    return h


if __name__ == "__main__":
    aggregate_all()
