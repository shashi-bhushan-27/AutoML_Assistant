"""
Before/after table: baseline results (results/) vs the re-run after the fixes (results_after_fixes/).

    python ablation/compare_runs.py [--before results] [--after results_after_fixes] [--out FILE.md]

Reads only the aggregated CSV/JSON files written by ``run_ablation.py aggregate``.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["adult", "credit", "california"]


def load(run):
    d = os.path.join(HERE, run)
    read = lambda f: pd.read_csv(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else pd.DataFrame()  # noqa
    with open(os.path.join(d, "summary.json")) as f:
        summary = json.load(f)
    checks = {}
    if os.path.exists(os.path.join(d, "code_checks.json")):
        with open(os.path.join(d, "code_checks.json")) as f:
            checks = json.load(f)
    return {"summary": summary, "checks": checks, "preproc": read("preproc_summary.csv"),
            "latency": read("preproc_endpoint_latency.csv"), "endpoint": read("preproc_endpoint.csv"),
            "ser": read("preproc_serialization.csv"), "shap": read("shap_config_summary.csv"),
            "rag": read("rag_summary.csv"), "bf": read("brute_force_summary.csv"), "parity": read("preproc_parity.csv")}


def fmt(v, f="{:.3f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f.format(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)


def stateful(run, ds, stat):
    p = run["preproc"]
    if p.empty:
        return None
    g = p[(p.dataset == ds) & (p.preprocessing == "stateful") & (p["mode"] == "app")]
    if g.empty:
        return None
    col = {"agree": "agreement_vs_offline_mean", "crash": "crash_rate_mean"}[stat]
    if col not in g:
        return None
    return float(g[col].min()) if stat == "agree" else float(g[col].max())


def endpoint_codes(run, ds):
    e = run["endpoint"]
    if e.empty or "status_code" not in e:
        return "–"
    g = e[(e.dataset == ds) & (e.variant.isin(["baseline", "reordered_columns"]))]
    return ", ".join(f"{int(k)}×{v}" for k, v in g.status_code.value_counts().sort_index().items())


def latency(run, ds, col="median_ms", model="XGBoost"):
    lat = run["latency"]
    if lat.empty or col not in lat:
        return None
    g = lat[(lat.dataset == ds) & (lat.model == model)]
    return float(g[col].median()) if len(g) else None


def shap_ok(run, ds, clean=False):
    s = run["shap"]
    if s.empty:
        return None
    g = s[(s.dataset == ds) & (s.config == "full")]
    if g.empty:
        return None
    col = "models_ok_or_budget_mean" if clean and "models_ok_or_budget_mean" in g else "models_ok_mean"
    return float(g[col].iloc[0]), int(g.n_models.iloc[0])


def best(run, ds):
    m = run["summary"]["headline"]["measured"].get(f"{ds}_best_single_model", {})
    return m


def rag(run, ds, cfg, col):
    r = run["rag"]
    if r.empty or col not in r:
        return None
    g = r[(r.dataset == ds) & (r.config == cfg)]
    return float(g[col].iloc[0]) if len(g) and pd.notna(g[col].iloc[0]) else None


def matcher(run):
    probes = run["checks"].get("fuzzy_matcher_probes") or []
    if not probes:
        return "–"
    df = pd.DataFrame(probes)
    return (f"{int((df.match_type != 'unmatched').sum())}/{len(df)} names mapped or flagged; "
            f"'RandomForestClassifierRegressor' → "
            f"{df[df.name == 'RandomForestClassifierRegressor'].match_type.iloc[0]}")


def meta(run):
    pairs = run["checks"].get("meta_learning_similarity") or []
    if not pairs:
        return "–"
    df = pd.DataFrame(pairs)
    cross = df[df["query"] != df["prior"]]
    same = df[df["query"] == df["prior"]]
    txt = f"{int(cross.match.sum())}/{len(cross)} cross-dataset pairs match"
    if len(same):
        txt += f"; same-dataset prior matches for {int(same.match.sum())}/{len(same)}"
    return txt


def build(before, after) -> str:
    rows = []
    add = lambda k, b, a: rows.append((k, b, a))  # noqa
    for ds in DATASETS:
        add(f"{ds}: stateful serving, worst agreement with offline (all batch sizes)",
            fmt(stateful(before, ds, "agree")), fmt(stateful(after, ds, "agree")))
        add(f"{ds}: stateful serving, worst crash rate", fmt(stateful(before, ds, "crash")),
            fmt(stateful(after, ds, "crash")))
    for ds in DATASETS:
        add(f"{ds}: /predict status codes (unperturbed + reordered requests)", endpoint_codes(before, ds),
            endpoint_codes(after, ds))
    for ds in DATASETS:
        add(f"{ds}: /predict single-row latency, XGBoost, median ms (TestClient round trip)",
            fmt(latency(before, ds), "{:.1f}"), fmt(latency(after, ds), "{:.1f}"))
        add(f"{ds}: … of which server-side (after only)", "–", fmt(latency(after, ds, "server_median_ms"), "{:.1f}"))
    for ds in DATASETS:
        b, a = before["ser"], after["ser"]
        pb = float(b[b.dataset == ds].pickle_mb.median()) if len(b) else None
        pa = float(a[a.dataset == ds].pickle_mb.median()) if len(a) else None
        add(f"{ds}: pipeline pickle size (MB)", fmt(pb, "{:.2f}"), fmt(pa, "{:.3f}"))
    for ds in DATASETS:
        sb, sa, sc = shap_ok(before, ds), shap_ok(after, ds), shap_ok(after, ds, clean=True)
        add(f"{ds}: SHAP families with values (app routing, of 6)", fmt(sb[0] if sb else None, "{:.1f}"),
            fmt(sa[0] if sa else None, "{:.1f}") + (f" ({sc[0]:.1f} incl. clean budget stop)" if sc else ""))
    for ds in DATASETS:
        mb, ma = best(before, ds), best(after, ds)
        key = "RMSE" if "RMSE" in mb else "Accuracy"
        extra = "R²" if key == "RMSE" else "F1 Score"
        add(f"{ds}: best single model ({key} / {extra})",
            f"{mb.get('model')} {fmt(mb.get(key))} / {fmt(mb.get(extra))}",
            f"{ma.get('model')} {fmt(ma.get(key))} / {fmt(ma.get(extra))}")
    for ds in DATASETS:
        add(f"{ds}: 'as shipped' advisor fallback rate", fmt(rag(before, ds, "as_shipped", "advisor_fallback_rate"), "{:.0%}"),
            fmt(rag(after, ds, "as_shipped", "advisor_fallback_rate"), "{:.0%}"))
        add(f"{ds}: full RAG, training-time reduction vs brute force",
            fmt(rag(before, ds, "full", "train_time_reduction_pct_mean"), "{:.0f}%"),
            fmt(rag(after, ds, "full", "train_time_reduction_pct_mean"), "{:.0f}%"))
        add(f"{ds}: meta-learning history in prompt (full / same-dataset prior)",
            f"{fmt(rag(before, ds, 'full', 'history_in_prompt_rate'), '{:.0%}')} / "
            f"{fmt(rag(before, ds, 'full_sameprior', 'history_in_prompt_rate'), '{:.0%}')}",
            f"{fmt(rag(after, ds, 'full', 'history_in_prompt_rate'), '{:.0%}')} / "
            f"{fmt(rag(after, ds, 'full_sameprior', 'history_in_prompt_rate'), '{:.0%}')}")
    add("Name matcher on the 30 probe names", matcher(before), matcher(after))
    add("Meta-learning rule between the benchmarks", meta(before), meta(after))
    lines = ["| Measure | Before (baseline) | After fixes |", "|---|---|---|"]
    lines += [f"| {k} | {b} | {a} |" for k, b, a in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="results")
    ap.add_argument("--after", default="results_after_fixes")
    ap.add_argument("--out")
    args = ap.parse_args()
    table = build(load(args.before), load(args.after))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
