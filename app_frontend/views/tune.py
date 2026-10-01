"""Step 6 - Tune: Optuna search on the training split; the ranges shown are exactly the ones used."""
import pandas as pd
import streamlit as st

from app_backend.model_tuner import ModelTuner, search_space
from app_backend.task_types import is_classification
from app_frontend.ui import charts
from app_frontend.ui import components as ui
from app_frontend.ui import jobs, nav
from app_frontend.ui import state as S

COMPARE = {"Classification": ["Accuracy", "Balanced Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC"],
           "Regression": ["RMSE", "MAE", "R²"]}


def _range_inputs(model: str, space: dict) -> dict:
    ranges = {}
    st.caption("These are the only parameters the tuner searches for this model; the defaults are shown.")
    for name, spec in space.items():
        kind = spec[0]
        if kind == "cat":
            ranges[name] = tuple(st.multiselect(name, spec[1], default=spec[1], key=f"w_rng_{model}_{name}"))
            continue
        c1, c2 = st.columns(2)
        if kind == "int":
            lo = c1.number_input(f"{name} min", value=int(spec[1]), step=1, key=f"w_rng_{model}_{name}_lo")
            hi = c2.number_input(f"{name} max", value=int(spec[2]), step=1, key=f"w_rng_{model}_{name}_hi")
        else:
            fmt = "%.4g"
            lo = c1.number_input(f"{name} min", value=float(spec[1]), format=fmt, key=f"w_rng_{model}_{name}_lo")
            hi = c2.number_input(f"{name} max", value=float(spec[2]), format=fmt, key=f"w_rng_{model}_{name}_hi")
        ranges[name] = (lo, hi)
    return ranges


def _work(tr, model, budget, folds, ranges):
    def run(job: jobs.Job):
        def progress(n_trials, elapsed, best):
            job.update(min(elapsed / budget, 0.99),
                       f"{n_trials} trials" + (f", best CV score {best:.4f}" if best is not None else ""))
        return ModelTuner(tr).tune_model(model, time_budget=budget, cv_folds=folds, param_ranges=ranges,
                                         progress_cb=progress, cancel_event=job.cancel_event)
    return run


def _apply(job, cfg):
    c, s = S.ctx(), S.state()
    res = job.result or {}
    if job.status != "done" or "Error" in res:
        S.record_step("tune", ok=False, error=(res.get("Error") or job.error or "cancelled").splitlines()[0])
        return
    tr = c["trainer"]
    tuned = f"{cfg['model'].replace(' (Tuned)', '')} (Tuned)"
    c["models"] = dict(tr.trained_models)
    s["predictions"], s["probas"] = dict(tr.predictions), dict(tr.prediction_probas)
    s.setdefault("tune", []).append({"model": cfg["model"], "tuned_name": tuned, "result": res, "config": cfg})
    S.wm().save_trained_models(S.current_ws().workspace_id, tr.trained_models)
    S.record_step("tune", models=[t["tuned_name"] for t in s["tune"]])
    S.save_state()


def _show(entry, pipe, s):
    res, model = entry["result"], entry["model"]
    before = next((r for r in s.get("results", []) if r.get("Model") == model), {})
    ui.tiles([("Trials", f"{res['trials_completed']} / {res['trials']}", "completed / started"),
              ("CV score (best)", f"{res['cv_best_score']:.4f}", f"{res['cv_metric']}, {res['cv_folds']}-fold {res['cv_scheme']}"),
              ("Test split", "same as untuned", "before/after are directly comparable")])
    params = pd.DataFrame([{"parameter": k, "best value": v, "searched range": str(res["param_ranges_used"].get(k))}
                           for k, v in res["Best Params"].items()])
    st.markdown("**Best parameters**")
    st.dataframe(params, width="stretch", hide_index=True)
    rows = [{"metric": m, "before": before.get(m), "after": res.get(m)} for m in COMPARE[pipe.task_type]
            if before.get(m) is not None and res.get(m) is not None]
    if rows:
        comp = pd.DataFrame(rows)
        comp["change"] = (comp["after"] - comp["before"]).round(4)
        if is_classification(pipe.task_type):
            long = comp.rename(columns={"before": "untuned", "after": "tuned"})
            ui.chart(charts.grouped_bars(long, "metric", ["untuned", "tuned"], f"{model}: test-split metrics before "
                                         "and after tuning", fmt=".3f"), comp, key=f"tune_{entry['tuned_name']}",
                     filename="tuning_before_after")
        else:
            st.dataframe(comp, width="stretch", hide_index=True)
            st.caption("RMSE and MAE: lower is better. R²: higher is better.")


def render():
    ws, statuses = nav.require_step("tune")
    c, s = S.ctx(), S.state()
    pipe = c["pipeline"]
    st.subheader("Tune", anchor=False)
    st.caption("Bayesian search (Optuna TPE) with cross-validation on the training split only; the tuned model "
               "is then scored on the same test split as the untuned one.")
    tr = c.get("trainer") or S.rebuild_trainer()
    tunable = [m for m in (tr.trained_models if tr else {}) if search_space(m)]
    problem = S.prerequisite_problem("tune")
    if problem or not tunable:
        st.markdown(problem or "None of the trained models has tunable hyperparameters in this app.")
        return
    tuner = ModelTuner(tr)
    scoring = tuner.scoring()
    job = jobs.get_job("tune")
    running = job is not None and job.status == "running"
    c1, c2, c3 = st.columns([2, 1, 1])
    model = c1.selectbox("Model", tunable, key="w_tune_model")
    budget = c2.slider("Time budget (s)", 10, 600, 60, 10, key="w_tune_budget")
    folds = c3.slider("CV folds", 2, 10, 3, key="w_tune_folds")
    cv_name = "TimeSeriesSplit" if tr.is_time_series else ("KFold" if not is_classification(pipe.task_type)
                                                           else "StratifiedKFold")
    ui.chips([ui.chip(f"CV: {folds}-fold {cv_name}", "info"),
              ui.chip(f"Scoring: {scoring['metric']} ({scoring['reason']})", "neutral", dot=False)])
    with st.expander("Search ranges", expanded=False):
        ranges = _range_inputs(model, search_space(model))
    bad = [k for k, v in ranges.items() if not isinstance(v[0], str) and len(v) == 2 and v[0] > v[1]]
    empty = [k for k, v in ranges.items() if len(v) == 0]
    if bad or empty:
        st.error(f"Fix the ranges: {', '.join(bad + empty)} (min must be ≤ max; pick at least one option).")
    if st.button(f"Tune {model}", type="primary", icon=":material/tune:", disabled=running or bool(bad or empty)):
        cfg = {"model": model, "budget": budget, "folds": folds, "ranges": ranges}
        st.session_state["tune_cfg"] = cfg
        jobs.start_job("tune", f"Tuning {model}", _work(tr, model, budget, folds, ranges), step="tune")
        st.rerun()
    if job is not None:
        if job.status == "running":
            jobs.render_running(job)
        elif not job.applied:
            job.applied = True
            _apply(job, st.session_state.get("tune_cfg", {}))
            if job.status == "done" and "Error" not in (job.result or {}):
                st.rerun()
        if job.status == "error" or "Error" in (job.result or {}):
            ui.banner("critical", f"Tuning failed: {(job.result or {}).get('Error') or job.error.splitlines()[0]}")
    for entry in reversed(s.get("tune") or []):
        st.divider()
        st.markdown(f"#### {entry['tuned_name']}")
        _show(entry, pipe, s)
