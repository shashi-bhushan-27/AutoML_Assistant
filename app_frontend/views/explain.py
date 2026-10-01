"""Step 5 - Explain: SHAP values in a background job with a time budget, progress and cancel."""
import pandas as pd
import streamlit as st

from app_backend import model_registry
from app_backend.shap_explainer import SHAPExplainer
from app_backend.task_types import is_classification
from app_frontend.ui import charts
from app_frontend.ui import components as ui
from app_frontend.ui import jobs, nav
from app_frontend.ui import state as S

STATUS_KIND = {"ok": "good", "budget_exceeded": "warning", "cancelled": "warning", "error": "critical"}
STATUS_TEXT = {"ok": "✓ Complete", "budget_exceeded": "! Time budget reached", "cancelled": "! Cancelled",
               "error": "✕ Failed"}


def _explainer(tr, model_name, class_index, pipe):
    classes = pipe.classes_
    return SHAPExplainer(tr.trained_models[model_name], tr.X_train, tr.X_test, task_type=pipe.task_type,
                         class_index=class_index,
                         class_names=[str(c) for c in pipe.decode_target(getattr(tr.trained_models[model_name],
                                                                                 "classes_", []))]
                         if classes is not None else None)


def _work(ex: SHAPExplainer, rows: int, budget: float):
    def run(job: jobs.Job):
        def progress(done, total):
            job.update(done / max(total, 1), f"{done}/{total} rows explained")
        res = ex.explain(max_rows=rows, time_budget_s=budget, progress_cb=progress, cancel_event=job.cancel_event)
        return {"explainer": ex, "result": res}
    return run


def _store(model_name, ex: SHAPExplainer, cfg):
    s = S.state()
    s.setdefault("shap", {})[model_name] = {
        "result": ex.last_result.to_dict(), "values": ex._shap_values, "expected": ex._expected_value,
        "X": ex.X_explained, "features": ex.feature_names, "config": cfg}
    S.record_step("explain", models=sorted(s["shap"]))
    S.save_state()


def _restore(entry) -> SHAPExplainer:
    """Rebuild a lightweight explainer object from stored values (for the charts)."""
    ex = SHAPExplainer.__new__(SHAPExplainer)
    ex.feature_names = entry["features"]
    ex._shap_values, ex._expected_value, ex.X_explained = entry["values"], entry["expected"], entry["X"]
    return ex


def _charts(model_name, entry):
    res = entry["result"]
    ex = _restore(entry)
    kind = STATUS_KIND.get(res["status"], "info")
    msg = (f"{res['rows_done']} of {res['rows_requested']} rows explained in {res['runtime_s']:.1f} s with "
           f"{res['explainer']}. Output explained: {res['output_explained']}.")
    if res["status"] == "budget_exceeded":
        msg += " The time budget stopped the run; the charts use the rows finished so far."
    ui.banner(kind, msg, title=STATUS_TEXT.get(res["status"], res["status"]))
    for w in res.get("warnings", []):
        ui.banner("warning", w)
    if ex._shap_values is None:
        if res.get("error"):
            ui.banner("critical", res["error"])
        return
    imp = ex.get_feature_importance_df(top_n=15)
    ui.chart(charts.bar(imp, "Feature", "SHAP Importance", f"Mean |SHAP value| - {model_name}", fmt=".4f",
                        value_title="mean |SHAP value|"), imp, key="shap_imp", filename=f"shap_importance_{model_name}",
             caption="Average absolute contribution of each feature over the explained rows.")
    bee = ex.get_beeswarm_data(top_n=12)
    ui.chart(charts.beeswarm(bee, f"SHAP values per row - {model_name}"), bee, key="shap_bee",
             filename=f"shap_beeswarm_{model_name}",
             caption="Each dot is one explained row. Right of zero pushes the explained output up; colour shows "
                     "whether that row's feature value is low (blue) or high (red) within the explained rows.")
    c1, c2 = st.columns(2)
    with c1:
        row = st.number_input("Row to break down", 0, max(0, len(ex.X_explained) - 1), 0, key="w_wf_row")
        wf = ex.get_waterfall_data(int(row))
        wf_df = pd.DataFrame({"feature": wf["features"], "contribution": wf["shap_contributions"]})
        ui.chart(charts.waterfall(wf, f"Row {row}: base value → prediction"), wf_df, key="shap_wf",
                 filename=f"shap_waterfall_{model_name}_row{row}",
                 caption="Red bars (+) raise the output, blue bars (−) lower it; the last bar is the model output.")
    with c2:
        feat = st.selectbox("Feature for the dependence plot", imp["Feature"].tolist(), key="w_dep_feat")
        dep = ex.get_dependence_data(feat)
        ui.chart(charts.scatter(dep, "Feature Value", "SHAP Value", f"Dependence - {feat}"), dep, key="shap_dep",
                 filename=f"shap_dependence_{model_name}_{feat}",
                 caption="Feature values are the model inputs (after preprocessing and scaling).")


def render():
    ws, statuses = nav.require_step("explain")
    c, s = S.ctx(), S.state()
    pipe = c["pipeline"]
    st.subheader("Explain", anchor=False)
    st.caption("SHAP values show how much each feature moved a model's output for individual test rows.")
    tr = c.get("trainer") or S.rebuild_trainer()
    names = [m for m in (tr.trained_models if tr else {}) if m not in model_registry.TIME_SERIES_MODELS + ["LSTM"]]
    problem = S.prerequisite_problem("explain")
    if problem or not names:
        st.markdown(problem or "Train at least one model first.")
        return
    if pipe.is_multi_output:
        ui.banner("info", "SHAP explanations are not available for multi-output models.")
        return
    job = jobs.get_job("explain")
    running = job is not None and job.status == "running"
    c1, c2, c3 = st.columns([2, 1, 1])
    model_name = c1.selectbox("Model", names, key="w_shap_model")
    rows = c2.slider("Rows to explain", 10, min(500, len(tr.X_test)), min(100, len(tr.X_test)), 10, key="w_shap_rows")
    budget = c3.slider("Time budget (s)", 30, 300, 120, 10, key="w_shap_budget")
    class_index = None
    if is_classification(pipe.task_type):
        classes = list(getattr(tr.trained_models[model_name], "classes_", []))
        if len(classes) > 2:
            names_c = [str(x) for x in pipe.decode_target(classes)]
            class_index = names_c.index(st.selectbox("Class to explain", names_c, key="w_shap_class"))
    ex = _explainer(tr, model_name, class_index, pipe)
    route = ex._get_model_type()
    ui.chips([ui.chip({"tree": "TreeExplainer", "linear": "LinearExplainer", "kernel": "KernelExplainer"}[route],
                      "info"), ui.chip(ex.route_reason(), "neutral", dot=False)])
    est_key = f"est_{model_name}_{class_index}"
    if est_key not in st.session_state:
        with st.spinner("Estimating runtime..."):
            st.session_state[est_key] = ex.estimate_seconds_per_row()
    per_row = st.session_state[est_key]
    if per_row is not None:
        est = per_row * rows
        text = f"Estimated time: about {est:.0f} s for {rows} rows ({per_row:.2f} s per row)."
        if est > budget:
            text += f" This exceeds the {budget} s budget: about {int(budget / per_row) if per_row else 0} rows " \
                    "will be explained before the budget stops the run."
            ui.banner("warning", text, title="! Estimate")
        else:
            st.caption(text)
    if st.button("Compute SHAP values", type="primary", icon=":material/play_arrow:", disabled=running):
        st.session_state["shap_cfg"] = {"model": model_name, "rows": rows, "budget": budget, "class_index": class_index}
        jobs.start_job("explain", f"SHAP for {model_name}", _work(ex, rows, budget), step="explain")
        st.rerun()
    if job is not None:
        if job.status == "running":
            jobs.render_running(job)
        elif not job.applied:
            job.applied = True
            if job.status == "error":
                S.record_step("explain", ok=False, error=job.error.splitlines()[0])
            else:
                cfg = st.session_state.get("shap_cfg", {})
                _store(cfg.get("model", model_name), job.result["explainer"], cfg)
                st.rerun()
        if job.status == "error":
            ui.banner("critical", f"SHAP failed: {job.error.splitlines()[0]}")
    entry = (s.get("shap") or {}).get(model_name)
    if entry:
        st.divider()
        _charts(model_name, entry)
    elif s.get("shap"):
        st.caption(f"Explained so far: {', '.join(sorted(s['shap']))}.")
