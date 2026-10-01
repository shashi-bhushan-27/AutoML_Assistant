"""Step 4 - Train & Compare: train the selected models in a background job; leaderboard; diagnostics."""
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from app_backend import model_registry
from app_backend.leaderboard import choose_primary_metric, majority_baseline, minority_share, select_best_model, successful
from app_backend.model_matcher import DEFAULTS
from app_backend.model_trainer import ModelTrainer
from app_backend.task_types import is_classification, normalize_task_type
from app_frontend.ui import charts
from app_frontend.ui import components as ui
from app_frontend.ui import jobs, nav
from app_frontend.ui import state as S

HIDDEN = ("Per-Class Report", "Per-Target Metrics", "Error", "AUC note")


def _work(splits, pipe, cfg, df, seed):
    def run(job: jobs.Job):
        tr = ModelTrainer(df, pipe.target_cols, pipe.task_type, cfg["is_time_series"], cfg["date_col"],
                          random_state=seed, class_weight="balanced" if cfg["class_weight"] else None)
        tr.set_preprocessed_data(splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"])

        def progress(i, n, name):
            job.update(i / max(n, 1), f"{i}/{n} models done" + (f", training {name}" if name != "done" else ""))

        res = tr.run_selected_models(cfg["models"], progress_cb=progress, cancel_event=job.cancel_event,
                                     add_ensemble=cfg["ensemble"])
        return {"trainer": tr, "results": res}
    return run


def _apply(job, cfg):
    ws, c, s = S.current_ws(), S.ctx(), S.state()
    if job.status != "done":
        S.record_step("train", ok=False, error=(job.error or "cancelled").splitlines()[0])
        return
    tr, res = job.result["trainer"], job.result["results"]
    pipe = c["pipeline"]
    minority = minority_share(c["splits"]["y_train"])
    metric = choose_primary_metric(pipe.task_type, minority)
    best = select_best_model(res, pipe.task_type, minority)
    c["trainer"], c["models"] = tr, dict(tr.trained_models)
    records = [{k: v for k, v in r.items()} for r in res.to_dict("records")]
    s.update(results=records, failed=list(tr.failed_models), metric=metric, train_config=cfg,
             predictions=dict(tr.predictions), probas=dict(tr.prediction_probas), scores=dict(tr.prediction_scores),
             baseline=majority_baseline(c["splits"]["y_train"], c["splits"]["y_test"], tr.pos_label)
             if is_classification(pipe.task_type) and not pipe.is_multi_output else None,
             shap={}, tune=[])
    ws.model_results = {"leaderboard": [{k: v for k, v in r.items() if k not in HIDDEN} for r in records]}
    ws.best_model, ws.best_score, ws.best_metric = (best["model"], best["score"], best["metric"]) if best else (None, None, None)
    ws.status = "completed" if best else ws.status
    skipped = S.wm().save_trained_models(ws.workspace_id, tr.trained_models)
    s["not_saved"] = skipped
    S.record_step("train", fp=S.fingerprint((ws.steps.get("prepare") or {}).get("fp"), cfg, job.finished),
                  models=len(tr.trained_models), failed=len(tr.failed_models))
    S.save_state()


def _form(supported, default, running, problem, task, n_rows):
    with st.form("train_form"):
        models = st.multiselect("Models to train", supported, default=[m for m in default if m in supported],
                                help="Pre-selected from the Recommend step when it ran; otherwise the task default.")
        c1, c2 = st.columns(2)
        cw = c1.toggle("Balanced class weights", value=False, disabled=not is_classification(task),
                       help="Classification only; passed as sample weights to models that accept them (not KNN).")
        ens = c2.toggle("Add a voting ensemble of the trained models", value=False,
                        help="Plain average of the trained models' predictions/probabilities. Members are not "
                             "re-fitted and no weights are learned.")
        slow = [m for m in models if (m in ("SVM", "KNN") and n_rows > 20_000) or (m == "Gradient Boosting" and n_rows > 100_000)]
        submitted = st.form_submit_button("Train", type="primary", icon=":material/model_training:",
                                          disabled=running or bool(problem))
    if slow:
        st.caption(f"Note: {', '.join(slow)} can take several minutes on {n_rows:,} training rows.")
    if submitted and not models:
        st.error("Select at least one model.")
        return None
    return {"models": models, "class_weight": cw, "ensemble": ens} if submitted else None


def _leaderboard(s, pipe):
    res = pd.DataFrame(s.get("results") or [])
    metric = s.get("metric") or choose_primary_metric(pipe.task_type)
    ok = successful(res)
    col, higher = metric["metric"], metric["higher_is_better"]
    ui.chips([ui.chip(f"Ranked by {col} ({'higher' if higher else 'lower'} is better)", "info"),
              ui.chip(metric["reason"], "neutral", dot=False)])
    if ok.empty or col not in ok:
        ui.banner("critical", "No model trained successfully.")
    else:
        ok = ok.sort_values(col, ascending=not higher).reset_index(drop=True)
        best = ok.iloc[0]
        tiles = [("Best model", str(best["Model"]), f"{col} {best[col]:.4f}")]
        base = s.get("baseline")
        if base:
            label = pipe.decode_target([base["majority_class"]])[0]
            tiles.append(("Majority-class baseline", f"{base.get(col, base['Accuracy']):.4f}",
                          f"{col} of always predicting '{label}'"))
        tiles.append(("Models trained", str(len(ok)), f"{len(s.get('failed') or [])} failed"))
        ui.tiles(tiles)
        shown = ok[[c for c in ok.columns if c not in HIDDEN]]
        if "Positive class" in shown:
            shown = shown.assign(**{"Positive class": pipe.decode_target(shown["Positive class"].astype(int))})
        st.dataframe(shown, width="stretch", hide_index=True,
                     column_config={col: st.column_config.NumberColumn(format="%.4f"),
                                    "Notes": st.column_config.TextColumn(width="large")})
        ui.download_df(shown, "leaderboard.csv", key="dl_lb")
        ref = (base.get(col), "majority baseline") if base and base.get(col) is not None else None
        fig = charts.bar(ok, "Model", col, f"{col} on the test split", ref=ref, fmt=".4f")
        ui.chart(fig, ok[["Model", col]], key="lb", filename="leaderboard")
        if "Time (s)" in ok:
            t = ok[["Model", "Time (s)"]].sort_values("Time (s)", ascending=False)
            ui.chart(charts.bar(t, "Model", "Time (s)", "Training time (seconds, fit only)", fmt=".2f"), t,
                     key="time", filename="training_time")
    failed = s.get("failed") or []
    if failed:
        ui.banner("critical", f"{len(failed)} model(s) failed; they are not in the leaderboard.", title="✕ Failed")
        st.dataframe(pd.DataFrame(failed), width="stretch", hide_index=True,
                     column_config={"Error": st.column_config.TextColumn(width="large")})
    if s.get("not_saved"):
        ui.banner("warning", f"Not saved to disk (not serialisable): {', '.join(s['not_saved'])}.")
    return ok


def _diagnostics(tr, pipe, ok):
    from sklearn.metrics import auc, confusion_matrix, roc_curve

    names = [m for m in ok["Model"] if m in tr.predictions] if not ok.empty else []
    if not names:
        return
    st.markdown("#### Diagnostics (test split)")
    y_true = np.asarray(tr.y_test)
    if is_classification(pipe.task_type) and not pipe.is_multi_output:
        model = st.selectbox("Model", names, key="w_diag_model")
        labels = sorted(np.unique(np.concatenate([y_true, np.asarray(tr.predictions[model])])))
        names_l = [str(x) for x in pipe.decode_target(labels)]
        cm = confusion_matrix(y_true, tr.predictions[model], labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"actual {n}" for n in names_l], columns=[f"pred {n}" for n in names_l])
        ui.chart(charts.confusion_matrix(cm, names_l, f"Confusion matrix - {model}"),
                 cm_df.reset_index().rename(columns={"index": ""}), key="cm", filename=f"confusion_{model}")
        if len(labels) == 2:
            pos = tr.pos_label
            curves, rows = {}, []
            for m in names[:8]:
                score = None
                if m in tr.prediction_probas:
                    cls = list(getattr(tr.trained_models.get(m), "classes_", labels))
                    score = tr.prediction_probas[m][:, cls.index(pos)]
                elif m in tr.prediction_scores and np.ndim(tr.prediction_scores[m]) == 1:
                    cls = list(getattr(tr.trained_models.get(m), "classes_", labels))
                    score = tr.prediction_scores[m] * (1 if cls[1] == pos else -1)
                if score is None:
                    continue
                fpr, tpr, _ = roc_curve(y_true == pos, score)
                curves[m] = (fpr, tpr, auc(fpr, tpr))
                rows.append({"Model": m, "AUC": round(auc(fpr, tpr), 4)})
            if curves:
                pos_name = pipe.decode_target([pos])[0]
                ui.chart(charts.roc_curves(curves, f"ROC curves (positive class '{pos_name}')"), pd.DataFrame(rows),
                         key="roc", filename="roc_curves",
                         caption="Models are told apart by colour, line style and marker shape; the table lists AUC.")
        report = next((r.get("Per-Class Report") for r in S.state().get("results", []) if r.get("Model") == model), None)
        if isinstance(report, dict):
            rep = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
            rep["class"] = [str(pipe.decode_target([int(float(x))])[0]) if str(x).replace(".", "").isdigit() else x
                            for x in rep["class"]]
            with st.expander(f"Per-class report - {model}"):
                st.dataframe(rep.round(4), width="stretch", hide_index=True)
    elif not is_classification(pipe.task_type) and not pipe.is_multi_output:
        model = st.selectbox("Model", names, key="w_diag_model")
        y_pred = np.asarray(tr.predictions[model], dtype=float)
        df = pd.DataFrame({"actual": y_true.astype(float), "predicted": y_pred, "residual": y_true - y_pred})
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(charts.pred_vs_actual(df["actual"], df["predicted"], f"Predicted vs actual - {model}"), df,
                     key="pva", filename=f"pred_vs_actual_{model}")
        with c2:
            ui.chart(charts.residual_hist(df["residual"], "Residuals"), df[["residual"]].describe().reset_index(),
                     key="resid", filename=f"residuals_{model}")
    tree_models = [m for m in names if hasattr(tr.trained_models.get(m), "feature_importances_")]
    if tree_models:
        m = st.selectbox("Built-in feature importance", tree_models, key="w_fi_model")
        imp = pd.DataFrame({"Feature": list(tr.X_train.columns), "Importance": tr.trained_models[m].feature_importances_})
        imp = imp.sort_values("Importance", ascending=False).head(15)
        ui.chart(charts.bar(imp, "Feature", "Importance", f"Top 15 features by built-in importance - {m}", fmt=".3f"),
                 imp, key="fi", filename=f"feature_importance_{m}",
                 caption="Impurity/gain importance from the model itself; see Explain for SHAP values.")


def _downloads(tr, ws):
    st.markdown("#### Download")
    models = [m for m in tr.trained_models if m not in model_registry.TIME_SERIES_MODELS]
    if not models:
        return
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    m = c1.selectbox("Model file", models, key="w_dl_model")
    try:
        c2.download_button("Model (.pkl)", pickle.dumps(tr.trained_models[m]), f"{m.replace(' ', '_')}_model.pkl",
                           mime="application/octet-stream", icon=":material/download:", width="stretch")
    except (pickle.PicklingError, TypeError) as exc:
        c2.caption(f"{m} cannot be pickled: {exc}")
    c3.download_button("Pipeline (.pkl)", pickle.dumps(S.ctx()["pipeline"]), "preprocessor_pipeline.pkl",
                       mime="application/octet-stream", icon=":material/download:", width="stretch")
    st.caption("Pickles run code when loaded: only load files you created. Serve with pipeline.transform(rows) "
               "before model.predict - or use the Deploy step.")


def render():
    ws, statuses = nav.require_step("train")
    c, s = S.ctx(), S.state()
    pipe = c["pipeline"]
    st.subheader("Train & Compare", anchor=False)
    tr = c.get("trainer") or S.rebuild_trainer()
    if tr is None:
        ui.banner("critical", "The prepared data could not be loaded; re-run Prepare.")
        return
    problem = S.prerequisite_problem("train")
    if problem:
        st.markdown(problem)
    rec_fresh = statuses["recommend"]["state"] == "done"
    default = s.get("selection") if rec_fresh and s.get("selection") else \
        DEFAULTS[normalize_task_type(pipe.task_type)]
    if not rec_fresh:
        st.caption("No current recommendation: the task's default models are pre-selected.")
    job = jobs.get_job("train")
    running = job is not None and job.status == "running"
    cfg = _form(tr.get_supported_models(), default, running, problem, pipe.task_type, len(c["splits"]["X_train"]))
    if cfg is not None:
        cfg.update(is_time_series=s.get("prepare_config", {}).get("is_time_series", False),
                   date_col=s.get("prepare_config", {}).get("date_col"))
        st.session_state["train_cfg"] = cfg
        jobs.start_job("train", f"Training {len(cfg['models'])} model(s)",
                       _work(c["splits"], pipe, cfg, c["df"], ws.seed), step="train")
        st.rerun()
    if job is not None:
        if job.status == "running":
            jobs.render_running(job)
        elif not job.applied:
            job.applied = True
            _apply(job, st.session_state.get("train_cfg", {}))
            if job.status == "done":
                st.rerun()
        if job.status == "error":
            ui.banner("critical", f"Training failed: {job.error.splitlines()[0]}")
            with st.expander("Error details"):
                st.code(job.error)
        elif job.status == "cancelled":
            ui.banner("warning", "Training was cancelled.")
    if not s.get("results"):
        return
    st.divider()
    ok = _leaderboard(s, pipe)
    if statuses["train"]["state"] == "stale" or not tr.trained_models:
        st.caption("Diagnostics and downloads need models trained on the current preparation.")
        return
    _diagnostics(tr, pipe, ok)
    _downloads(tr, ws)
    st.divider()
    b1, b2, b3 = st.columns(3)
    with b1:
        nav.link("explain", "Explain a model", ":material/psychology:")
    with b2:
        nav.link("tune", "Tune a model", ":material/tune:")
    with b3:
        nav.link("deploy", "Deploy", ":material/rocket_launch:")
