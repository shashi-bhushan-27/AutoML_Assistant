"""Step 2 - Prepare: fit the preprocessing pipeline on the training split; parity check; data charts."""
import pandas as pd
import streamlit as st

from app_backend.preprocessing_engine.engine import AutoPreprocessor
from app_backend.task_types import detect_task_type, is_classification
from app_frontend.ui import charts
from app_frontend.ui import components as ui
from app_frontend.ui import jobs, nav
from app_frontend.ui import state as S

FITTED_LABEL = {"train": "training rows", "all rows": "all rows", "all rows (class vocabulary only)":
                "all rows (class labels only)", "all rows (descriptive only)": "all rows (descriptive only)",
                "n/a": "n/a"}


def _work(df: pd.DataFrame, cfg: dict):
    def run(job: jobs.Job):
        job.update(0.1, "Fitting every step on the training rows")
        p = AutoPreprocessor(target_col=cfg["targets"], task_type="auto", is_time_series=cfg["is_time_series"],
                             date_col=cfg["date_col"], test_size=cfg["test_size"], apply_smote=cfg["smote"],
                             random_state=cfg["seed"], cap_outliers=cfg["cap_outliers"], verbose=False)
        out = p.fit_transform(df=df)
        job.update(0.85, "Checking train/serve parity on held-out rows")
        idx = out["X_test"].index[:500]
        raw = df.loc[idx].drop(columns=cfg["targets"])
        parity = p.parity_check(raw, out["X_test"])
        return {"pipeline": p, "out": out, "parity": parity}
    return run


def _apply(job: jobs.Job, cfg: dict):
    ws, c, s = S.current_ws(), S.ctx(), S.state()
    if job.status != "done":
        S.record_step("prepare", ok=False, error=(job.error or "cancelled").splitlines()[0])
        return
    p, out = job.result["pipeline"], job.result["out"]
    c["pipeline"] = p
    c["splits"] = {k: out[k] for k in ("X_train", "X_test", "y_train", "y_test")}
    c["models"] = {}  # models trained on an earlier preparation are incompatible with the new features
    ws.target_col = cfg["targets"][0] if len(cfg["targets"]) == 1 else cfg["targets"]
    ws.task_type = p.task_type
    ws.user_config["seed"] = cfg["seed"]
    ws.preprocessing_steps = p.full_log
    report = p.get_report()
    report.pop("profile", None)
    s.update(prepare_config=cfg, prepare_report=report, parity=job.result["parity"].to_dict("records"),
             predictions={}, probas={}, scores={})
    S.wm().save_preprocessor(ws.workspace_id, p)
    S.wm().save_training_data(ws.workspace_id, out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    S.record_step("prepare", fp=S.fingerprint(ws.dataset_hash, cfg))
    S.save_state()
    S.rebuild_trainer()


def _config_form(df: pd.DataFrame, s: dict, ws, running: bool):
    prev = s.get("prepare_config") or {}
    cols = list(df.columns)
    default_t = [t for t in prev.get("targets", []) if t in cols] or \
        ([s["suggested_target"]] if s.get("suggested_target") in cols else [cols[-1]])
    with st.form("prepare_form"):
        targets = st.multiselect("Target column(s)", cols, default=default_t,
                                 help="One column for standard prediction; several for multi-output prediction.")
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Test split", 0.1, 0.4, float(prev.get("test_size", 0.2)), 0.05)
        seed = c2.number_input("Split seed", 0, 1_000_000, int(prev.get("seed", ws.seed)), step=1,
                               help="Same seed + same data = the same split and the same results.")
        cap = c3.toggle("Cap outliers (IQR)", value=prev.get("cap_outliers", True),
                        help="Q1/Q3 ± 1.5×IQR, learned on training rows. Skipped for columns where it would "
                             "create a constant (e.g. mostly-zero columns).")
        c4, c5 = st.columns(2)
        ts = c4.toggle("Time series (chronological split)", value=prev.get("is_time_series", False))
        date_options = ["(none)"] + cols
        date_col = c4.selectbox("Date column (time series only)", date_options,
                                index=date_options.index(prev["date_col"]) if prev.get("date_col") in cols else 0)
        smote = c5.toggle("SMOTE on the training rows", value=prev.get("smote", False),
                          help="Classification only. Oversamples minority classes in the training rows; the "
                               "test rows are never resampled. The outcome is shown after the run.")
        submitted = st.form_submit_button("Run preprocessing", type="primary", icon=":material/play_arrow:",
                                          disabled=running)
    if not submitted:
        return None
    if not targets:
        st.error("Select at least one target column.")
        return None
    if ts and date_col == "(none)":
        st.error("Choose the date column for a time-series split.")
        return None
    task = detect_task_type(df[targets[0]])
    if smote and (not is_classification(task) or len(targets) > 1):
        st.warning("SMOTE only applies to single-target classification; it will be skipped.")
        smote = False
    return {"targets": targets, "test_size": test_size, "seed": int(seed), "cap_outliers": cap,
            "is_time_series": ts, "date_col": None if date_col == "(none)" else date_col, "smote": smote}


def _results(p, s):
    rep = s.get("prepare_report") or p.get_report()
    summ = rep.get("summary", {})
    ui.tiles([("Task", summ.get("task_type", "-"), None),
              ("Features", f"{len(p.input_columns_)} → {summ.get('features_final')}", "raw columns → model features"),
              ("Train rows", f"{summ.get('train_samples', 0):,}", "after SMOTE" if rep.get("smote", {}).get("status") == "applied" else None),
              ("Test rows", f"{summ.get('test_samples', 0):,}", f"{summ.get('split_strategy')} split, seed {summ.get('split_seed')}"),
              ("Duplicates removed", f"{summ.get('duplicates_removed', 0):,}", "before the split, all rows"),
              ("Rows without target", f"{summ.get('rows_dropped_missing_target', 0):,}", "dropped")])

    parity = pd.DataFrame(s.get("parity") or [])
    if not parity.empty:
        n_ok, n = int(parity["passed"].sum()), len(parity)
        if n_ok == n:
            ui.banner("good", f"Train/serve parity: PASS. transform() on held-out raw rows reproduced the training "
                              f"representation for all {n} features (tolerance 1e-9).", title="✓ Parity")
        else:
            bad = ", ".join(parity.loc[~parity["passed"], "feature"].head(8))
            ui.banner("critical", f"Train/serve parity: FAIL for {n - n_ok} of {n} features ({bad}).", title="✕ Parity")
        with st.expander("Parity check per feature"):
            st.dataframe(parity, width="stretch", hide_index=True)
            if st.button("Re-run parity check", key="rerun_parity"):
                raw = S.raw_holdout(500)
                s["parity"] = p.parity_check(raw, S.ctx()["splits"]["X_test"]).to_dict("records")
                S.save_state()
                st.rerun()

    skipped = rep.get("capping_skipped") or {}
    if skipped:
        ui.banner("info", "Outlier capping was skipped (column kept unchanged) for: "
                  + "; ".join(f"{c} ({why})" for c, why in skipped.items()), title="i Not capped")
    dropped = rep.get("dropped_features", {})
    msgs = []
    if dropped.get("constant_in_train"):
        msgs.append(f"constant in the training rows: {', '.join(dropped['constant_in_train'])}")
    if dropped.get("high_correlation"):
        msgs.append("|r| > 0.95 with a kept column: " + ", ".join(f"{a} (~{b})" for a, b in dropped["high_correlation"].items()))
    if dropped.get("mostly_missing"):
        msgs.append(f"> 50% missing: {', '.join(dropped['mostly_missing'])}")
    if dropped.get("empty"):
        msgs.append(f"empty: {', '.join(dropped['empty'])}")
    if msgs:
        ui.banner("warning", "Dropped features - " + "; ".join(msgs), title="! Dropped")
    smote = rep.get("smote") or {}
    if smote.get("status") not in (None, "not requested"):
        kind = {"applied": "good", "skipped": "warning", "failed": "critical"}[smote["status"]]
        icon = {"applied": "✓", "skipped": "!", "failed": "✕"}[smote["status"]]
        ui.banner(kind, f"{smote.get('reason')}. Training rows {smote.get('rows_before', 0):,} → "
                        f"{smote.get('rows_after', 0):,}; class counts {smote.get('class_counts_after')}.",
                  title=f"{icon} SMOTE {smote['status']}")
    imb = rep.get("imbalance") or {}
    if imb.get("is_imbalanced"):
        ui.banner("warning", f"Imbalanced target: the minority class is {imb['minority_share']:.2%} of the training "
                             "rows. The leaderboard will rank models by F1 of the minority class and show the "
                             "majority-class baseline.", title="! Imbalance")

    st.markdown("#### Steps")
    steps = pd.DataFrame([{"Step": e["step"], "Status": e.get("status"),
                           "Fitted on": FITTED_LABEL.get(e.get("fitted_on", "n/a"), e.get("fitted_on")),
                           "Action": e["action"], "Reason": e["reason"]}
                          for e in rep.get("steps", [])])
    if steps.empty:
        st.caption("No step log was saved for this preparation; re-run Prepare to see it.")
        return
    show_skipped = st.toggle("Show skipped steps", value=False, key="w_show_skipped")
    view = steps if show_skipped else steps[steps["Status"] == "applied"]
    st.dataframe(view, width="stretch", hide_index=True,
                 column_config={"Reason": st.column_config.TextColumn(width="large")})
    ui.download_df(steps, "preprocessing_steps.csv", key="dl_steps")
    st.caption("Split first, then every statistic (caps, skew transforms, fill values, encodings, selected "
               "features, scaler) is learned from the training rows only. Rows marked 'all rows' are structural "
               "(deduplication, empty columns) or descriptive (profile, the list of class labels).")


def _charts(df: pd.DataFrame, p):
    st.markdown("#### Data overview")
    target = p.target_col
    if is_classification(p.task_type) and not p.is_multi_output:
        counts = df[target].astype(str).value_counts().rename_axis("class").reset_index(name="rows")
        counts["share"] = (counts["rows"] / counts["rows"].sum()).round(4)
        fig = charts.bar(counts, "class", "rows", f"Class balance of '{target}'", fmt=",d", value_title="rows")
        ui.chart(fig, counts, key="balance", filename="class_balance")
    num = [c for c in df.select_dtypes(include="number").columns if c not in p.target_cols][:20]
    if len(num) >= 2:
        corr = df[num].corr().round(3)
        ui.chart(charts.correlation_heatmap(corr), corr.reset_index().rename(columns={"index": "column"}),
                 key="corr", filename="correlation", caption="Computed on all rows for display only.")


def render():
    ws, statuses = nav.require_step("prepare")
    c, s = S.ctx(), S.state()
    df = c.get("df")
    st.subheader("Prepare", anchor=False)
    st.caption("Choose the target and split, then fit the preprocessing pipeline. The same fitted pipeline is "
               "used for training, the /predict API and the exported training script.")
    job = jobs.get_job("prepare")
    running = job is not None and job.status == "running"
    cfg = _config_form(df, s, ws, running)
    if cfg is not None:
        st.session_state["prepare_cfg"] = cfg
        jobs.start_job("prepare", "Preprocessing", _work(df, cfg), step="prepare")
        st.rerun()
    if job is not None:
        if job.status == "running":
            jobs.render_running(job)
        elif not job.applied:
            job.applied = True
            _apply(job, st.session_state.get("prepare_cfg", {}))
            if job.status == "done":
                st.toast("Preprocessing complete")
                st.rerun()  # refresh the stepper
        if job.status == "error":
            ui.banner("critical", f"Preprocessing failed: {job.error.splitlines()[0]}")
            with st.expander("Error details"):
                st.code(job.error)
        elif job.status == "cancelled":
            ui.banner("warning", "Preprocessing was cancelled; the previous pipeline (if any) is unchanged.")
    p = c.get("pipeline")
    if p is None:
        if (ws.steps.get("prepare") or {}).get("state") == "error" and job is None:
            ui.banner("critical", f"The last run failed: {ws.steps['prepare'].get('error')}")
        return
    _results(p, s)
    _charts(df, p)
    st.divider()
    b1, b2 = st.columns(2)
    with b1:
        nav.link("recommend", "Next: Get model recommendations", ":material/arrow_forward:")
    with b2:
        nav.link("train", "Skip to Train & Compare", ":material/fast_forward:")
