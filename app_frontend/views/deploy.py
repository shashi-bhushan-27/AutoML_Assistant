"""Step 7 - Deploy: readiness check against the real /predict endpoint, request schema, examples, exports."""
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from app_backend import model_registry
from app_backend.code_generator import generate_training_code
from app_backend.leaderboard import successful
from app_backend.report_generator import ReportGenerator
from app_backend.task_types import is_classification
from app_frontend.ui import components as ui
from app_frontend.ui import nav
from app_frontend.ui import state as S


def _rows_payload(df: pd.DataFrame):
    return json.loads(df.to_json(orient="records", date_format="iso"))


def readiness_check(ws, model_name: str, n: int = 5) -> dict:
    """POST 5 held-out raw rows to /predict (in-process FastAPI app, and the running server if reachable)
    and compare the answers with the offline predictions of the same rows."""
    from fastapi.testclient import TestClient

    import app_backend.main_api as api

    c = S.ctx()
    pipe, tr = c["pipeline"], c["trainer"]
    rows = S.raw_holdout(n)
    payload = {"workspace_id": ws.workspace_id, "model_name": model_name, "data": _rows_payload(rows)}
    X = c["splits"]["X_test"].loc[rows.index]
    offline = tr.trained_models[model_name].predict(X)
    offline = pipe.decode_target(offline) if is_classification(pipe.task_type) and not pipe.is_multi_output else offline
    offline = [v.item() if isinstance(v, np.generic) else v for v in np.asarray(offline, dtype=object)]
    out = {"model": model_name, "rows": len(rows), "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "targets": []}

    def compare(preds):
        if preds is None:
            return None
        if is_classification(pipe.task_type):
            return [str(a) == str(b) for a, b in zip(preds, offline)]
        return [abs(float(a) - float(b)) <= 1e-6 * (1 + abs(float(b))) for a, b in zip(preds, offline)]

    client = TestClient(api.app)
    client.post("/predict", json=payload)  # warm the artefact cache
    t0 = time.perf_counter()
    r = client.post("/predict", json=payload)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}
    out["targets"].append({"target": "in-process FastAPI app", "status": r.status_code,
                           "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                           "server_ms": (body.get("timing_ms") or {}).get("total"),
                           "matches": compare(body.get("predictions")), "warnings": body.get("warnings", []),
                           "detail": body.get("detail")})
    try:
        import httpx

        t0 = time.perf_counter()
        cold = httpx.post(f"{nav.API_BASE_URL}/predict", json=payload, timeout=60.0)  # loads the artefacts
        cold_ms = round((time.perf_counter() - t0) * 1000, 1)
        t0 = time.perf_counter()
        r2 = httpx.post(f"{nav.API_BASE_URL}/predict", json=payload, timeout=10.0)
        b2 = r2.json() if r2.headers.get("content-type", "").startswith("application/json") else {"detail": r2.text}
        out["targets"].append({"target": nav.API_BASE_URL, "status": r2.status_code,
                               "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                               "server_ms": (b2.get("timing_ms") or {}).get("total"),
                               "matches": compare(b2.get("predictions")), "warnings": b2.get("warnings", []),
                               "detail": b2.get("detail"),
                               "note": f"first (cold) request {cold_ms} ms, HTTP {cold.status_code}"})
    except (httpx.HTTPError, OSError, ValueError) as exc:
        out["targets"].append({"target": nav.API_BASE_URL, "status": None, "latency_ms": None, "server_ms": None,
                               "matches": None, "warnings": [], "detail": f"not reachable ({type(exc).__name__})"})
    out["offline"] = offline
    return out


def _show_check(chk):
    for t in chk["targets"]:
        if t["status"] is None:
            ui.banner("info", f"{t['target']}: {t['detail']}. Start the API with "
                              "`uvicorn app_backend.main_api:app --port 8000` to check it too.", title="i API server")
            continue
        ok = t["status"] == 200 and t["matches"] is not None and all(t["matches"])
        kind = "good" if ok else "critical"
        agree = f"{sum(t['matches'])}/{len(t['matches'])} predictions equal the offline ones" if t["matches"] is not None else "no predictions"
        server = f", {t['server_ms']:.1f} ms server-side" if t.get("server_ms") is not None else ""
        note = f" ({t['note']})" if t.get("note") else ""
        ui.banner(kind, f"{t['target']}: HTTP {t['status']}, {t['latency_ms']} ms warm round trip{server}{note}; "
                        f"{agree}." + (f" Detail: {t['detail']}" if t["status"] != 200 else ""),
                  title="✓ Ready" if ok else "✕ Not ready")
        for w in t.get("warnings") or []:
            st.caption(f"Warning from the API: {w}")


def _examples(ws, model_name, pipe):
    schema = pd.DataFrame([{"column": col, "kind": v["kind"], "dtype": v["dtype"],
                            "required": "yes" if v["required"] else "no (ignored)",
                            "example": v.get("example"),
                            "allowed / seen values": ", ".join(v.get("categories", [])[:8])
                            + (" …" if len(v.get("categories", [])) > 8 else "")}
                           for col, v in pipe.input_schema_.items()])
    st.markdown("#### Request schema")
    st.dataframe(schema, width="stretch", hide_index=True)
    st.caption("Missing required columns or non-numeric values in numeric columns return HTTP 422 naming the "
               "columns. Extra columns are ignored; unseen categories and missing values are handled and "
               "reported in `warnings`.")
    row = S.raw_holdout(1)
    example = _rows_payload(row)[0] if row is not None else {c: v["example"] for c, v in pipe.input_schema_.items()}
    example = {k: v for k, v in example.items() if pipe.input_schema_.get(k, {}).get("required")}
    body = {"workspace_id": ws.workspace_id, "model_name": model_name, "data": [example]}
    body_json = json.dumps(body, indent=2, default=str)
    tab_curl, tab_py, tab_csv = st.tabs(["curl", "Python", "CSV batch"])
    with tab_curl:
        st.code(f"curl -X POST {nav.API_PUBLIC_URL}/predict \\\n  -H 'Content-Type: application/json' \\\n"
                f"  -d '{json.dumps(body, default=str)}'", language="bash")
    with tab_py:
        st.code(f"import requests\n\npayload = {body_json}\n\nr = requests.post(\"{nav.API_PUBLIC_URL}/predict\", "
                "json=payload, timeout=30)\nr.raise_for_status()\nprint(r.json()[\"predictions\"], "
                "r.json()[\"warnings\"])", language="python")
    with tab_csv:
        st.code(f"curl -X POST {nav.API_PUBLIC_URL}/predict/csv/{ws.workspace_id}/{model_name.replace(' ', '%20')} "
                "\\\n  -F 'file=@new_rows.csv'", language="bash")
    st.markdown(f"Interactive API documentation: [{nav.API_PUBLIC_URL}/docs]({nav.API_PUBLIC_URL}/docs) · "
                f"schema endpoint: `GET /workspaces/{ws.workspace_id}/schema`")


def _report_data(ws, pipe, s):
    res = successful(pd.DataFrame(s.get("results") or []))
    metric = (s.get("metric") or {}).get("metric")
    lb = [f"{r['Model']}: {metric} {r.get(metric):.4f}" for r in res.to_dict("records") if r.get(metric) is not None]
    adv = s.get("advisor") or {}
    return {"dataset_name": ws.dataset_name, "dataset_shape": ws.dataset_shape, "task_type": pipe.task_type,
            "target_cols": pipe.target_cols, "split": f"{pipe.splitter.strategy}, test size {pipe.test_size}, seed {pipe.random_state}",
            "metric": metric, "metric_reason": (s.get("metric") or {}).get("reason"), "leaderboard": lb,
            "best_model": ws.best_model, "best_score": ws.best_score, "baseline": s.get("baseline"),
            "failed_models": [f"{f['Model']}: {f['Error'][:120]}" for f in s.get("failed") or []],
            "preprocessing_steps": [e for e in pipe.full_log if e.get("status") == "applied"],
            "recommendations": s.get("selection") or [],
            "recommendation_source": f"{adv.get('source')} ({adv.get('model')})" if adv else "not run"}


def render():
    ws, statuses = nav.require_step("deploy")
    c, s = S.ctx(), S.state()
    pipe = c["pipeline"]
    st.subheader("Deploy", anchor=False)
    tr = c.get("trainer") or S.rebuild_trainer()
    names = [m for m in (tr.trained_models if tr else {}) if m not in model_registry.TIME_SERIES_MODELS + ["LSTM"]]
    problem = S.prerequisite_problem("deploy")
    if problem or not names:
        st.markdown(problem or "Train at least one model first.")
        return
    default = names.index(ws.best_model) if ws.best_model in names else 0
    model_name = st.selectbox("Model to serve", names, index=default, key="w_deploy_model")
    if st.button("Run readiness check", type="primary", icon=":material/fact_check:"):
        with st.spinner("Calling /predict..."):
            chk = readiness_check(ws, model_name)
        s["deploy_check"] = chk
        ok = any(t["status"] == 200 and t["matches"] and all(t["matches"]) for t in chk["targets"])
        S.record_step("deploy", ok=ok, error=None if ok else "readiness check failed", model=model_name)
        S.save_state()
        st.rerun()
    chk = s.get("deploy_check")
    if chk and chk.get("model") == model_name:
        st.caption(f"Last check {chk['checked_at']} on {chk['rows']} held-out rows.")
        _show_check(chk)
    _examples(ws, model_name, pipe)

    st.divider()
    st.markdown("#### Export")
    c1, c2 = st.columns(2)
    with c1:
        tuned = next((t for t in reversed(s.get("tune") or []) if t["tuned_name"] == model_name), None)
        code = generate_training_code(ws.dataset_name or "dataset.csv", pipe.target_cols, model_name,
                                      (tuned or {}).get("result", {}).get("Best Params"), pipe.task_type,
                                      preprocessor=pipe, seed=ws.seed,
                                      class_weight="balanced" if (s.get("train_config") or {}).get("class_weight") else None)
        st.download_button("Training script (.py)", code, f"train_{model_name.replace(' ', '_').lower()}.py",
                           mime="text/x-python", icon=":material/code:", width="stretch")
        st.caption("Re-runs this app's pipeline with the same seed (needs this repository on the Python path).")
        with st.expander("Preview script"):
            st.code(code, language="python")
    with c2:
        if st.button("Generate experiment report", icon=":material/description:", width="stretch"):
            with st.spinner("Writing the report..."):
                s["report"] = ReportGenerator().generate(_report_data(ws, pipe, s))
            S.save_state()
        rep = s.get("report")
        if rep:
            if rep["source"] == "llm":
                ui.chips([ui.chip(f"Written by LLM {rep['model']}", "info")])
            else:
                ui.chips([ui.chip("Template fallback (no LLM)", "warning")])
                if rep.get("error"):
                    st.caption(f"Reason: {rep['error']}")
            st.download_button("Report (.md)", rep["text"], f"{(ws.dataset_name or 'experiment')}_report.md",
                               mime="text/markdown", icon=":material/download:", width="stretch")
            with st.expander("Report", expanded=True):
                st.markdown(rep["text"])
