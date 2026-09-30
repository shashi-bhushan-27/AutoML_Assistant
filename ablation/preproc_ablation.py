"""
Ablation B — stateful vs stateless preprocessing (training–serving skew).

For one (dataset, seed) unit, with models already trained on the output of the
app's ``AutoPreprocessor.fit_transform``:

* offline reference  : predictions on ``X_test`` as produced by ``fit_transform``
                       (the representation behind every reported metric).
* stateful (app)     : preprocessor saved with ``AutoPreprocessor.save`` (pickle),
                       reloaded with ``AutoPreprocessor.load``; raw test rows go
                       through ``transform`` and straight into ``model.predict``,
                       as ``/predict`` does.
* stateless (app)    : a fresh ``AutoPreprocessor`` is fitted on every serving
                       batch (batch sizes 1/16/256/full) and its output is fed to
                       the model.
* ``aligned`` mode   : diagnostic only, NOT app behaviour — the same transformed
                       batch re-indexed to the training columns (missing -> 0), so
                       value skew can be measured separately from schema crashes.

Also: feature-level parity of ``transform`` vs ``fit_transform``, schema
perturbations (reordered columns, unseen category, missing column) and the real
FastAPI ``/predict`` endpoint via ``TestClient``.
"""
import contextlib
import io
import os
import time
import warnings

import numpy as np
import pandas as pd

from common import TMP_DIR

BATCH_SIZES = [1, 16, 256, "full"]
N_STREAM = 1024          # serving rows used for batch sizes 1/16/256
N_SCHEMA = 64            # rows per schema-perturbation request
N_LATENCY = 30           # single-row /predict requests timed per model
MIN_MINORITY_IN_STREAM = 50  # below this, the classification stream is minority-enriched
MODES = ("app", "aligned")

MISSING_COLUMN = {"adult": "age", "credit": "V14", "california": "MedInc"}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _predict(model, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(X))


def _agreement(pred, ref, task):
    pred, ref = np.asarray(pred), np.asarray(ref)
    if len(ref) == 0:
        return None
    if task == "Regression":
        pred, ref = pred.astype(float), ref.astype(float)
        return float(np.mean(np.abs(pred - ref) <= 1e-6 * (1 + np.abs(ref))))
    return float(np.mean(pred == ref))


def _metrics(y_true, pred, task):
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    y_true, pred = np.asarray(y_true), np.asarray(pred)
    if len(y_true) == 0:
        return {}
    if task == "Regression":
        pred = pred.astype(float)
        return {"RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
                "R2": float(r2_score(y_true, pred)) if len(y_true) > 1 else None}
    avg = "binary" if len(np.unique(y_true)) <= 2 else "weighted"
    return {"Accuracy": float(accuracy_score(y_true, pred)),
            "F1": float(f1_score(y_true, pred, average=avg, zero_division=0))}


def _err(e):
    first = str(e).splitlines()[0] if str(e) else ""
    return f"{type(e).__name__}: {first[:160]}"


def _stateful_transform(prep, batch):
    with _quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prep.transform(batch.copy())


def _stateless_transform(batch, task_type):
    from app_backend.preprocessing_engine.engine import AutoPreprocessor

    p = AutoPreprocessor(target_col=None, task_type=task_type, verbose=False)
    with _quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.fit_transform(df=batch.copy())["X_train"]


def feature_parity(X_serv, X_ref):
    """Does transform(raw) reproduce fit_transform's representation of the same rows?"""
    common = [c for c in X_ref.columns if c in X_serv.columns]
    mismatched = {}
    for c in common:
        a = pd.to_numeric(X_serv[c], errors="coerce").astype(float).values
        b = X_ref[c].astype(float).values
        bad = ~np.isclose(a, b, atol=1e-6, equal_nan=True)
        if bad.any():
            diff = np.abs(a - b)
            mismatched[c] = {"rows_differing_frac": float(bad.mean()),
                             "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else None}
    missing = [c for c in X_ref.columns if c not in X_serv.columns]
    n_cells = len(X_ref) * len(X_ref.columns)
    bad_cells = sum(v["rows_differing_frac"] for v in mismatched.values()) * len(X_ref) + len(missing) * len(X_ref)
    return {
        "n_train_features": len(X_ref.columns),
        "n_serving_features": len(X_serv.columns),
        "same_column_order": list(X_serv.columns) == list(X_ref.columns),
        "missing_at_serving": missing,
        "extra_at_serving": [c for c in X_serv.columns if c not in X_ref.columns],
        "n_common_features_mismatched": len(mismatched),
        "mismatched_features": mismatched,
        "cell_mismatch_frac": float(bad_cells / n_cells) if n_cells else None,
    }


def serve_stream(prep, models, rows, refs, y_test, train_cols, task, batch_size, stateless):
    """Serve ``rows`` in batches; one transform per batch shared by all models."""
    n = len(rows)
    size = n if batch_size == "full" else int(batch_size)
    keys = [(m, mode) for m in models for mode in MODES]
    preds = {k: pd.Series(index=rows.index, dtype=object) for k in keys}
    crashes = {k: 0 for k in keys}
    errors = {k: {} for k in keys}
    n_batches, t_transform, transform_crash, rows_out = 0, 0.0, 0, 0
    for start in range(0, n, size):
        batch = rows.iloc[start:start + size]
        n_batches += 1
        t0 = time.perf_counter()
        try:
            Xb = _stateless_transform(batch, prep.task_type) if stateless else _stateful_transform(prep, batch)
            terr = None
            rows_out += len(Xb)
        except Exception as e:
            Xb, terr = None, _err(e)
            transform_crash += 1
        t_transform += time.perf_counter() - t0
        for m, model in models.items():
            for mode in MODES:
                k = (m, mode)
                if Xb is None:
                    crashes[k] += 1
                    errors[k][terr] = errors[k].get(terr, 0) + 1
                    continue
                Xin = Xb if mode == "app" else Xb.reindex(columns=train_cols, fill_value=0)
                try:
                    preds[k].loc[Xin.index] = _predict(model, Xin)
                except Exception as e:
                    crashes[k] += 1
                    msg = _err(e)
                    errors[k][msg] = errors[k].get(msg, 0) + 1
    out = []
    for (m, mode), pred in preds.items():
        got = pred.notna()
        rec = {"model": m, "mode": mode, "batch_size": str(batch_size), "n_rows": n, "n_batches": n_batches,
               "crash_rate": crashes[(m, mode)] / n_batches, "row_coverage": float(got.mean()),
               "rows_dropped_by_preprocessor": int(n - rows_out) if transform_crash == 0 else None,
               "transform_crash_rate": transform_crash / n_batches,
               "transform_rows_per_s": n / t_transform if t_transform > 0 else None,
               "errors": dict(sorted(errors[(m, mode)].items(), key=lambda kv: -kv[1])[:3])}
        if got.any():
            idx = pred.index[got]
            r = refs[m].loc[idx].values
            p = pred.loc[idx].values
            p = p.astype(float) if task == "Regression" else p.astype(r.dtype)
            rec["agreement_vs_offline"] = _agreement(p, r, task)
            if task == "Regression":
                rec["mean_abs_pred_diff"] = float(np.mean(np.abs(p - r.astype(float))))
            rec.update({f"served_{k}": v for k, v in _metrics(y_test.loc[idx], p, task).items()})
            rec.update({f"offline_{k}": v for k, v in _metrics(y_test.loc[idx], r, task).items()})
        out.append(rec)
    return out, preds


def _agreement_between(pa, pb, task):
    """Agreement of two served prediction series on the rows both of them served."""
    both = pa.notna() & pb.notna()
    if not both.any():
        return None, 0.0
    a, b = pa[both].values, pb[both].values
    if task == "Regression":
        a, b = a.astype(float), b.astype(float)
    else:
        a, b = a.astype(str), b.astype(str)
    return _agreement(a, b, task), float(both.mean())


def make_schema_variants(raw_rows, dataset):
    base = raw_rows.iloc[:N_SCHEMA]
    variants = {"baseline": (base, None), "reordered_columns": (base[base.columns[::-1]], None)}
    cat_cols = [c for c in base.columns if base[c].dtype == object]
    if cat_cols:
        v = base.copy()
        v[cat_cols[0]] = "__unseen_category__"
        variants["unseen_category"] = (v, cat_cols[0])
    miss = MISSING_COLUMN.get(dataset)
    if miss in base.columns:
        variants["missing_column"] = (base.drop(columns=[miss]), miss)
    return variants


def schema_tests(prep, models, variants, refs, task):
    """Stateful app path (transform -> predict) under schema perturbations."""
    out = []
    for name, (rows, col) in variants.items():
        try:
            X, terr = _stateful_transform(prep, rows), None
        except Exception as e:
            X, terr = None, _err(e)
        for m, model in models.items():
            rec = {"variant": name, "model": m, "perturbed_column": col}
            if X is None:
                rec.update({"status": "crash", "error": terr})
            else:
                try:
                    p = _predict(model, X)
                    r = refs[m].loc[rows.index].values
                    p = p.astype(float) if task == "Regression" else p.astype(r.dtype)
                    rec.update({"status": "ok", "agreement_vs_offline": _agreement(p, r, task)})
                except Exception as e:
                    rec.update({"status": "crash", "error": _err(e)})
            out.append(rec)
    return out


def endpoint_tests(prep, models, variants, refs, raw_test, target, task, name):
    """Real FastAPI /predict, workspace written by the real WorkspaceManager (temp dir)."""
    from fastapi.testclient import TestClient

    from rag_ablation import temp_workspace_dir

    def payload(df):
        return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                for r in df.to_dict(orient="records")]

    results, latency = [], []
    with temp_workspace_dir(name) as wmod:
        import app_backend.main_api as api

        wm = wmod.WorkspaceManager()
        ws = wm.create_workspace(dataset_name=name, dataset_shape=(len(raw_test), raw_test.shape[1] + 1))
        ws.target_col = target
        ws.task_type = prep.task_type
        ws.status = "completed"
        wm.save_workspace(ws)
        wm.save_preprocessor(ws.workspace_id, prep)
        wm.save_trained_models(ws.workspace_id, dict(models))
        client = TestClient(api.app)
        body = lambda m, rows: {"workspace_id": ws.workspace_id, "model_name": m, "data": payload(rows)}

        for vname, (rows, _) in variants.items():
            for m in models:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    r = client.post("/predict", json=body(m, rows))
                rec = {"variant": vname, "model": m, "status_code": r.status_code,
                       "raw_fallback_triggered": "Using raw data" in buf.getvalue()}
                if r.status_code == 200:
                    ref = refs[m].loc[rows.index].values
                    p = np.asarray(r.json()["predictions"])
                    p = p.astype(float) if task == "Regression" else p.astype(ref.dtype)
                    rec["agreement_vs_offline"] = _agreement(p, ref, task)
                else:
                    try:
                        rec["detail"] = str(r.json().get("detail"))[:240]
                    except Exception:
                        rec["detail"] = r.text[:240]
                results.append(rec)

        # single-row latency through the endpoint (workspace reload + transform + predict)
        singles = [raw_test.iloc[[i]] for i in range(min(N_LATENCY, len(raw_test)))]
        pkl_models = os.path.getsize(os.path.join(wmod.WORKSPACE_DIR, f"{ws.workspace_id}_trained_models.pkl"))
        pkl_prep = os.path.getsize(os.path.join(wmod.WORKSPACE_DIR, f"{ws.workspace_id}_pipeline.pkl"))
        for m in models:
            times, codes = [], []
            with contextlib.redirect_stdout(io.StringIO()):
                client.post("/predict", json=body(m, singles[0]))   # warm-up
                for rows in singles:
                    t0 = time.perf_counter()
                    r = client.post("/predict", json=body(m, rows))
                    times.append((time.perf_counter() - t0) * 1000)
                    codes.append(r.status_code)
            model_ms = []
            for i in range(len(singles)):
                t0 = time.perf_counter()
                _predict(models[m], prep.X_test.iloc[[i]])
                model_ms.append((time.perf_counter() - t0) * 1000)
            latency.append({"model": m, "n_requests": len(times),
                            "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                            "status_codes": {str(c): codes.count(c) for c in sorted(set(codes))},
                            "model_only_median_ms": float(np.median(model_ms)),
                            "trained_models_pkl_mb": pkl_models / 1e6, "pipeline_pkl_mb": pkl_prep / 1e6})
    return results, latency


def run_preproc_ablation(dataset, seed, df, target, prep, trained_models, model_names, task):
    from app_backend.preprocessing_engine.engine import AutoPreprocessor

    models = {m: trained_models[m] for m in model_names if m in trained_models}
    X_test, y_test = prep.X_test, prep.y_test
    train_cols = list(prep.X_train.columns)
    raw_test = df.loc[X_test.index].drop(columns=[target])

    # 1) serialise -> reload with the app's own save/load
    path = os.path.join(TMP_DIR, f"prep_{dataset}_s{seed}.pkl")
    t0 = time.perf_counter()
    with _quiet():
        prep.save(path)
    t_save = time.perf_counter() - t0
    t0 = time.perf_counter()
    prep_s = AutoPreprocessor.load(path)
    t_load = time.perf_counter() - t0
    serialization = {"pickle_mb": os.path.getsize(path) / 1e6, "save_s": t_save, "load_s": t_load}
    os.remove(path)

    # 2) offline reference predictions
    refs = {m: pd.Series(_predict(mod, X_test), index=X_test.index) for m, mod in models.items()}

    # 3) feature-level parity of transform() vs fit_transform()
    t0 = time.perf_counter()
    X_serv = _stateful_transform(prep_s, raw_test)
    parity = feature_parity(X_serv, X_test)
    parity["transform_rows_per_s"] = len(raw_test) / (time.perf_counter() - t0)

    # 4) serving streams
    rng = np.random.RandomState(1000 + seed)
    idx = np.sort(rng.choice(len(raw_test), size=min(N_STREAM, len(raw_test)), replace=False))
    stream_kind = "random"
    if task != "Regression":
        minority = y_test.value_counts().idxmin()
        pos = np.where(y_test.values == minority)[0]
        if (y_test.values[idx] == minority).sum() < MIN_MINORITY_IN_STREAM:
            # e.g. Credit Card Fraud: a random 1,024-row stream holds ~2 frauds, so F1 is undefined.
            # Enrich: all minority test rows (capped at half the stream) + random majority rows.
            pos = rng.choice(pos, size=min(len(pos), N_STREAM // 2), replace=False)
            neg_pool = np.setdiff1d(np.arange(len(raw_test)), pos)
            neg = rng.choice(neg_pool, size=N_STREAM - len(pos), replace=False)
            idx = np.sort(np.concatenate([pos, neg]))
            stream_kind = f"minority-enriched ({len(pos)} of {len(idx)} rows are class {minority})"
    stream = raw_test.iloc[idx]
    serving, stateful_preds = [], {}
    for stateless in (False, True):
        for bs in BATCH_SIZES:
            rows = raw_test if bs == "full" else stream
            recs, preds = serve_stream(prep_s, models, rows, refs, y_test, train_cols, task, bs, stateless)
            for rec in recs:
                rec["preprocessing"] = "stateless" if stateless else "stateful"
                rec["stream"] = "full test set" if bs == "full" else stream_kind
                key = (rec["model"], rec["mode"])
                if not stateless:
                    stateful_preds[(bs,) + key] = preds[key]
                else:  # the metric asked for: agreement of stateless serving with stateful serving
                    for ref_mode in ("app", "aligned"):
                        agr, cov = _agreement_between(preds[key], stateful_preds[(bs, rec["model"], ref_mode)], task)
                        rec[f"agreement_vs_stateful_{ref_mode}"] = agr
                        rec[f"rows_served_by_both_{ref_mode}"] = cov
                serving.append(rec)

    # 5) schema perturbations on the stateful path + the real /predict endpoint
    variants = make_schema_variants(raw_test, dataset)
    schema = schema_tests(prep_s, models, variants, refs, task)
    try:
        endpoint, latency = endpoint_tests(prep_s, models, variants, refs, raw_test, target, task,
                                           f"api_{dataset}_s{seed}")
    except Exception as e:  # recorded, never hidden
        endpoint, latency = [{"error": _err(e)}], []

    return {"serialization": serialization, "parity": parity, "serving": serving, "schema": schema,
            "endpoint": endpoint, "endpoint_latency": latency, "stream": stream_kind}
