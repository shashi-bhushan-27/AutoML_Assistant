"""
Ablation C — SHAP explainer routing and float32 pre-casting.

Configurations (subclasses of the app's ``SHAPExplainer``; nothing else changes):

  full           the app class as-is: keyword routing on the class name + float32 cast
  forced_tree    ``_get_model_type`` always returns "tree"   (TreeExplainer for all)
  forced_kernel  ``_get_model_type`` always returns "kernel" (KernelExplainer for all)
  no_cast        routing kept, ``_ensure_dataframe`` does not cast to float32, so
                 bool one-hot columns reach SHAP unchanged

Every job calls ``compute_shap_values()`` (app default: first 200 test rows,
50-row background for KernelExplainer) in a forked child process with a wall-
clock timeout (default 300 s). The child reports runtime, peak additional RSS
and, for KernelExplainer, how many rows were finished (used to extrapolate the
time to completion when the job times out).
"""
import contextlib
import io
import multiprocessing as mp
import os
import threading
import time
import traceback

import numpy as np
import pandas as pd

CONFIGS = ["full", "forced_tree", "forced_kernel", "no_cast"]
MAX_ROWS = 200  # SHAPExplainer.compute_shap_values default


def _explainer_class(config):
    from app_backend.shap_explainer import SHAPExplainer

    if config == "full":
        return SHAPExplainer

    if config == "forced_tree":
        class ForcedTree(SHAPExplainer):
            def _get_model_type(self):
                return "tree"
        return ForcedTree

    if config == "forced_kernel":
        class ForcedKernel(SHAPExplainer):
            def _get_model_type(self):
                return "kernel"
        return ForcedKernel

    if config == "no_cast":
        class NoCast(SHAPExplainer):
            @staticmethod
            def _ensure_dataframe(X, feature_names=None):
                # identical to the app version minus the float32 cast
                if isinstance(X, pd.DataFrame):
                    return X.reset_index(drop=True)
                if isinstance(X, np.ndarray):
                    cols = feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])]
                    return pd.DataFrame(X, columns=cols)
                return pd.DataFrame(X)
        return NoCast

    raise ValueError(config)


def _child(conn, config, model, X_train, X_test, task, progress, peak_mb):
    import psutil
    import shap

    proc = psutil.Process()
    base = proc.memory_info().rss
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                cur = (proc.memory_info().rss - base) / 1e6
                if cur > peak_mb.value:
                    peak_mb.value = cur
            except Exception:
                pass
            time.sleep(0.05)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    # progress counter for KernelExplainer (one call to .explain per explained row)
    orig_explain = shap.KernelExplainer.explain

    def counted_explain(self, *a, **k):
        r = orig_explain(self, *a, **k)
        progress.value += 1
        return r

    shap.KernelExplainer.explain = counted_explain

    out = {"config": config}
    buf = io.StringIO()
    try:
        cls = _explainer_class(config)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            ex = cls(model, X_train, X_test, task_type=task)
            vals = ex.compute_shap_values(max_rows=MAX_ROWS)
        out["runtime_s"] = time.perf_counter() - t0
        out["route"] = ex._get_model_type()
        out["explainer"] = type(ex._explainer).__name__ if ex._explainer is not None else None
        out["X_test_dtypes"] = {str(k): int(v) for k, v in ex.X_test.dtypes.astype(str).value_counts().items()}
        if vals is None:
            out["status"] = "error"
        else:
            vals = np.asarray(vals, dtype=float)
            finite = np.isfinite(vals)
            out["shape"] = list(vals.shape)
            out["nan_frac"] = float(1 - finite.mean())
            out["status"] = "ok" if finite.all() else "nan_values"
        out["message"] = buf.getvalue().strip()[-400:]
    except Exception as e:  # SHAPExplainer catches most errors itself; anything else is recorded
        out["status"] = "error"
        out["message"] = f"{type(e).__name__}: {e}"[-400:] + " | " + traceback.format_exc()[-300:]
    finally:
        stop.set()
    out["peak_mem_mb"] = float(peak_mem_value(peak_mb))
    out["kernel_rows_done"] = int(progress.value)
    conn.send(out)
    conn.close()


def peak_mem_value(v):
    return max(0.0, v.value)


def run_job(config, model, X_train, X_test, task, timeout_s=300.0):
    ctx = mp.get_context("fork")
    progress = ctx.Value("i", 0)
    peak = ctx.Value("d", 0.0)
    parent, child = ctx.Pipe(duplex=False)
    t0 = time.perf_counter()
    p = ctx.Process(target=_child, args=(child, config, model, X_train, X_test, task, progress, peak))
    p.start()
    child.close()
    res = None
    if parent.poll(timeout_s):
        try:
            res = parent.recv()
        except EOFError:
            res = None
    wall = time.perf_counter() - t0
    if res is None:
        timed_out = p.is_alive()
        p.kill()
        p.join()
        rows = int(progress.value)
        res = {"config": config, "status": "timeout" if timed_out else "crashed_process",
               "runtime_s": wall, "peak_mem_mb": float(peak_mem_value(peak)), "kernel_rows_done": rows,
               "exitcode": p.exitcode}
        if timed_out and rows > 0:
            res["extrapolated_runtime_s"] = wall * MAX_ROWS / rows
    else:
        p.join()
    res["wall_s"] = wall
    res["timeout_s"] = timeout_s
    return res


def run_shap_ablation(trained_models, model_names, X_train, X_test, task, timeout_s=300.0, log=print):
    rows = []
    for m in model_names:
        model = trained_models.get(m)
        if model is None:
            continue
        for cfg in CONFIGS:
            r = run_job(cfg, model, X_train, X_test, task, timeout_s=timeout_s)
            r["model"] = m
            r["model_class"] = type(model).__name__
            rows.append(r)
            log(f"    shap {m:<20} {cfg:<14} {r['status']:<10} {r.get('runtime_s', 0):7.1f}s "
                f"mem+{r.get('peak_mem_mb', 0):.0f}MB {r.get('explainer', '')} {str(r.get('message', ''))[:90]}")
    return rows
