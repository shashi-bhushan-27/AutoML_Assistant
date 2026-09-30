"""
One experimental unit = one (dataset, seed):

  1. app AutoPreprocessor.fit_transform on the full dataset (as the UI does),
     split seed = ``seed``
  2. brute force: every model returned by ModelTrainer.get_supported_models(),
     trained one at a time through ModelTrainer.run_selected_models
  3. ablation B (preprocessing) on the trained models
  4. ablation C (SHAP) on the trained models

Writes results/raw/unit_<dataset>_s<seed>.json.
"""
import contextlib
import io
import os
import time
import warnings

import numpy as np
import pandas as pd

from common import (DATASET_META, PREPROC_MODELS, RAW_DIR, SHAP_MODELS, dump_json,
                    environment_info, load_dataset)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _warm_imports():
    # run_selected_models imports these lazily; import once so timings exclude import cost
    import xgboost  # noqa: F401
    from sklearn import ensemble, linear_model, neighbors, svm, tree  # noqa: F401


def run_unit(dataset, seed, do_preproc=True, do_shap=True, shap_timeout=300.0):
    from app_backend.model_trainer import ModelTrainer
    from app_backend.preprocessing_engine.engine import AutoPreprocessor
    from app_backend.statistical_engine import analyze_dataset

    warnings.filterwarnings("ignore")
    _warm_imports()
    meta = DATASET_META[dataset]
    target = meta["target"]
    df = load_dataset(dataset)
    stats = analyze_dataset(df.copy(), target_col=target)
    task = stats["task_type"]
    _log(f"unit {dataset} seed={seed} rows={len(df)} task={task}")

    # 1) preprocessing exactly as the UI calls it (only the split seed is varied)
    prep = AutoPreprocessor(target_col=target, task_type="auto", is_time_series=False,
                            apply_smote=False, verbose=False)
    prep.splitter.random_state = seed
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = prep.fit_transform(df=df)
    prep_time = time.perf_counter() - t0
    X_train, X_test, y_train, y_test = out["X_train"], out["X_test"], out["y_train"], out["y_test"]
    prep_info = {
        "fit_transform_s": prep_time, "fit_rows_per_s": len(df) / prep_time,
        "n_train": len(X_train), "n_test": len(X_test), "n_features": X_train.shape[1],
        "rows_after_dedup": len(X_train) + len(X_test),
        "feature_dtypes": {str(k): int(v) for k, v in X_train.dtypes.astype(str).value_counts().items()},
        "scaler": type(prep.scaler.scaler).__name__ if prep.scaler.scaler is not None else None,
        "dropped_features": prep.selector.get_dropped_features(),
        "skew_transforms": prep.transformer.transforms,
        "outlier_capped_features": sorted(prep.transformer.outlier_caps),
        "encoders": {c: e["type"] for c, e in prep.encoder.encoders.items()},
        "imputers": {c: v.get("strategy") for c, v in prep.imputer.imputers.items()},
    }

    # 2) brute force over every supported model
    trainer = ModelTrainer(df, target, task, stats["is_time_series"], stats.get("time_column"))
    trainer.set_preprocessed_data(X_train, X_test, y_train, y_test)
    brute = []
    for name in trainer.get_supported_models():
        np.random.seed(seed)  # models with random_state=None draw from the global RNG
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            res = trainer.run_selected_models([name])
        wall = time.perf_counter() - t0
        rec = res.iloc[0].to_dict() if len(res) else {"Model": name, "Error": "no result"}
        rec = {k: v for k, v in rec.items() if k not in ("Per-Class Report", "Per-Target Metrics")}
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        rec["wall_s"] = wall
        brute.append(rec)
        _log(f"  {dataset} s{seed} {name:<20} {wall:7.1f}s "
             + " ".join(f"{k}={rec[k]}" for k in ("Accuracy", "F1 Score", "RMSE", "R²", "Error") if rec.get(k) is not None))

    result = {"dataset": dataset, "seed": seed, "task": task, "stats": stats,
              "preprocessing": prep_info, "brute_force": brute, "environment": environment_info()}
    path = os.path.join(RAW_DIR, f"unit_{dataset}_s{seed}.json")
    dump_json(result, path)  # checkpoint: RAG evaluation only needs the brute-force part

    # 3) preprocessing ablation
    if do_preproc:
        from preproc_ablation import run_preproc_ablation

        t0 = time.perf_counter()
        result["preproc"] = run_preproc_ablation(dataset, seed, df, target, prep, trainer.trained_models,
                                                 PREPROC_MODELS[task], task)
        _log(f"  {dataset} s{seed} preprocessing ablation done in {time.perf_counter() - t0:.0f}s")
        dump_json(result, path)

    # 4) SHAP ablation
    if do_shap:
        from shap_ablation import run_shap_ablation

        t0 = time.perf_counter()
        result["shap"] = run_shap_ablation(trainer.trained_models, SHAP_MODELS[task], X_train, X_test, task,
                                           timeout_s=shap_timeout, log=_log)
        _log(f"  {dataset} s{seed} SHAP ablation done in {time.perf_counter() - t0:.0f}s")
        dump_json(result, path)

    _log(f"unit {dataset} seed={seed} finished")
    return path


def rerun_preproc(dataset, seed):
    """Re-run only ablation B for an existing unit (same split and seed-identical model fits)."""
    import json

    from app_backend.model_trainer import ModelTrainer
    from app_backend.preprocessing_engine.engine import AutoPreprocessor
    from app_backend.statistical_engine import analyze_dataset
    from preproc_ablation import run_preproc_ablation

    warnings.filterwarnings("ignore")
    _warm_imports()
    path = os.path.join(RAW_DIR, f"unit_{dataset}_s{seed}.json")
    with open(path) as f:
        result = json.load(f)
    target = DATASET_META[dataset]["target"]
    df = load_dataset(dataset)
    stats = analyze_dataset(df.copy(), target_col=target)
    task = stats["task_type"]
    prep = AutoPreprocessor(target_col=target, task_type="auto", is_time_series=False,
                            apply_smote=False, verbose=False)
    prep.splitter.random_state = seed
    with contextlib.redirect_stdout(io.StringIO()):
        out = prep.fit_transform(df=df)
    trainer = ModelTrainer(df, target, task, stats["is_time_series"], stats.get("time_column"))
    trainer.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    for name in PREPROC_MODELS[task]:
        np.random.seed(seed)  # identical to the brute-force fit of the same model
        with contextlib.redirect_stdout(io.StringIO()):
            res = trainer.run_selected_models([name])
        ref = next(r for r in result["brute_force"] if r["Model"] == name)
        key = "RMSE" if task == "Regression" else "Accuracy"
        assert abs(float(res.iloc[0][key]) - float(ref[key])) < 1e-9, f"{name} refit differs from brute force"
    t0 = time.perf_counter()
    result["preproc"] = run_preproc_ablation(dataset, seed, df, target, prep, trainer.trained_models,
                                             PREPROC_MODELS[task], task)
    dump_json(result, path)
    _log(f"  {dataset} s{seed} preprocessing ablation re-run in {time.perf_counter() - t0:.0f}s")
