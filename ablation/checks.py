"""
Targeted, deterministic checks of claims in the paper against the code.
Writes results/code_checks.json.
"""
import contextlib
import io
import os
import re
import warnings

import numpy as np
import pandas as pd

from common import DATASET_META, REPO_ROOT, RESULTS_DIR, dump_json, load_dataset
from rag_ablation import supported_models_for, ui_best_model, ui_fuzzy_select

# Names an LLM plausibly emits for tabular tasks, incl. the paper's own example.
HALLUCINATION_PROBES = [
    "Random Forest", "random forest", "RandomForest", "Random Forest Classifier", "RandomForestClassifier",
    "RandomForestClassifierRegressor", "XGBoost", "XGBClassifier", "XGBRegressor", "xgboost (weighted)",
    "LightGBM", "CatBoost", "HistGradientBoostingClassifier", "Gradient Boosting Machine",
    "Support Vector Machine", "SVC", "Linear SVM", "K-Nearest Neighbors", "KNeighborsClassifier",
    "Logistic Regression", "LogisticRegression", "Neural Network", "MLPClassifier", "Stacking Ensemble",
    "Isolation Forest", "Naive Bayes", "Ridge Classifier", "Linear Regression", "Bayesian Ridge", "Prophet",
]


def _grep(pattern, paths):
    hits = []
    for p in paths:
        with open(os.path.join(REPO_ROOT, p), encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if re.search(pattern, line, flags=re.I):
                    hits.append(f"{p}:{i}: {line.strip()[:120]}")
    return hits


def check_string_label_xgboost():
    """Adult with its original string label, through the UI path (AutoPreprocessor -> ModelTrainer)."""
    from app_backend.model_trainer import ModelTrainer
    from app_backend.preprocessing_engine.engine import AutoPreprocessor

    df = load_dataset("adult").copy()
    df["class"] = np.where(df["class"] == 1, ">50K", "<=50K")
    prep = AutoPreprocessor(target_col="class", task_type="auto", verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        out = prep.fit_transform(df=df)
    tr = ModelTrainer(df, "class", "Classification")
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = tr.run_selected_models(["XGBoost", "Logistic Regression"])
    return res.to_dict(orient="records")


def check_target_encoding_leakage(seed=0):
    """SmartEncoder target-encodes 11-100-level categoricals with the FULL target vector
    before the split. Compare with the train-only encoding of the same split."""
    from sklearn.model_selection import train_test_split

    df = load_dataset("adult")
    y = df["class"]
    out = {}
    X = df.drop(columns=["class"]).drop_duplicates()
    y = y.loc[X.index]
    idx_train, idx_test = train_test_split(X.index, test_size=0.2, stratify=y, random_state=seed)
    for col in ["education", "occupation", "native-country"]:
        s = X[col].fillna("__MISSING__").astype(str)
        full = y.groupby(s).mean()
        train_only = y.loc[idx_train].groupby(s.loc[idx_train]).mean()
        joined = pd.concat([full.rename("full"), train_only.rename("train")], axis=1)
        unseen_in_train = int(joined["train"].isna().sum())
        d = (joined["full"] - joined["train"]).abs()
        test_rows_in_unseen = int(s.loc[idx_test].isin(joined.index[joined["train"].isna()]).sum())
        out[col] = {"n_levels": int(len(joined)), "max_abs_diff": float(d.max()),
                    "mean_abs_diff": float(d.mean()), "levels_only_in_test": unseen_in_train,
                    "test_rows_with_label_derived_code_from_own_split": test_rows_in_unseen}
    return out


def check_regression_best_model():
    """The UI never stores best_model for regression workspaces (case mismatch)."""
    import json

    from common import RAW_DIR

    p = os.path.join(RAW_DIR, "unit_california_s0.json")
    if not os.path.exists(p):
        return {"skipped": "unit_california_s0.json not available yet"}
    with open(p) as f:
        brute = json.load(f)["brute_force"]
    res = pd.DataFrame([r for r in brute if not r.get("Error")])
    best, score = ui_best_model(res, "Regression")  # value written by the analysis step
    tracked = os.path.join(REPO_ROOT, "workspaces", "index.json")
    idx = {"workspaces": []}
    if os.path.exists(tracked):
        with open(tracked) as f:
            idx = json.load(f)
    reg = [w for w in idx["workspaces"] if w.get("task_type") == "Regression"]
    return {"best_model_recorded_for_california": best, "best_score": score,
            "repo_workspaces_regression": len(reg),
            "repo_workspaces_regression_with_best_model": sum(1 for w in reg if w.get("best_model"))}


def check_meta_similarity():
    """Pairwise outcome of the app's meta-learning rule between the benchmarks (including each dataset
    against itself, i.e. a prior run on the same data)."""
    from app_backend.statistical_engine import analyze_dataset

    stats = {ds: analyze_dataset(load_dataset(ds).copy(), target_col=DATASET_META[ds]["target"])
             for ds in DATASET_META}
    try:
        from app_backend import meta_learning
    except ImportError:
        meta_learning = None
    pairs = []
    for a in stats:
        for b in stats:
            same_task = stats[a]["task_type"] == stats[b]["task_type"]
            if meta_learning is not None:
                sim = meta_learning.similarity(stats[a]["profile"], stats[b]["profile"])
                pairs.append({"query": a, "prior": b, "same_task": same_task, "similarity": sim,
                              "threshold": meta_learning.SIMILARITY_THRESHOLD,
                              "match": same_task and sim >= meta_learning.SIMILARITY_THRESHOLD})
                continue
            if a == b:
                continue
            ra, ca, rb, cb = stats[a]["rows"], stats[a]["columns"], stats[b]["rows"], stats[b]["columns"]
            row_diff = abs(rb - ra) / max(ra, 1)
            col_diff = abs(cb - ca) / max(ca, 1)
            pairs.append({"query": a, "prior": b, "same_task": same_task, "row_diff": row_diff,
                          "col_diff": col_diff, "match": same_task and (row_diff < 0.5 or col_diff < 0.3)})
    return pairs


def check_app_target_encoding_train_only(seed=0):
    """After the fixes: the app's fitted target encodings equal a TargetEncoder fitted on the training rows only."""
    from sklearn.preprocessing import TargetEncoder

    from app_backend.preprocessing_engine.encoder import as_str
    from app_backend.preprocessing_engine.engine import AutoPreprocessor

    df = load_dataset("adult")
    prep = AutoPreprocessor(target_col="class", verbose=False, random_state=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        out = prep.fit_transform(df=df)
    res = {}
    for col, info in prep.encoder.encoders.items():
        if info["type"] != "target":
            continue
        raw = df.loc[out["X_train"].index, col].to_numpy()
        values = as_str(np.where(pd.isna(raw), prep.imputer.imputers[col]["value"], raw).astype(object))
        ref = TargetEncoder(target_type="binary", smooth="auto").fit(values.reshape(-1, 1), np.asarray(out["y_train"]))
        app = info["encoder"]
        same_levels = list(ref.categories_[0]) == list(app.categories_[0])
        res[col] = {"levels": int(len(app.categories_[0])), "same_levels_as_train_only": same_levels,
                    "max_abs_diff_vs_train_only": float(np.max(np.abs(ref.encodings_[0] - app.encodings_[0])))
                    if same_levels else None}
    return res


def check_fuzzy_probes():
    rows = []
    for task in ("Classification", "Regression"):
        sup = supported_models_for(task)
        for name in HALLUCINATION_PROBES:
            sel, types, used_default = ui_fuzzy_select([name], sup, task)
            rows.append({"task": task, "name": name, "match_type": types[0],
                         "mapped_to": sel[0] if types[0] != "unmatched" else None,
                         "ui_default_used": used_default, "exact_in_registry": name in sup})
    return rows


def check_paper_vs_code():
    py = ["app_backend/llm_rag_core.py", "app_backend/model_trainer.py", "app_backend/shap_explainer.py",
          "app_backend/workspace_manager.py", "app_frontend/main_ui.py"]
    return {
        "levenshtein_or_threshold_0.85": _grep(r"levenshtein|difflib|rapidfuzz|fuzzywuzzy|0\.85", py),
        "ensemble_or_stacking_models": _grep(r"Stacking|Voting|ensemble\.Stack", ["app_backend/model_trainer.py"]),
        "llm_model_strings": _grep(r"model_name\s*=", ["app_backend/llm_rag_core.py"]),
        "llm_output_validation": _grep(r"json\.loads|recommendations", ["app_backend/llm_rag_core.py"]),
        "similarity_rule": _grep(r"row_diff < 0\.5|col_diff", ["app_backend/workspace_manager.py"]),
        "prophet_fallback_rule": _grep(r"smart_fallback\s*=", ["app_backend/llm_rag_core.py"]),
        "api_raw_data_fallback": _grep(r"Using raw data", ["app_backend/main_api.py"]),
        "transform_steps": _grep(r"X = self\.(transformer|imputer|encoder|scaler|selector)\.", [
            "app_backend/preprocessing_engine/engine.py"]),
    }


def check_retrieval_latency(n=30):
    """Latency of the app's FAISS retriever (MiniLM embedding + top-3 search), warm."""
    import time

    from app_backend.llm_rag_core import get_rag_chain
    from app_backend.statistical_engine import analyze_dataset

    t0 = time.perf_counter()
    retriever = get_rag_chain()
    load_s = time.perf_counter() - t0
    out = {"index_load_s": load_s, "n_chunks": int(retriever.vectorstore.index.ntotal), "per_dataset": {}}
    for ds in DATASET_META:
        stats = analyze_dataset(load_dataset(ds).copy(), target_col=DATASET_META[ds]["target"])
        q = f"Dataset Statistics: {str(stats)}"
        retriever.invoke(q)  # warm-up
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            docs = retriever.invoke(q)
            ts.append((time.perf_counter() - t0) * 1000)
        out["per_dataset"][ds] = {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
                                  "rules": [d.page_content.strip().splitlines()[0][:60] for d in docs]}
    return out


def check_no_cast_linear_explainer(seed=0):
    """Reproduce the no-float32-cast failure of LinearExplainer on Adult (bool one-hot columns)."""
    import traceback

    import shap

    from app_backend.model_trainer import ModelTrainer
    from app_backend.preprocessing_engine.engine import AutoPreprocessor

    df = load_dataset("adult")
    prep = AutoPreprocessor(target_col="class", task_type="auto", verbose=False)
    prep.splitter.random_state = seed
    with contextlib.redirect_stdout(io.StringIO()):
        out = prep.fit_transform(df=df)
    tr = ModelTrainer(df, "class", "Classification")
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr.run_selected_models(["Logistic Regression"])
    model = tr.trained_models["Logistic Regression"]
    X = out["X_train"].reset_index(drop=True)
    res = {"dtypes": {str(k): int(v) for k, v in X.dtypes.astype(str).value_counts().items()}}
    for name, data in [("float32_cast", X.astype(np.float32)), ("no_cast", X)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap.LinearExplainer(model, data, feature_perturbation="correlation_dependent")
            res[name] = "ok"
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            res[name] = {"error": f"{type(e).__name__}: {e}",
                         "raised_in": [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in tb[-3:]]}
    # the bool-subtract failure mode named in the paper, on a pure-bool array
    try:
        b = X.select_dtypes(bool).values
        _ = b - b.mean(0).astype(bool)
        res["numpy_bool_subtract"] = "ok"
    except Exception as e:
        res["numpy_bool_subtract"] = f"{type(e).__name__}: {e}"
    return res


def check_adult_capital_columns(seeds=(0, 1, 2, 3, 4)):
    """Diagnostic (not app behaviour): XGBoost on the app's Adult features, with and without the raw
    capital-gain / capital-loss columns that IQR capping turns into constants and the selector drops."""
    from app_backend.model_trainer import ModelTrainer
    from app_backend.preprocessing_engine.engine import AutoPreprocessor

    df = load_dataset("adult")
    res = {"app_features": [], "with_raw_capital_columns": [], "constant_after_capping": {}}
    for seed in seeds:
        prep = AutoPreprocessor(target_col="class", task_type="auto", verbose=False)
        prep.splitter.random_state = seed
        with contextlib.redirect_stdout(io.StringIO()):
            out = prep.fit_transform(df=df)
        caps = prep.transformer.outlier_caps
        res["constant_after_capping"] = {c: list(map(float, caps[c])) for c in ("capital-gain", "capital-loss") if c in caps}
        for key, extra in (("app_features", False), ("with_raw_capital_columns", True)):
            Xtr, Xte = out["X_train"].copy(), out["X_test"].copy()
            if extra:
                for c in ("capital-gain", "capital-loss"):
                    Xtr[c] = df.loc[Xtr.index, c].values
                    Xte[c] = df.loc[Xte.index, c].values
            tr = ModelTrainer(df, "class", "Classification")
            tr.set_preprocessed_data(Xtr, Xte, out["y_train"], out["y_test"])
            np.random.seed(seed)
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = tr.run_selected_models(["XGBoost"]).iloc[0]
            res[key].append(float(r["Accuracy"]))
    for key in ("app_features", "with_raw_capital_columns"):
        v = np.asarray(res[key])
        res[key + "_mean"], res[key + "_std"] = float(v.mean()), float(v.std(ddof=1))
    return res


def run_checks():
    out = {
        "adult_capital_columns": check_adult_capital_columns(),
        "shap_no_cast_linear": check_no_cast_linear_explainer(),
        "retrieval_latency": check_retrieval_latency(),
        "paper_vs_code": check_paper_vs_code(),
        "fuzzy_matcher_probes": check_fuzzy_probes(),
        "meta_learning_similarity": check_meta_similarity(),
        "regression_best_model_bookkeeping": check_regression_best_model(),
        "adult_string_label_training": check_string_label_xgboost(),
        "target_encoding_leakage_adult": check_target_encoding_leakage(),
    }
    try:
        out["app_target_encoding_train_only"] = check_app_target_encoding_train_only()
    except ImportError as exc:  # baseline code has no as_str helper
        out["app_target_encoding_train_only"] = {"skipped": str(exc)}
    path = os.path.join(RESULTS_DIR, "code_checks.json")
    dump_json(out, path)
    print("wrote", path)
    return out
