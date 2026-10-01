"""
Ablation A — RAG recommendation engine.

Configurations (one component changed at a time):

  as_shipped   ModelAdvisor exactly as in the repository (Groq model
               ``llama-3.3-70b-versatile``). Recorded to document what the app
               does today; Groq no longer serves that model, so every call ends
               in the advisor's exception fallback.
  full         FAISS retrieval (k=3) + meta-learning priors + UI fuzzy matcher.
               The LLM is substituted (default ``openai/gpt-oss-120b``) because
               the hard-coded model is unavailable; everything else is the app.
  no_rag       Same LLM and prompt template, but the retriever returns no
               documents (empty ``{context}``).
  no_meta      ``similar_workspaces=None`` (no historical priors in the prompt).
  no_fuzzy     Same advisor as ``full``; the raw recommended names go straight to
               ``ModelTrainer.run_selected_models`` (UI matcher bypassed).
  full_sameprior  Supplementary: ``full`` plus a prior workspace of the *same*
               dataset (best case for meta-learning).

Priors for ``full`` come from completed workspaces of the *other* benchmark
datasets, seeded through the real ``WorkspaceManager`` API in a temporary
workspace directory and retrieved with the real ``find_similar_workspaces``.

Model fits are deterministic given the split seed, so the metrics of a selected
model set are looked up from the brute-force run of the same (dataset, seed)
instead of being refit; the name->class dispatch still goes through the real
``ModelTrainer.run_selected_models``.
"""
import contextlib
import io
import json
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common import (DATASET_META, RAW_DIR, SEEDS, TMP_DIR, load_dataset, load_json,
                    primary_metrics)

LLM_LOG = os.path.join(RAW_DIR, "llm_calls.jsonl")
DEFAULT_SUBSTITUTE = "openai/gpt-oss-120b"
CONFIGS = ["as_shipped", "full", "no_rag", "no_meta", "no_fuzzy", "full_sameprior"]

# After the fixes the matcher and the best-model bookkeeping live in the backend
# (app_backend.model_matcher / app_backend.leaderboard) and are used directly; the
# verbatim UI ports below are kept for the baseline run on older code.
try:
    from app_backend.llm_rag_core import get_llm_model
    from app_backend.model_matcher import select_models as _backend_select
    from app_backend.leaderboard import select_best_model as _backend_best
    SHIPPED_MODEL = get_llm_model()          # the model the app now uses by default (GROQ_MODEL)
except ImportError:  # baseline code
    _backend_select = _backend_best = None
    SHIPPED_MODEL = "llama-3.3-70b-versatile"


def ui_fuzzy_select(recommendations, all_supported_models, task):
    """Model-name matching as the app does it: the backend matcher when present, otherwise a verbatim
    port of the old app_frontend/main_ui.py matcher.

    Returns (selection, per-name match types, used_default_fallback).
    """
    if _backend_select is not None:
        selection, results, used_default = _backend_select(list(recommendations or []), all_supported_models, task)
        return selection, [r.match_type for r in results], used_default
    return _legacy_ui_fuzzy_select(recommendations, all_supported_models, task)


def _legacy_ui_fuzzy_select(recommendations, all_supported_models, task):
    """Port of the pre-fix app_frontend/main_ui.py, Tab 3 "Fuzzy matching implementation"."""
    default_selection = []
    match_types = []
    normalized_supported = {m.lower().replace(" ", ""): m for m in all_supported_models}

    for rec in recommendations:
        rec_clean = str(rec).lower().replace(" ", "")
        # Direct match
        if rec in all_supported_models:
            default_selection.append(rec)
            match_types.append("exact")
        # Fuzzy match
        elif rec_clean in normalized_supported:
            default_selection.append(normalized_supported[rec_clean])
            match_types.append("normalized")
        # Partial match (e.g. "Random Forest Classifier" -> "Random Forest")
        else:
            matched = False
            for clean_key, valid_name in normalized_supported.items():
                if clean_key in rec_clean or rec_clean in clean_key:
                    default_selection.append(valid_name)
                    matched = True
                    break
            match_types.append("substring" if matched else "unmatched")

    # Deduplicate (the UI uses list(set(...)); sorted here only for stable logs)
    default_selection = sorted(set(default_selection))

    used_default = False
    if not default_selection:
        used_default = True
        if task == "Regression":
            default_selection = ["XGBoost", "Random Forest", "Gradient Boosting"]
        else:
            default_selection = ["XGBoost", "Random Forest", "Logistic Regression"]
        default_selection = [m for m in default_selection if m in all_supported_models]
    return default_selection, match_types, used_default


def ui_best_model(results_df: pd.DataFrame, ws_task_type: str, minority_share=None):
    """Best-model bookkeeping as the app does it (backend when present, else the old UI port)."""
    if _backend_best is not None:
        best = _backend_best(results_df, ws_task_type, minority_share)
        return (best["model"], best["score"]) if best else (None, None)
    return _legacy_ui_best_model(results_df, ws_task_type)


def _legacy_ui_best_model(results_df: pd.DataFrame, ws_task_type: str):
    """Port of the pre-fix best-model bookkeeping in main_ui.py (training block).

    Note the comparison with lower-case "regression": the analysis step stores
    ``stats['task_type']`` ("Regression"), so for regression the UI looks for an
    "Accuracy" column, finds none and never records a best model.
    """
    sort_col = "RMSE" if ws_task_type == "regression" else "Accuracy"
    if sort_col in results_df.columns:
        if sort_col == "RMSE":
            best_idx = results_df[sort_col].idxmin()
        else:
            best_idx = results_df[sort_col].idxmax()
        return results_df.loc[best_idx, "Model"], float(results_df.loc[best_idx, sort_col])
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Temporary workspace store (the real WorkspaceManager, redirected to a temp dir)
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def temp_workspace_dir(name: str):
    import app_backend.workspace_manager as wmod

    root = os.path.join(TMP_DIR, name)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    saved = (wmod.WORKSPACE_DIR, wmod.DATA_DIR, wmod.UPLOADS_DIR, wmod.WORKSPACE_INDEX)
    wmod.WORKSPACE_DIR = os.path.join(root, "workspaces")
    wmod.DATA_DIR = os.path.join(root, "data")
    wmod.UPLOADS_DIR = os.path.join(root, "data", "uploads")
    wmod.WORKSPACE_INDEX = os.path.join(wmod.WORKSPACE_DIR, "index.json")
    try:
        yield wmod
    finally:
        wmod.WORKSPACE_DIR, wmod.DATA_DIR, wmod.UPLOADS_DIR, wmod.WORKSPACE_INDEX = saved


def seed_prior_workspace(wm, dataset: str, stats: dict, brute_results: List[dict]):
    """Create a completed workspace the way the UI would after training all models."""
    results_df = pd.DataFrame([r for r in brute_results if not r.get("Error")])
    # the UI passes df.shape; Workspace.from_dict() calls tuple(dataset_shape), so it must not be None
    ws = wm.create_workspace(dataset_name=f"{dataset}.csv", dataset_shape=(stats["rows"], stats["columns"]))
    ws.task_type = stats["task_type"]          # value stored by the analysis step
    ws.profile_summary = stats
    ws.recommendations = list(results_df["Model"])
    best, score = ui_best_model(results_df, ws.task_type, stats.get("minority_class_share"))
    ws.best_model, ws.best_score = best, score
    ws.status = "completed"
    wm.save_workspace(ws)
    return {"dataset": dataset, "best_model": best, "best_score": score}


# ─────────────────────────────────────────────────────────────────────────────
# Advisor variants
# ─────────────────────────────────────────────────────────────────────────────

class _ChainProxy:
    """Wraps the RetrievalQA chain: logs query/context/usage, retries on 429."""

    def __init__(self, chain, retriever, max_rate_limit_retries: int = 8):
        self._chain = chain
        self._retriever = retriever
        self._max_retries = max_rate_limit_retries
        self.last = {}

    def invoke(self, query):
        from langchain_core.callbacks import get_usage_metadata_callback

        rec = {"query": query, "context_rules": [], "attempts": 0, "error": None}
        try:
            docs = self._retriever.invoke(query) if self._retriever is not None else []
            rec["context_rules"] = [_rule_headers(d.page_content) for d in docs]
            rec["context_chars"] = sum(len(d.page_content) for d in docs)
        except Exception as exc:  # retrieval problems are recorded, not hidden
            rec["context_error"] = repr(exc)
        self.last = rec
        for attempt in range(self._max_retries + 1):
            rec["attempts"] = attempt + 1
            try:
                with get_usage_metadata_callback() as cb:
                    out = self._chain.invoke(query)
                rec["usage"] = {k: dict(v) for k, v in cb.usage_metadata.items()}
                rec["raw_response"] = out.get("result") if isinstance(out, dict) else str(out)
                return out
            except Exception as exc:
                msg = str(exc)
                rec["error"] = f"{type(exc).__name__}: {msg[:500]}"
                if ("429" in msg or "rate_limit" in msg.lower()) and attempt < self._max_retries:
                    time.sleep(min(90, 15 * (attempt + 1)))
                    continue
                raise


def _rule_headers(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("[RULE")] or [text.strip()[:60]]


def build_advisor(llm_model: Optional[str], use_rag: bool = True):
    """Instantiate the app's ModelAdvisor, optionally swapping the LLM / retriever."""
    from typing import List as _List

    from langchain_classic.chains import RetrievalQA
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_groq import ChatGroq

    from app_backend.llm_rag_core import ModelAdvisor

    class EmptyRetriever(BaseRetriever):
        """w/o RAG: the knowledge base contributes nothing to the prompt."""

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> _List[Document]:
            return []

    advisor = ModelAdvisor()  # real constructor: FAISS retriever + ChatGroq(llama-3.3-70b-versatile)
    if llm_model and llm_model != SHIPPED_MODEL:
        advisor.llm = ChatGroq(temperature=0.3, model_name=llm_model)  # same temperature as the app
    retriever = advisor.retriever if use_rag else EmptyRetriever()
    chain = RetrievalQA.from_chain_type(
        llm=advisor.llm, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": advisor.prompt},
    )
    advisor.chain = _ChainProxy(chain, retriever if use_rag else None)
    return advisor


def supported_models_for(task: str) -> List[str]:
    from app_backend.model_trainer import ModelTrainer

    return ModelTrainer(pd.DataFrame(), None, task_type=task).get_supported_models()


def classify_advisor_output(out) -> str:
    """Which branch of ModelAdvisor.get_recommendations produced the result."""
    if not isinstance(out, dict):
        return "non_dict"
    if out.get("source") == "llm":  # after the fixes the advisor states its source explicitly
        return "json"
    reasons = out.get("reasoning") or []
    first = reasons[0] if isinstance(reasons, list) and reasons else None
    if first == "System fallback due to error.":
        return "exception_fallback"
    if first == "Fallback selection.":
        return "parse_fallback"
    if first == "Extracted from AI response.":
        return "keyword_extraction"
    if first == "Selected based on general performance.":
        return "list_parser"
    if "recommendations" not in out:
        return "json_missing_key"
    return "json"


# ─────────────────────────────────────────────────────────────────────────────
# LLM call collection
# ─────────────────────────────────────────────────────────────────────────────

def _load_units(dataset: str) -> Dict[int, dict]:
    units = {}
    for s in SEEDS:
        p = os.path.join(RAW_DIR, f"unit_{dataset}_s{s}.json")
        if os.path.exists(p):
            units[s] = load_json(p)
    return units


def collect_llm_calls(datasets: List[str], n_calls: int, llm_model: str,
                      configs: List[str] = None, pause_s: float = 8.0):
    """Make the LLM calls for every (dataset, config) and append them to llm_calls.jsonl.

    Resumable: calls already present in the log are skipped.
    """
    from app_backend.statistical_engine import analyze_dataset

    configs = configs or CONFIGS
    done = set()
    if os.path.exists(LLM_LOG):
        with open(LLM_LOG) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dataset"], r["config"], r["call_idx"]))

    stats_by_ds = {}
    for ds in DATASET_META:
        df = load_dataset(ds)
        stats_by_ds[ds] = analyze_dataset(df.copy(), target_col=DATASET_META[ds]["target"])

    # Seed-0 brute-force results provide the "completed prior sessions".
    brute0 = {}
    for ds in DATASET_META:
        u = _load_units(ds).get(0)
        if u is None:
            raise RuntimeError(f"unit_{ds}_s0.json missing: run the training units for seed 0 first")
        brute0[ds] = u["brute_force"]

    advisors = {}

    def get_advisor(kind):
        if kind not in advisors:
            if kind == "shipped":
                advisors[kind] = build_advisor(SHIPPED_MODEL, use_rag=True)
            elif kind == "rag":
                advisors[kind] = build_advisor(llm_model, use_rag=True)
            elif kind == "norag":
                advisors[kind] = build_advisor(llm_model, use_rag=False)
        return advisors[kind]

    for ds in datasets:
        stats = stats_by_ds[ds]
        task = stats["task_type"]
        supported = supported_models_for(task)
        others = [o for o in DATASET_META if o != ds]

        with temp_workspace_dir(f"meta_{ds}_other") as wmod:
            wm = wmod.WorkspaceManager()
            priors_other = [seed_prior_workspace(wm, o, stats_by_ds[o], brute0[o]) for o in others]
            similar_other = wm.find_similar_workspaces(stats)
        with temp_workspace_dir(f"meta_{ds}_same") as wmod:
            wm = wmod.WorkspaceManager()
            priors_same = [seed_prior_workspace(wm, o, stats_by_ds[o], brute0[o]) for o in others]
            priors_same.append(seed_prior_workspace(wm, ds, stats, brute0[ds]))
            similar_same = wm.find_similar_workspaces(stats)

        meta_info = {"priors_other": priors_other, "similar_other": similar_other,
                     "priors_same": priors_same, "similar_same": similar_same}
        print(f"[llm] {ds}: similar(other datasets)={len(similar_other)} similar(+same dataset)={len(similar_same)}",
              flush=True)

        for i in range(n_calls):
            for cfg in configs:
                if (ds, cfg, i) in done:
                    continue
                if cfg == "as_shipped":
                    adv, sim, model_used = get_advisor("shipped"), similar_other, SHIPPED_MODEL
                elif cfg == "no_rag":
                    adv, sim, model_used = get_advisor("norag"), similar_other, llm_model
                elif cfg == "no_meta":
                    adv, sim, model_used = get_advisor("rag"), None, llm_model
                elif cfg == "full_sameprior":
                    adv, sim, model_used = get_advisor("rag"), similar_same, llm_model
                else:  # full, no_fuzzy
                    adv, sim, model_used = get_advisor("rag"), similar_other, llm_model

                buf = io.StringIO()
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(buf):
                    out = adv.get_recommendations(stats, supported, similar_workspaces=sim)
                latency = time.perf_counter() - t0
                proxy = adv.chain.last
                rec = {
                    "dataset": ds, "config": cfg, "call_idx": i, "llm_model": model_used,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": latency,
                    "advisor_output": out, "advisor_branch": classify_advisor_output(out),
                    "history_in_prompt": "HISTORICAL INTELLIGENCE" in (proxy.get("query") or ""),
                    "n_similar_workspaces": len(sim) if sim else 0,
                    "context_rules": proxy.get("context_rules"),
                    "context_chars": proxy.get("context_chars"),
                    "raw_response": proxy.get("raw_response"),
                    "llm_error": proxy.get("error") if proxy.get("raw_response") is None else None,
                    "attempts": proxy.get("attempts"), "usage": proxy.get("usage"),
                    "meta": meta_info if i == 0 else None,
                }
                with open(LLM_LOG, "a") as f:
                    f.write(json.dumps(rec, default=_json_default) + "\n")
                print(f"[llm] {ds} {cfg} #{i}: {rec['advisor_branch']} {latency:.1f}s "
                      f"recs={(out or {}).get('recommendations') if isinstance(out, dict) else out}", flush=True)
                if cfg != "as_shipped":
                    time.sleep(pause_s)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation of recommendation sets against the brute-force cache
# ─────────────────────────────────────────────────────────────────────────────

def run_names_through_trainer(names, task: str, cached: Dict[str, dict]):
    """Send names through the real ModelTrainer.run_selected_models dispatch.

    ``train_sklearn_model`` is patched to return the cached result of the same
    (dataset, seed) fit, so no model is refit.
    """
    from app_backend.model_trainer import ModelTrainer

    class CachedTrainer(ModelTrainer):
        def train_sklearn_model(self, name, model_class, **kwargs):
            r = cached.get(name)
            if r is None:
                return {"Model": name, "Error": "not in brute-force cache"}
            return dict(r)

    tr = CachedTrainer(pd.DataFrame(), None, task_type=task)
    tr._data_is_set = True
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            df = tr.run_selected_models(names)
        exc = None
    except Exception as e:  # e.g. unhashable recommendation objects
        df, exc = pd.DataFrame(), f"{type(e).__name__}: {e}"
    skipped = [ln.split("Warning: ")[1].split(" not found")[0]
               for ln in buf.getvalue().splitlines() if "not found in model map" in ln]
    skipped += [f["Model"] for f in getattr(tr, "failed_models", [])
                if "Not supported" in f["Error"] or "Invalid model name" in f["Error"]]
    return df, skipped, exc


def evaluate_calls(records: List[dict], units_by_ds: Dict[str, Dict[int, dict]]) -> pd.DataFrame:
    """One row per (LLM call, seed, evaluation mode)."""
    rows = []
    for rec in records:
        ds = rec["dataset"]
        task = DATASET_META[ds]["task"]
        sel_metric, sec_metric, higher = primary_metrics(task)
        supported = supported_models_for(task)
        out = rec["advisor_output"]
        raw_names = out.get("recommendations", []) if isinstance(out, dict) else (out or [])
        if not isinstance(raw_names, list):
            raw_names = [raw_names]

        modes = ["fuzzy"]
        if rec["config"] == "no_fuzzy":
            modes = ["raw"]
        elif rec["config"] == "full":
            modes = ["fuzzy", "raw_paired"]  # paired no-fuzzy analysis on the same calls

        for mode in modes:
            if mode == "fuzzy":
                names, match_types, used_default = ui_fuzzy_select(raw_names, supported, task)
            else:
                names, match_types, used_default = list(raw_names), None, False
            for seed, unit in units_by_ds[ds].items():
                cached = {r["Model"]: r for r in unit["brute_force"]}
                res, skipped, exc = run_names_through_trainer(names, task, cached)
                ok = res[~res["Error"].notna()] if ("Error" in res.columns and len(res)) else res
                if len(ok) and sel_metric in ok.columns:
                    ok = ok[ok[sel_metric].notna()]
                trained = list(ok["Model"]) if len(ok) else []
                if trained:
                    best_idx = ok[sel_metric].idxmax() if higher else ok[sel_metric].idxmin()
                    best = ok.loc[best_idx]
                    best_model, best_sel, best_sec = best["Model"], float(best[sel_metric]), float(best[sec_metric])
                    best_f1 = float(ok["F1 Score"].max()) if "F1 Score" in ok.columns else None
                else:
                    best_model = best_sel = best_sec = best_f1 = None
                train_time = float(sum(cached[m]["wall_s"] for m in trained))
                rows.append({
                    "dataset": ds, "config": rec["config"] if mode != "raw_paired" else "no_fuzzy_paired",
                    "call_idx": rec["call_idx"], "seed": seed, "llm_model": rec["llm_model"],
                    "advisor_branch": rec["advisor_branch"], "history_in_prompt": rec["history_in_prompt"],
                    "n_recommended": len(raw_names),
                    "n_exact_valid": sum(1 for n in raw_names if n in supported),
                    "n_after_matcher": (sum(1 for t in match_types if t not in ("unmatched", "unsupported"))
                                        if match_types else None),
                    "ui_default_fallback": used_default,
                    "n_requested": len(names), "n_trained": len(trained), "n_skipped": len(skipped),
                    "trainer_exception": exc, "runtime_failure": (len(trained) == 0) or (exc is not None),
                    "selected": "|".join(names if all(isinstance(n, str) for n in names) else map(str, names)),
                    "best_model": best_model, "best_" + sel_metric: best_sel, "best_" + sec_metric: best_sec,
                    "best_F1_any": best_f1,
                    "llm_latency_s": rec["latency_s"], "train_time_s": train_time,
                    "search_time_s": train_time + rec["latency_s"],
                })
    return pd.DataFrame(rows)


def brute_force_rows(units_by_ds: Dict[str, Dict[int, dict]]) -> pd.DataFrame:
    rows = []
    for ds, units in units_by_ds.items():
        task = DATASET_META[ds]["task"]
        sel_metric, sec_metric, higher = primary_metrics(task)
        for seed, unit in units.items():
            ok = pd.DataFrame([r for r in unit["brute_force"] if not r.get("Error")])
            best_idx = ok[sel_metric].idxmax() if higher else ok[sel_metric].idxmin()
            best = ok.loc[best_idx]
            rows.append({
                "dataset": ds, "config": "brute_force", "seed": seed,
                "n_requested": len(unit["brute_force"]), "n_trained": len(ok),
                "best_model": best["Model"], "best_" + sel_metric: float(best[sel_metric]),
                "best_" + sec_metric: float(best[sec_metric]),
                "best_F1_any": float(ok["F1 Score"].max()) if "F1 Score" in ok.columns else None,
                "train_time_s": float(sum(r["wall_s"] for r in unit["brute_force"])),
                "search_time_s": float(sum(r["wall_s"] for r in unit["brute_force"])),
                "runtime_failure": False,
            })
    return pd.DataFrame(rows)
