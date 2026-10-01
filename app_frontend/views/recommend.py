"""Step 3 - Recommend: statistics -> meta-learning priors -> retrieval + LLM -> name matching."""
import json

import pandas as pd
import streamlit as st

from app_backend import meta_learning
from app_backend.llm_rag_core import get_llm_model
from app_backend.model_matcher import select_models
from app_backend.model_registry import supported_models
from app_backend.statistical_engine import analyze_dataset
from app_frontend.ui import components as ui
from app_frontend.ui import nav
from app_frontend.ui import state as S
from app_frontend.ui.services import advisor, llm_health

MATCH_LABEL = {"exact": "✓ exact", "alias": "✓ alias", "fuzzy": "≈ fuzzy", "unsupported": "✕ unsupported",
               "unmatched": "✕ unmatched"}


def _run(ws, df, pipe):
    s = S.state()
    cfg = s.get("prepare_config", {})
    with st.status("Getting recommendations...", expanded=True) as status:
        st.write("Computing dataset statistics")
        stats = analyze_dataset(df, pipe.target_col)
        supported = supported_models(pipe.task_type, cfg.get("is_time_series", False))
        st.write("Looking for similar past workspaces (meta-learning)")
        similar = S.wm().find_similar_workspaces(stats, exclude_id=ws.workspace_id)
        st.write(f"Retrieving rules and asking the LLM ({get_llm_model()})")
        out = advisor(get_llm_model()).get_recommendations(stats, supported, similar_workspaces=similar)
        selection, matches, used_default = select_models(out["recommendations"], supported, pipe.task_type)
        status.update(label="Recommendations ready" if out["source"] == "llm" else "Fallback recommendations",
                      state="complete" if out["source"] == "llm" else "error", expanded=False)
    stats_small = {k: v for k, v in stats.items() if k not in ("numerical_columns", "categorical_columns")}
    s.update(stats=stats, advisor=out, selection=selection, matches=[m.to_dict() for m in matches],
             used_default=used_default)
    ws.profile_summary = stats_small | {"profile": stats["profile"]}
    ws.recommendations = selection
    S.record_step("recommend", source=out["source"], model=out.get("model"))
    S.save_state()


def render():
    ws, statuses = nav.require_step("recommend")
    c, s = S.ctx(), S.state()
    st.subheader("Recommend", anchor=False)
    st.caption("The dataset statistics and the most relevant rules from the knowledge base are sent to the LLM, "
               "which proposes models; each proposed name is then matched to a model the app can train.")
    health = llm_health(get_llm_model())
    st.markdown(ui.llm_chip(health), unsafe_allow_html=True)
    if not health.get("ok"):
        st.caption(f"{health.get('error')}. The app will use the default models for this task instead.")
    problem = S.prerequisite_problem("recommend")
    if problem:
        st.markdown(problem)
    if st.button("Get recommendations", type="primary", icon=":material/auto_awesome:", disabled=bool(problem)):
        _run(ws, c["df"], c["pipeline"])
        st.rerun()

    out = s.get("advisor")
    if not out:
        return
    if out["source"] == "llm":
        ui.banner("good", f"LLM {out['model']} answered and the output passed validation "
                          f"(attempt {out.get('attempts', 1)}).", title="✓ LLM")
    else:
        ui.banner("warning", "Fallback used: the LLM was unavailable or its answer was invalid, so default models "
                             f"were selected. Reason: {out.get('error')}", title="! Fallback used")

    st.markdown("#### Proposed models and how they were matched")
    matches = s.get("matches", [])
    reasons = out.get("reasoning") or []
    table = pd.DataFrame([{"LLM name": m["raw"], "Matched to": m["matched"] or "-",
                           "Match": MATCH_LABEL[m["match_type"]],
                           "Score": round(m["score"], 1) if m["match_type"] == "fuzzy" else None,
                           "Note": m["note"], "LLM reasoning": reasons[i] if i < len(reasons) else ""}
                          for i, m in enumerate(matches)])
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={"LLM reasoning": st.column_config.TextColumn(width="large")})
    st.caption("Match types: exact name; alias (scikit-learn/XGBoost class names, abbreviations); fuzzy (typo, "
               "similarity ≥ 85); unsupported (a real model the app does not train, or not for this task); "
               "unmatched (no plausible model). Unsupported and unmatched names are not trained.")
    if s.get("used_default"):
        ui.banner("warning", "None of the proposed names matched a supported model, so the task's default models "
                             f"were selected: {', '.join(s.get('selection', []))}.")
    else:
        ui.chips([ui.chip(f"Selected for training: {', '.join(s.get('selection', []))}", "info")])

    if out.get("history_used"):
        sim = pd.DataFrame(out.get("similar_workspaces", []))
        ui.banner("info", f"Meta-learning: {len(sim)} similar past workspace(s) were added to the prompt as "
                          "historical context.", title="i Meta-learning")
        cols = [c_ for c_ in ("dataset", "best_model", "metric", "best_score", "similarity", "reason") if c_ in sim]
        st.dataframe(sim[cols], width="stretch", hide_index=True,
                     column_config={"reason": st.column_config.TextColumn(width="large")})
    else:
        st.caption(f"Meta-learning: no past workspace of the same task had profile similarity ≥ "
                   f"{meta_learning.SIMILARITY_THRESHOLD:.2f}, so no history was added to the prompt.")

    with st.expander(f"Retrieved rules ({len(out.get('retrieved_rules', []))}) and retrieval query"):
        st.caption(f"Retrieval facts: {out.get('retrieval_facts')}")
        for r in out.get("retrieved_rules", []):
            st.markdown(f"**{r['rule']}**")
            st.code(r["text"], language=None)
    with st.expander("Raw LLM response"):
        st.code(out.get("raw_response") or "(no response)", language=None)
    with st.expander("Dataset statistics sent to the LLM"):
        st.json(json.loads(json.dumps(s.get("stats", {}), default=str)), expanded=False)
    st.divider()
    nav.link("train", "Next: Train & Compare", ":material/arrow_forward:")
