"""UI-level checks with Streamlit's AppTest (no browser): the fallback chip/banner (B-4), the match table (B-10),
gating of locked steps, and the Workspaces page."""
from streamlit.testing.v1 import AppTest


def _page_script(view: str):
    """Script body run by AppTest: registers the pages (hidden nav) and renders one view."""
    import streamlit as st

    import app_frontend.views.recommend as recommend
    from app_frontend.ui import nav
    from app_frontend.ui import state as S
    from app_frontend.views import data, deploy, explain, home, prepare, train, tune

    recommend.llm_health = lambda model=None: {"model": "llama-3.3-70b-versatile", "ok": False,
                                               "error": "model 'llama-3.3-70b-versatile' is not served for this key (HTTP 404)"}
    views = {"data": data, "prepare": prepare, "recommend": recommend, "train": train, "explain": explain,
             "tune": tune, "deploy": deploy}
    nav.PAGES = {"home": st.Page(home.render, title="Workspaces", url_path="home", default=True)}
    for name, title in S.STEPS:
        nav.PAGES[name] = st.Page(views[name].render, title=title, url_path=name)
    st.navigation(list(nav.PAGES.values()), position="hidden")
    views[st.session_state["_view"]].render()


def _prepared_ctx(temp_workspace, adult_df, advisor_output=None):
    from app_backend.preprocessing_engine.engine import AutoPreprocessor
    from app_frontend.ui.state import dataset_fingerprint, fingerprint

    wm = temp_workspace.WorkspaceManager()
    ws = wm.create_workspace("adult.csv", adult_df.shape)
    p = AutoPreprocessor(target_col="class", verbose=False)
    out = p.fit_transform(df=adult_df)
    data_fp = dataset_fingerprint(adult_df)
    prep_fp = fingerprint("prep")
    ws.steps = {"data": {"state": "done", "fp": data_fp},
                "prepare": {"state": "done", "fp": prep_fp, "upstream": data_fp}}
    state = {"prepare_config": {"targets": ["class"]}}
    if advisor_output:
        ws.steps["recommend"] = {"state": "done", "fp": "r", "upstream": prep_fp}
        state.update(advisor_output)
    splits = {k: out[k] for k in ("X_train", "X_test", "y_train", "y_test")}
    return {"ws": ws, "df": adult_df, "pipeline": p, "splits": splits, "state": state}


def _run(ctx, view):
    at = AppTest.from_function(_page_script, args=(view,), default_timeout=60)
    at.session_state["ctx"] = ctx
    at.session_state["_view"] = view
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _all_text(at) -> str:
    import html

    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    return html.unescape("\n".join(str(p) for p in parts))


def test_fallback_is_shown_with_chip_and_reason(temp_workspace, adult_df):
    """B-4 acceptance: with a 404 from Groq the page shows the unavailable chip and 'Fallback used'."""
    fallback = {"advisor": {"recommendations": ["XGBoost", "Random Forest"], "reasoning": ["System fallback due to error."] * 2,
                            "source": "fallback", "error": "NotFoundError: Error code: 404 - model does not exist",
                            "model": "llama-3.3-70b-versatile", "retrieved_rules": [], "history_used": False,
                            "similar_workspaces": [], "retrieval_facts": "classification task", "raw_response": None},
                "selection": ["XGBoost", "Random Forest"], "used_default": False,
                "matches": [{"raw": "XGBoost", "matched": "XGBoost", "match_type": "exact", "score": 100.0, "note": ""},
                            {"raw": "LightGBM", "matched": None, "match_type": "unsupported", "score": 100.0,
                             "note": "LightGBM is not trained by this app"}]}
    at = _run(_prepared_ctx(temp_workspace, adult_df, fallback), "recommend")
    text = _all_text(at)
    assert "LLM: llama-3.3-70b-versatile · unavailable" in text
    assert "Fallback used" in text and "404" in text
    table = at.dataframe[0].value
    assert list(table["Match"]) == ["✓ exact", "✕ unsupported"]
    assert "no history was added" in text


def test_locked_step_is_gated(temp_workspace, adult_df):
    ctx = _prepared_ctx(temp_workspace, adult_df)
    at = _run(ctx, "explain")
    text = _all_text(at)
    assert "This step is locked" in text and "Needs Train & Compare" in text
    assert not at.button  # no action is offered on a locked step


def test_stale_banner_after_data_change(temp_workspace, adult_df):
    ctx = _prepared_ctx(temp_workspace, adult_df)
    ctx["ws"].steps["data"]["fp"] = "new-upload"   # the dataset changed after Prepare ran
    at = _run(ctx, "prepare")
    assert "Results on this page are stale" in _all_text(at)


def test_workspaces_page_lists_and_offers_samples(temp_workspace, adult_df):
    _prepared_ctx(temp_workspace, adult_df)  # one workspace in the (temporary) store

    def home_script():
        from app_frontend.views import home

        home.render()

    at = AppTest.from_function(home_script, default_timeout=60)
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "New workspace" in labels and "Load sample" in labels and "Open" in labels
