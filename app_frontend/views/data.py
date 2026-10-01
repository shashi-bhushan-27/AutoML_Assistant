"""Step 1 - Data: upload (encoding detection, content-hash keyed), preview, column summary, column removal."""
import hashlib

import pandas as pd
import streamlit as st

from app_backend.preprocessing_engine.ingestion import MAX_UPLOAD_MB, DataIngestor
from app_frontend.ui import charts
from app_frontend.ui import components as ui
from app_frontend.ui import nav
from app_frontend.ui import state as S


def set_dataset(df: pd.DataFrame, name: str, info: dict, upload_hash: str):
    """Store a new dataset version. The saved CSV is read back so the session sees exactly what is on disk."""
    ws, manager = S.current_ws(), S.wm()
    manager.save_dataset(ws.workspace_id, df)
    df = manager.load_dataset(ws.workspace_id)
    fp = S.dataset_fingerprint(df)
    ws.dataset_name, ws.dataset_shape, ws.dataset_hash = name, df.shape, fp
    S.ctx()["df"] = df
    s = S.state()
    s["upload_info"], s["upload_hash"] = info, upload_hash
    S.record_step("data", fp=fp, rows=int(len(df)), columns=int(df.shape[1]))
    S.save_state()


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        rows.append({"Column": c, "Type": str(s.dtype), "Missing %": round(float(s.isna().mean() * 100), 2),
                     "Unique": int(s.nunique()), "Example": str(non_null.iloc[0])[:40] if len(non_null) else ""})
    return pd.DataFrame(rows)


def render():
    ws, statuses = nav.require_step("data")
    c, s = S.ctx(), S.state()
    st.subheader("Data", anchor=False)

    up = st.file_uploader(f"Upload a CSV file (max {MAX_UPLOAD_MB} MB). The encoding is detected automatically.",
                          type=["csv"], key="w_upload")
    if up is not None:
        raw = up.getvalue()
        digest = hashlib.sha256(raw).hexdigest()[:16]
        # keyed on content (not the file name), so re-uploading an edited file with the same name works;
        # "processed_upload" stops the widget's retained file from undoing later column removals
        seen = digest in (s.get("upload_hash"), st.session_state.get("processed_upload"))
        st.session_state["processed_upload"] = digest
        if not seen:
            if len(raw) > MAX_UPLOAD_MB * 1e6:
                ui.banner("critical", f"The file is {len(raw) / 1e6:.0f} MB; the limit is {MAX_UPLOAD_MB} MB.")
            else:
                try:
                    df_new, info = DataIngestor().read_csv_bytes(raw)
                except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                    ui.banner("critical", f"Could not read the CSV: {exc}")
                else:
                    if df_new.empty or df_new.shape[1] < 2:
                        ui.banner("critical", "The file needs at least one feature column and a target column.")
                    else:
                        had_results = bool((ws.steps or {}).get("prepare"))
                        set_dataset(df_new, up.name, info, digest)
                        st.toast(f"Loaded {up.name}: {df_new.shape[0]:,} rows × {df_new.shape[1]} columns")
                        if had_results:
                            st.session_state["data_changed"] = True
                        st.rerun()

    df = c.get("df")
    if df is None:
        ui.banner("info", "No dataset in this workspace yet. Upload a CSV to start.")
        return
    if st.session_state.pop("data_changed", False):
        ui.banner("warning", "The dataset changed, so Prepare and every later step are now marked stale. "
                             "Re-run Prepare to continue.", title="! Stale")

    info = s.get("upload_info") or {}
    dup = int(df.duplicated().sum())
    ui.tiles([("Rows", f"{len(df):,}", None), ("Columns", str(df.shape[1]), None),
              ("Missing cells", f"{df.isna().mean().mean():.1%}", None),
              ("Duplicate rows", f"{dup:,}", "removed before the split in Prepare" if dup else None),
              ("Encoding", str(info.get("encoding", "n/a")), f"{info.get('size_mb', 0):.1f} MB" if info else None)])
    for w in info.get("warnings", []):
        ui.banner("warning", w)
    if info.get("sample"):
        st.caption(f"Sample dataset: {info['sample']}")

    tab_preview, tab_columns, tab_dist = st.tabs(["Preview", "Columns", "Distributions"])
    with tab_preview:
        st.dataframe(df.head(200), width="stretch")
        st.caption(f"First {min(200, len(df))} of {len(df):,} rows.")
    with tab_columns:
        summary = column_summary(df)
        st.dataframe(summary, width="stretch", hide_index=True)
        ui.download_df(summary, "column_summary.csv", key="dl_colsum")
        with st.form("drop_form"):
            drop = st.multiselect("Remove columns (IDs, leaks, free text)", list(df.columns), key="w_drop")
            if st.form_submit_button("Remove selected columns", icon=":material/delete:"):
                if not drop:
                    st.warning("Select at least one column.")
                elif len(drop) >= df.shape[1] - 1:
                    st.error("Keep at least one feature and the target.")
                else:
                    new = df.drop(columns=drop)
                    digest = S.fingerprint("dropped", s.get("upload_hash"), drop)
                    set_dataset(new, ws.dataset_name, info, digest)
                    st.session_state["data_changed"] = bool((ws.steps or {}).get("prepare"))
                    st.rerun()
    with tab_dist:
        num = df.select_dtypes(include="number").columns.tolist()
        cat = [x for x in df.columns if x not in num]
        if num:
            chosen = st.multiselect("Numeric columns", num, default=num[:6], max_selections=9, key="w_hist_cols")
            if chosen:
                fig = charts.histograms(df, chosen)
                stats = df[chosen].describe().T.reset_index().rename(columns={"index": "column"})
                ui.chart(fig, stats.round(4), key="hist", filename="distributions",
                         caption="Row counts per bin; the table view shows summary statistics.")
        if cat:
            col = st.selectbox("Categorical column", cat, key="w_cat_col")
            counts = df[col].astype(str).value_counts().head(15).rename_axis(col).reset_index(name="rows")
            fig = charts.bar(counts, col, "rows", f"Most frequent values of {col} (top 15)", fmt=",d",
                             value_title="rows")
            ui.chart(fig, counts, key="catbar", filename=f"{col}_counts")

    st.divider()
    nav.link("prepare", "Next: Prepare the data", ":material/arrow_forward:")
