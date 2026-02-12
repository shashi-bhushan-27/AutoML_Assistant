import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import pickle

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_backend.statistical_engine import analyze_dataset
from app_backend.llm_rag_core import ModelAdvisor
from app_backend.model_trainer import ModelTrainer
from app_backend.model_tuner import ModelTuner
from app_backend.preprocessing_engine.engine import AutoPreprocessor
from app_backend.workspace_manager import WorkspaceManager, Workspace
from app_backend.workspace_manager import WorkspaceManager, Workspace
from app_backend.report_generator import ReportGenerator
from app_backend.code_generator import generate_training_code

# --- SET CONFIG ---
st.set_page_config(
    page_title="AutoML Assistant", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD CUSTOM CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try loading CSS, handle path if needed
css_path = os.path.join(os.path.dirname(__file__), 'assets/style.css')
if os.path.exists(css_path):
    local_css(css_path)

# --- HERO SECTION ---
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">AutoML <span style="color: #ec4899">Assistant</span></h1>
        <p style="font-size: 1.2rem; color: #94a3b8; max-width: 600px; margin: 0 auto;">
            Automate your data science workflow with AI. Upload, Clean, Analyze, and Train in minutes.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'stats' not in st.session_state: st.session_state.stats = None
if 'recommendations' not in st.session_state: st.session_state.recommendations = []
if 'trainer' not in st.session_state: st.session_state.trainer = None
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'df_clean' not in st.session_state: st.session_state.df_clean = None
if 'preprocessor' not in st.session_state: st.session_state.preprocessor = None
if 'preprocess_report' not in st.session_state: st.session_state.preprocess_report = None
if 'current_workspace' not in st.session_state: st.session_state.current_workspace = None
if 'workspace_manager' not in st.session_state: st.session_state.workspace_manager = WorkspaceManager()
if 'view_mode' not in st.session_state: st.session_state.view_mode = "home"  # "home" or "workspace"

# === WORKSPACE-FIRST UI FLOW ===
wm = st.session_state.workspace_manager
workspaces = wm.list_workspaces()

# Show workspace selector if no active workspace
if st.session_state.current_workspace is None:
    st.markdown("### 📁 Select or Create a Workspace")
    
    col_create, col_spacer = st.columns([1, 3])
    with col_create:
        if st.button("➕ Create New Workspace", type="primary", use_container_width=True):
            # 1. Clear session state to prevent leakage
            st.session_state.df_clean = None
            st.session_state.stats = None
            st.session_state.recommendations = []
            st.session_state.trainer = None
            st.session_state.results_df = None
            st.session_state.preprocessor = None
            st.session_state.preprocess_report = None
            st.session_state.X_train = None
            st.session_state.X_test = None
            
            # Reset file uploader tracker
            if 'last_uploaded_file' in st.session_state:
                del st.session_state['last_uploaded_file']
            
            # 2. Create and assign new workspace
            new_ws = wm.create_workspace(dataset_name="New Analysis", dataset_shape=(0, 0))
            st.session_state.current_workspace = new_ws
            st.rerun()
    
    if workspaces:
        st.markdown("#### Previous Sessions")
        cols = st.columns(3)
        for idx, ws_data in enumerate(workspaces):
            with cols[idx % 3]:
                status_icon = "🟢" if ws_data['status'] == "completed" else "🟡"
                st.markdown(f"""
                <div style="border: 1px solid #444; border-radius: 10px; padding: 12px; margin-bottom: 8px; background: rgba(50,50,60,0.5);">
                    <h5 style="margin: 0;">{status_icon} {ws_data['dataset_name']}</h5>
                    <p style="font-size: 0.8rem; color: #888;">Task: {(ws_data['task_type'] or 'Unknown').title()}</p>
                    <p style="font-size: 0.8rem;">Best: {ws_data['best_model'] or 'N/A'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("📂 Open", key=f"open_{ws_data['workspace_id']}", use_container_width=True):
                        loaded_ws = wm.load_workspace(ws_data['workspace_id'])
                        st.session_state.current_workspace = loaded_ws
                        
                        # Restore saved state
                        saved_df = wm.load_dataset(ws_data['workspace_id'])
                        if saved_df is not None:
                            st.session_state.df_clean = saved_df
                        
                        saved_state = wm.load_session_state(ws_data['workspace_id'])
                        if saved_state:
                            st.session_state.stats = saved_state.get('stats')
                            st.session_state.recommendations = saved_state.get('recommendations', [])
                            st.session_state.preprocess_report = saved_state.get('preprocess_report')
                        
                        st.rerun()
                
                with btn_col2:
                    if st.button("🗑️", key=f"del_{ws_data['workspace_id']}", use_container_width=True, help="Delete workspace"):
                        wm.delete_workspace(ws_data['workspace_id'])
                        st.toast(f"Deleted workspace: {ws_data['dataset_name']}")
                        st.rerun()
    else:
        st.info("No workspaces yet. Create one to get started!")
    
    st.stop()  # Don't show tabs unless workspace is selected

# Active workspace header
ws = st.session_state.current_workspace
col_back, col_title, col_status = st.columns([1, 4, 1])
with col_back:
    if st.button("← All Workspaces"):
        st.session_state.current_workspace = None
        st.rerun()
with col_title:
    st.markdown(f"### 🔬 {ws.dataset_name}")
with col_status:
    st.caption(f"Status: {ws.status}")

st.divider()

# --- WORKFLOW TABS ---
tab1, tab2, tab2b, tab3, tab4, tab_docs = st.tabs(["📂 1. Upload", "🔧 2. Preprocessing", "🔍 3. Analysis", "🤖 4. Training", "🚀 5. Optimization", "📖 Docs & Help"])

# ==========================
# TAB 1: UPLOAD & CLEAN
# ==========================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Data Import", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        # Prevent infinite rerun loop
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            try:
                df_original = pd.read_csv(uploaded_file, on_bad_lines='skip')
                
                # Update workspace immediately
                ws = st.session_state.current_workspace
                if ws:
                    ws.dataset_name = uploaded_file.name
                    ws.dataset_shape = df_original.shape
                    ws.add_event("dataset_uploaded", f"Uploaded {uploaded_file.name}")
                    wm.save_workspace(ws)
                    wm.save_dataset(ws.workspace_id, df_original)
                
                st.session_state.df_clean = df_original
                st.session_state.last_uploaded_file = uploaded_file.name
                st.toast("Dataset uploaded and saved successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error parsing CSV: {e}")
    
    # Show status if using stored data
    elif st.session_state.df_clean is not None and st.session_state.current_workspace:
         st.success(f"✅ Using stored dataset: **{st.session_state.current_workspace.dataset_name}**")
         st.caption("Upload a new file above to overwrite.")

    # Display Current Dataset (from upload or loaded workspace)
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        with col2:
            st.markdown(f"### Current Data: {st.session_state.current_workspace.dataset_name if st.session_state.current_workspace else 'Loaded'}")
            st.dataframe(df.head(5), use_container_width=True)
            
            st.divider()
            
            cols_to_drop = st.multiselect("Select columns to remove:", df.columns)
            
            if cols_to_drop:
                if st.button("Apply Column Removal"):
                    new_df = df.drop(columns=cols_to_drop)
                    st.session_state.df_clean = new_df
                    
                    if st.session_state.current_workspace:
                         ws = st.session_state.current_workspace
                         wm.save_dataset(ws.workspace_id, new_df)
                         ws.dataset_shape = new_df.shape
                         wm.save_workspace(ws)
                    
                    st.success(f"Removed {len(cols_to_drop)} columns.")
                    st.rerun()
    else:
        with col2:
            st.info("👆 Upload a CSV file to get started.")

# ==========================
# TAB 2: PREPROCESSING
# ==========================
with tab2:
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        st.markdown("### 🔧 Advanced Preprocessing Engine", unsafe_allow_html=True)
        st.write("Automated preprocessing with full explainability.")
        
        col_p1, col_p2 = st.columns([1, 2])
        
        with col_p1:
            st.markdown("#### Configuration")
            target_col = st.selectbox("Select Target Variable", df.columns, key="preprocess_target")
            
            is_timeseries = st.checkbox("Time Series Data?", value=False)
            date_col = None
            if is_timeseries:
                date_col = st.selectbox("Date Column", df.columns)
            
            apply_smote = st.checkbox("Apply SMOTE for imbalance?", value=False)
            
            if st.button("🚀 Run Preprocessing", type="primary"):
                with st.spinner("Running preprocessing pipeline..."):
                    preprocessor = AutoPreprocessor(
                        target_col=target_col,
                        task_type="auto",
                        is_time_series=is_timeseries,
                        date_col=date_col,
                        apply_smote=apply_smote,
                        verbose=True
                    )
                    
                    result = preprocessor.fit_transform(df=df)
                    st.session_state.preprocessor = preprocessor
                    st.session_state.preprocess_report = preprocessor.get_report()
                    
                    # Store for training
                    st.session_state.X_train = result["X_train"]
                    st.session_state.X_test = result["X_test"]
                    st.session_state.y_train = result["y_train"]
                    st.session_state.y_test = result["y_test"]
                    
                    # Update workspace
                    if st.session_state.current_workspace:
                        ws = st.session_state.current_workspace
                        ws.target_col = target_col
                        ws.task_type = preprocessor.task_type
                        ws.preprocessing_steps = preprocessor.full_log
                        ws.add_event("preprocessing_complete", f"Applied {len(preprocessor.full_log)} preprocessing steps")
                        st.session_state.workspace_manager.save_workspace(ws)
                        
                        # Save session state for restoration
                        st.session_state.workspace_manager.save_session_state(ws.workspace_id, {
                            'stats': st.session_state.stats,
                            'recommendations': st.session_state.recommendations,
                            'preprocess_report': st.session_state.preprocess_report
                        })
                    
                    st.success("✅ Preprocessing Complete!")
        
        with col_p2:
            if st.session_state.preprocess_report:
                report = st.session_state.preprocess_report
                
                st.markdown("#### 📊 Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Task Type", report['summary']['task_type'].title())
                m2.metric("Features", report['summary']['features_final'])
                m3.metric("Train Samples", report['summary']['train_samples'])
                m4.metric("Test Samples", report['summary']['test_samples'])
                
                st.divider()
                
                col_applied, col_skipped = st.columns(2)
                
                with col_applied:
                    st.markdown("#### ✅ Applied Steps")
                    for step in report['applied_steps'][:10]:  # Limit display
                        st.markdown(f"**{step['step']}**: {step['action']}")
                        st.caption(f"Reason: {step['reason']}")
                
                with col_skipped:
                    st.markdown("#### ⏭️ Skipped Steps")
                    for step in report['skipped_steps'][:10]:
                        st.markdown(f"**{step['step']}**: {step['action']}")
                        st.caption(f"Reason: {step['reason']}")
                
                with st.expander("📋 Full Preprocessing Report"):
                    st.json(report)
                
                # Export options
                st.divider()
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    if st.button("💾 Export Pipeline"):
                        try:
                            pipeline_path = "preprocessor_pipeline.pkl"
                            st.session_state.preprocessor.save(pipeline_path)
                            st.success(f"Saved to {pipeline_path}")
                        except Exception as e:
                            st.error(f"Export failed: {e}")
    else:
        st.info("Please upload data in Step 1 first.")

# ==========================
# TAB 3: ANALYSIS
# ==========================
with tab2b:
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        col_c1, col_c2 = st.columns([1, 3])
        
        with col_c1:
            st.markdown("### Configuration", unsafe_allow_html=True)
            if not df.empty:
                target_col = st.selectbox("Select Target Variable", df.columns)
                
                if st.button("🔍 Analyze Dataset", type="primary"):
                    with st.spinner("Consulting AI Brain..."):
                        # 1. Statistical Analysis
                        stats = analyze_dataset(df, target_col=target_col)
                        st.session_state.stats = stats
                        
                        # 2. Initialize Trainer
                        temp_trainer = ModelTrainer(df, target_col, stats['task_type'], stats['is_time_series'], stats.get('time_column'))
                        st.session_state.trainer = temp_trainer 
                        
                        # 3. Get AI Recommendations
                        @st.cache_resource
                        def get_advisor():
                            return ModelAdvisor()
                        
                        advisor = get_advisor()
                        supported_models = temp_trainer.get_supported_models()
                        
                        # Meta-Learning: Find similar past workspaces
                        wm = st.session_state.workspace_manager
                        
                        # Fix for Stale Object Reference (Hot-Reload)
                        if not hasattr(wm, 'find_similar_workspaces'):
                            st.warning("🔄 Refreshing Workspace Manager...")
                            wm = WorkspaceManager() # Re-instantiate to get new methods
                            st.session_state.workspace_manager = wm
                            
                        similar_ws = wm.find_similar_workspaces(stats)
                        st.session_state.similar_workspaces = similar_ws # Store for UI
                        
                        raw_recs = advisor.get_recommendations(stats, supported_models, similar_workspaces=similar_ws)
                        
                        # Handle new JSON structure vs legacy list
                        if isinstance(raw_recs, dict):
                            st.session_state.recommendations = raw_recs.get("recommendations", [])
                            st.session_state.reasoning = raw_recs.get("reasoning", [])
                        else:
                            st.session_state.recommendations = raw_recs
                            st.session_state.reasoning = []
                        
                        # Reset model selector state to apply new recommendations
                        if "model_selector" in st.session_state:
                            del st.session_state["model_selector"]
                        
                        # 4. Update workspace with user-selected target
                        if st.session_state.current_workspace:
                            ws = st.session_state.current_workspace
                            ws.target_col = target_col
                            ws.task_type = stats['task_type']
                            ws.profile_summary = stats
                            ws.add_event("analysis_complete", f"Target: {target_col}, Task: {stats['task_type']}")
                            st.session_state.workspace_manager.save_workspace(ws)
            else:
                st.error("Dataset is empty.")
        
        with col_c2:
            if st.session_state.stats:
                st.markdown("### 📊 Data Insights", unsafe_allow_html=True)
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows", st.session_state.stats.get('rows', 0))
                m2.metric("Features", st.session_state.stats.get('columns', 0))
                m3.metric("Task Type", st.session_state.stats.get('task_type', 'Unknown'))
                
                # Meta-Learning Badge
                if st.session_state.get('similar_workspaces'):
                    count = len(st.session_state.similar_workspaces)
                    st.info(f"💡 **Meta-Learning Active**: Recommendations biased by {count} similar past experiment(s).")
                m2.metric("Features", st.session_state.stats.get('columns', 0))
                m3.metric("Task Type", st.session_state.stats.get('task_type', 'Unknown'))
                
                st.markdown("#### AI Recommendations")
                st.success(f"**Strategy:** {', '.join(st.session_state.recommendations)}")
                
                # Upgrade 3: Explainability Layer
                if st.session_state.get('reasoning'):
                    with st.expander("🧠 Why This Model?", expanded=True):
                        st.caption("The AI Advisor selected these models because:")
                        for r in st.session_state.reasoning:
                            st.markdown(f"- {r}")
                
                # Preprocessing Report
                if st.session_state.trainer:
                    st.markdown("#### 🔧 Preprocessing Summary")
                    preprocess_summary = st.session_state.trainer.get_preprocessing_summary()
                    st.info(preprocess_summary)
                    
                    with st.expander("View Detailed Transformations"):
                        report = st.session_state.trainer.get_preprocessing_report()
                        if report:
                            st.json(report)
                
                with st.expander("View Detailed Statistics JSON"):
                    st.json(st.session_state.stats)
    else:
        st.info("Please upload data in Step 1 first.")

# ==========================
# TAB 3: MODEL TRAINING
# ==========================
with tab3:
    if st.session_state.stats is not None and st.session_state.trainer is not None:
        st.markdown("### 🛠️ Model Selection", unsafe_allow_html=True)
        
        all_supported_models = st.session_state.trainer.get_supported_models()
        
        # Debug: Show what AI recommended
        with st.expander("Debug: AI Recommendations"):
            st.write(f"Raw recommendations: {st.session_state.recommendations}")
            st.write(f"Supported models: {all_supported_models}")
        
        # Default selection: Intersection of AI Recs and Supported Models
        # Fuzzy matching implementation
        default_selection = []
        normalized_supported = {m.lower().replace(" ", ""): m for m in all_supported_models}
        
        for rec in st.session_state.recommendations:
            rec_clean = str(rec).lower().replace(" ", "")
            # Direct match
            if rec in all_supported_models:
                default_selection.append(rec)
            # Fuzzy match
            elif rec_clean in normalized_supported:
                default_selection.append(normalized_supported[rec_clean])
            # Partial match (e.g. "Random Forest Classifier" -> "Random Forest")
            else:
                for clean_key, valid_name in normalized_supported.items():
                    if clean_key in rec_clean or rec_clean in clean_key:
                        default_selection.append(valid_name)
                        break
        
        # Deduplicate
        default_selection = list(set(default_selection))
        
        # If no match, use sensible defaults based on task type
        if not default_selection:
            task = st.session_state.stats.get('task_type', 'Regression')
            if task == "Regression":
                default_selection = ["XGBoost", "Random Forest", "Gradient Boosting"]
            else:
                default_selection = ["XGBoost", "Random Forest", "Logistic Regression"]
            default_selection = [m for m in default_selection if m in all_supported_models]
        
        selected_models = st.multiselect(
            "Select models to train:",
            options=all_supported_models,
            default=default_selection,
            key="model_selector"
        )
        
        if st.button("🚀 Start Training", type="primary"):
            if not selected_models:
                st.error("Select at least one model.")
            else:
                with st.status("Training models...", expanded=True) as status:
                    trainer = st.session_state.trainer
                    
                    # Use preprocessed data if available (from Tab 2)
                    if st.session_state.get('X_train') is not None:
                        trainer.set_preprocessed_data(
                            st.session_state.X_train,
                            st.session_state.X_test,
                            st.session_state.y_train,
                            st.session_state.y_test
                        )
                    
                    results_df = trainer.run_selected_models(selected_models)
                    st.session_state.results_df = results_df
                    
                    # Update workspace with training results
                    if st.session_state.current_workspace and not results_df.empty:
                        ws = st.session_state.current_workspace
                        ws.recommendations = selected_models
                        ws.model_results = results_df.to_dict()
                        
                        # Find best model
                        sort_col = "RMSE" if ws.task_type == "regression" else "Accuracy"
                        if sort_col in results_df.columns:
                            if sort_col == "RMSE":
                                best_idx = results_df[sort_col].idxmin()
                            else:
                                best_idx = results_df[sort_col].idxmax()
                            ws.best_model = results_df.loc[best_idx, "Model"]
                            ws.best_score = float(results_df.loc[best_idx, sort_col])
                        
                        ws.status = "completed"
                        ws.add_event("training_complete", f"Trained {len(selected_models)} models. Best: {ws.best_model}")
                        st.session_state.workspace_manager.save_workspace(ws)
                    
                    status.update(label="✅ Training Complete!", state="complete", expanded=False)
        
        # LEADERBOARD
        if st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            
            st.divider()
            st.markdown("### 🏆 Leaderboard", unsafe_allow_html=True)
            
            # Debug: Show raw results
            with st.expander("Debug: Raw Results"):
                st.dataframe(results_df)
            
            # Check if we have any successful results
            if "Error" in results_df.columns:
                # Filter out rows with errors
                success_df = results_df[results_df["Error"].isna() | (results_df["Error"] == "")]
                error_df = results_df[results_df["Error"].notna() & (results_df["Error"] != "")]
                
                if not error_df.empty:
                    st.warning(f"⚠️ {len(error_df)} model(s) failed to train. Check Debug for details.")
            else:
                success_df = results_df
            
            if not success_df.empty:
                sort_metric = "RMSE" if st.session_state.stats['task_type'] == "Regression" else "Accuracy"
                ascending = True if sort_metric == "RMSE" else False
                
                if sort_metric in success_df.columns:
                    results_df = results_df.sort_values(by=sort_metric, ascending=ascending)
                    
                    col_l1, col_l2 = st.columns([2, 1])
                    
                    with col_l1:
                         # Plot
                        fig = px.bar(
                            results_df, 
                            x="Model", 
                            y=sort_metric, 
                            color="Model", 
                            title=f"Model Performance ({sort_metric})", 
                            text_auto=True,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_l2:
                         # Table
                        if sort_metric == "RMSE":
                            styled_df = results_df.style.highlight_min(axis=0, color='#10b981', subset=[sort_metric]) # Green for low error
                        else:
                            styled_df = results_df.style.highlight_max(axis=0, color='#10b981', subset=[sort_metric]) # Green for high acc
                        
                        st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(results_df)
            else:
                st.warning("No successful results found.")
    else:
        st.info("Complete analysis in Step 2 first.")

# ==========================
# TAB 4: OPTIMIZATION
# ==========================
with tab4:
    if st.session_state.results_df is not None:
        st.markdown("### ⚡ Advanced Hyperparameter Tuning", unsafe_allow_html=True)
        st.write("Fine-tune your model with custom settings.")
        
        results_df = st.session_state.results_df
        valid_models = results_df['Model'].tolist() if 'Model' in results_df.columns else []
        
        if valid_models:
            col_opt_1, col_opt_2 = st.columns([1, 1])
            
            with col_opt_1:
                model_to_tune = st.selectbox("Select model to optimize:", valid_models)
                
                st.markdown("#### Tuning Settings")
                time_budget = st.number_input("Time Budget (seconds)", min_value=10, max_value=600, value=30, 
                                   help="Max time to spend exploring hyperparameters.")
                cv_folds = st.slider("Cross-Validation Folds", min_value=2, max_value=10, value=3,
                                     help="Number of folds for cross-validation.")
            
            with col_opt_2:
                st.markdown("#### Parameter Customization")
                with st.expander("🔧 Advanced: Edit Parameter Ranges", expanded=False):
                    st.caption("Leave blank to use defaults. Format: comma-separated values.")
                    
                    custom_params = None
                    model_lower = model_to_tune.lower()
                    
                    if "xgboost" in model_lower or "random forest" in model_lower or "gradient" in model_lower:
                        max_depth_input = st.text_input("max_depth (e.g., 3,5,7,10)", "")
                        n_est_input = st.text_input("n_estimators (e.g., 100,200,300)", "")
                        
                        if max_depth_input or n_est_input:
                            custom_params = {}
                            if max_depth_input:
                                try:
                                    custom_params['max_depth'] = [int(x.strip()) for x in max_depth_input.split(',')]
                                except: pass
                            if n_est_input:
                                try:
                                    custom_params['n_estimators'] = [int(x.strip()) for x in n_est_input.split(',')]
                                except: pass
                                
                    elif "ridge" in model_lower or "lasso" in model_lower:
                        alpha_input = st.text_input("alpha (e.g., 0.1,1.0,10.0)", "")
                        if alpha_input:
                            try:
                                custom_params = {'alpha': [float(x.strip()) for x in alpha_input.split(',')]}
                            except: pass
                            
                    elif "svm" in model_lower:
                        c_input = st.text_input("C (e.g., 0.1,1,10)", "")
                        if c_input:
                            try:
                                custom_params = {'C': [float(x.strip()) for x in c_input.split(',')]}
                            except: pass
                    else:
                        st.info("Default parameters will be used for this model.")
            
            st.divider()
            
            if st.button(f"✨ Optimize {model_to_tune}", type="primary"):
                with st.spinner(f"Auto-tuning {model_to_tune} using Bayesian Optimization (Optuna) for {time_budget}s..."):
                    tuner = ModelTuner(st.session_state.trainer)
                    tuned_res = tuner.tune_model(model_to_tune, time_budget=time_budget, cv_folds=cv_folds, custom_params=custom_params)
                    
                    if "Error" in tuned_res:
                        st.error(tuned_res["Error"])
                    else:
                        st.success("🎉 Optimization Complete!")
                        st.json(tuned_res)
                        
                        # Export
                        tuned_name = f"{model_to_tune} (Tuned)"
                        model_obj = st.session_state.trainer.trained_models.get(tuned_name)
                        
                        if model_obj:
                            try:
                                pickle_out = pickle.dumps(model_obj)
                                st.download_button(
                                    label="💾 Download Tuned Model",
                                    data=pickle_out,
                                    file_name=f"{model_to_tune.replace(' ','_')}_tuned.pkl",
                                    mime="application/octet-stream"
                                )
                                
                                # Export Python Code
                                st.divider()
                                with st.expander("📜 View/Copy Training Code", expanded=True):
                                    st.write("run this code in your environment (e.g. Colab) to train this model.")
                                    
                                    # Get params from result
                                    best_params = tuned_res.get("Best Params", {})
                                    
                                    code = generate_training_code(
                                        dataset_name=f"{ws.dataset_name}",
                                        target_col=ws.target_col,
                                        model_name=model_to_tune,
                                        best_params=best_params,
                                        task_type=ws.task_type
                                    )
                                    st.code(code, language='python')
                                    
                            except Exception as e:
                                st.warning(f"Serialization failed: {e}")
        else:
            st.warning("No valid models to tune.")
    else:
        st.info("Train a model in Step 3 first.")

# ==========================
# TAB 6: DOCS & HELP
# ==========================
with tab_docs:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <h2 style="font-size: 2.2rem; margin-bottom: 0.3rem;">📖 Documentation & <span style="color: #ec4899">Help Center</span></h2>
        <p style="font-size: 1rem; color: #94a3b8; max-width: 600px; margin: 0 auto;">
            Everything you need to know to get the most out of AutoML Assistant.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---- SECTION 1: GETTING STARTED ----
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    AutoML Assistant guides you through a **5-step workflow** to go from raw data to a trained, optimized machine learning model:

    | Step | Tab | What Happens |
    |------|-----|-------------|
    | **1** | 📂 Upload | Import your CSV dataset and optionally remove unwanted columns |
    | **2** | 🔧 Preprocessing | Automatically clean, encode, and scale your data |
    | **3** | 🔍 Analysis | AI analyzes your data and recommends the best models |
    | **4** | 🤖 Training | Train multiple ML models and compare results on a leaderboard |
    | **5** | 🚀 Optimization | Fine-tune the best model with Bayesian hyperparameter search |

    > **Tip:** Start by creating a **Workspace** — it saves your entire session so you can come back anytime.
    """)

    st.divider()

    # ---- SECTION 2: STEP-BY-STEP GUIDE ----
    st.markdown("### 📂 Step-by-Step Guide")

    with st.expander("📂 Step 1 — Upload & Clean", expanded=False):
        st.markdown("""
        **Goal:** Import your dataset into the platform.

        1. Click **"➕ Create New Workspace"** on the home screen.
        2. In the **Upload** tab, drag-and-drop or browse for a **CSV file**.
        3. A preview of the first 5 rows will appear on the right.
        4. *(Optional)* Use the **"Select columns to remove"** dropdown to drop irrelevant columns (e.g., IDs, names).
        5. Click **"Apply Column Removal"** to confirm.

        **What file formats are supported?**
        Currently, only **`.csv`** files are supported. Make sure your CSV uses commas as delimiters.

        **How large can my file be?**
        There is no hard limit, but files over **200 MB** may cause slow performance. For very large datasets, consider sampling your data before uploading.
        """)

    with st.expander("🔧 Step 2 — Preprocessing", expanded=False):
        st.markdown("""
        **Goal:** Automatically clean and prepare your data for ML training.

        1. Select your **Target Variable** (the column you want to predict).
        2. *(Optional)* Toggle **"Time Series Data?"** if your data is temporal and select the date column.
        3. *(Optional)* Toggle **"Apply SMOTE"** if you have an imbalanced classification dataset.
        4. Click **"🚀 Run Preprocessing"**.

        The engine will automatically:
        - Detect column types (numerical, categorical, datetime)
        - Impute missing values
        - Remove or cap outliers
        - Encode categorical features (One-Hot or Label Encoding)
        - Scale numerical features (Standard or MinMax Scaling)
        - Split data into Train/Test sets (80/20 by default)

        **Understanding the Report:**
        - **✅ Applied Steps** — Transformations that were applied and why.
        - **⏭️ Skipped Steps** — Steps that were not needed (e.g., "No missing values found").
        - You can expand **"📋 Full Preprocessing Report"** for the complete JSON details.

        **Exporting the Pipeline:**
        Click **"💾 Export Pipeline"** to save the fitted preprocessor as a `.pkl` file for reuse.
        """)

    with st.expander("🔍 Step 3 — Analysis", expanded=False):
        st.markdown("""
        **Goal:** Let the AI analyze your dataset and recommend the best models.

        1. Select your **Target Variable**.
        2. Click **"🔍 Analyze Dataset"**.
        3. The system will:
           - Generate statistical profiles (correlation, skewness, class balance, etc.)
           - Send the profile to the **Groq AI (Llama 3.1)** via a RAG pipeline
           - Return **3-5 recommended models** with reasoning

        **Key UI Elements:**
        - **Data Insights** — Row count, feature count, and detected task type (Regression/Classification).
        - **AI Recommendations** — The models the AI suggests, auto-selected for training.
        - **"🧠 Why This Model?"** — Expandable panel explaining the AI's reasoning.
        - **"💡 Meta-Learning Active"** badge appears when the system uses results from your past experiments to improve suggestions.

        > **Note:** This step requires a valid **GROQ_API_KEY** in your `.env` file. If AI fails, the system falls back to safe defaults (XGBoost, Random Forest).
        """)

    with st.expander("🤖 Step 4 — Training", expanded=False):
        st.markdown("""
        **Goal:** Train one or more ML models and compare their performance.

        1. The AI-recommended models are **pre-selected** in the multi-select dropdown.
        2. You can add or remove models from the list.
        3. Click **"🚀 Start Training"**.
        4. A live progress indicator shows training status.

        **After Training:**
        - A **🏆 Leaderboard** ranks all models by performance:
          - **Regression** → Sorted by RMSE (lower is better)
          - **Classification** → Sorted by Accuracy (higher is better)
        - An interactive **bar chart** visualizes the comparison.
        - The **best model** is highlighted in green.

        **What if a model fails?**
        Failed models appear with a warning. Check the **"Debug: Raw Results"** expander for error details. The leaderboard still shows successful models.
        """)

    with st.expander("🚀 Step 5 — Optimization", expanded=False):
        st.markdown("""
        **Goal:** Squeeze extra performance from your best model via hyperparameter tuning.

        1. Select the model to optimize from the dropdown.
        2. Set a **Time Budget** (default: 30 seconds). More time = more exploration.
        3. Set **Cross-Validation Folds** (default: 3).
        4. *(Optional)* Expand **"🔧 Advanced: Edit Parameter Ranges"** to customize hyperparameter search spaces.
        5. Click **"✨ Optimize [Model]"**.

        The system uses **Optuna** (Bayesian optimization) to intelligently search the parameter space, pruning bad trials early.

        **After Optimization:**
        - The best parameters and score are displayed.
        - Click **"💾 Download Tuned Model"** to export as a `.pkl` file.
        - Expand **"📜 View/Copy Training Code"** to get a ready-to-run Python script for reproducing the result (e.g., in Google Colab).
        """)

    st.divider()

    # ---- SECTION 3: COMMON ERRORS & TROUBLESHOOTING ----
    st.markdown("### ⚠️ Common Errors & Troubleshooting")

    errors_data = [
        {
            "error": "❌ Error parsing CSV",
            "cause": "Your CSV file is malformed, uses a non-standard delimiter (e.g., semicolons), or has encoding issues.",
            "fix": "Open the file in Excel or Google Sheets, then re-export as a standard UTF-8 CSV with commas."
        },
        {
            "error": "'Please upload data in Step 1 first'",
            "cause": "You jumped to Preprocessing, Analysis, or Training before uploading a dataset.",
            "fix": "Go back to the **📂 Upload** tab and upload a CSV file first."
        },
        {
            "error": "AI Analysis hangs or returns fallback models",
            "cause": "The Groq API key is missing, invalid, or the API is temporarily unavailable.",
            "fix": "1. Check your `.env` file has a valid `GROQ_API_KEY`.\n2. Test the key by running `python test_groq_connection.py`.\n3. The system still works with fallback models (XGBoost, Random Forest) even if the API fails."
        },
        {
            "error": "Vector Store not found!",
            "cause": "The FAISS vector store (used by the AI advisor) has not been built yet.",
            "fix": "Run this command once: `python app_backend/llm_rag_core.py` — this builds the knowledge base index."
        },
        {
            "error": "SMOTE error / Too few samples",
            "cause": "SMOTE requires at least 2 samples per class in the minority class, and only works for classification tasks.",
            "fix": "1. Ensure your target variable is categorical (not continuous).\n2. Ensure each class has at least 2 samples.\n3. If your dataset is very small (< 50 rows), disable SMOTE."
        },
        {
            "error": "Model training fails / Convergence warning",
            "cause": "Some models (e.g., SVM, Logistic Regression) fail to converge on certain data shapes or when features are not scaled.",
            "fix": "1. Run **Preprocessing** (Step 2) before training — it handles scaling automatically.\n2. Try tree-based models (XGBoost, Random Forest) which are more robust.\n3. Remove highly correlated or constant columns."
        },
        {
            "error": "KeyError or missing column after preprocessing",
            "cause": "The target column was dropped or renamed during preprocessing.",
            "fix": "Do NOT remove your target column in Step 1. Make sure the target variable name matches exactly in all steps."
        },
        {
            "error": "Workspace data missing after reload",
            "cause": "The Streamlit file uploader resets on page reload, but your data is actually saved.",
            "fix": "Look for the green **'✅ Using stored dataset'** banner. Your data is safe — no need to re-upload."
        },
        {
            "error": "Serialization / Pickle error on download",
            "cause": "Some model objects cannot be serialized (rare edge case with custom estimators).",
            "fix": "Try a different model. Standard scikit-learn and XGBoost models export without issues."
        },
        {
            "error": "Prophet / ARIMA / SARIMAX fails",
            "cause": "Time-series models require a valid datetime column and properly ordered data.",
            "fix": "1. Toggle **'Time Series Data?'** in preprocessing.\n2. Ensure your date column is in a parseable format (e.g., YYYY-MM-DD).\n3. Sort your data by date before uploading."
        },
    ]

    for item in errors_data:
        with st.expander(f"🔴 {item['error']}"):
            st.markdown(f"**Likely Cause:** {item['cause']}")
            st.markdown(f"**How to Fix:** {item['fix']}")

    st.divider()

    # ---- SECTION 4: TIPS & BEST PRACTICES ----
    st.markdown("### 💡 Tips & Best Practices")

    col_tips1, col_tips2 = st.columns(2)

    with col_tips1:
        st.markdown("""
        #### Data Preparation
        - **Clean your CSV** before uploading — remove blank rows and fix typos.
        - **Drop ID/index columns** in Step 1 — they add noise and confuse models.
        - **Aim for 100+ rows** for meaningful results. Very small datasets (< 30 rows) will produce unreliable models.
        - **Check class balance** for classification — if one class has 95% of the data, enable SMOTE or collect more data.
        """)

    with col_tips2:
        st.markdown("""
        #### Model Selection
        - **Start with the AI recommendations** — they are tailored to your specific data profile.
        - **XGBoost and Random Forest** are reliable general-purpose choices.
        - **Linear/Logistic Regression** work best when features have linear relationships with the target.
        - **SVM** can be slow on large datasets (> 10k rows) — use with caution.
        - **Time-Series?** Use Prophet for seasonality-heavy data, ARIMA for stationary data.
        """)

    st.markdown("""
    #### Optimization Tips
    - **Give more time** during optimization (60-120 seconds) for complex models like XGBoost.
    - **Increase CV folds** (5-10) for small datasets to get a more reliable score.
    - **Customizing parameters** is optional — Optuna's defaults are already smart.
    - After optimization, always **download the tuned model** and the **training code** for reproducibility.
    """)

    st.divider()

    # ---- SECTION 5: SUPPORTED MODELS ----
    st.markdown("### 📋 Supported Models")

    col_models1, col_models2 = st.columns(2)

    with col_models1:
        st.markdown("""
        #### Standard Models
        | Model | Task | Best For |
        |-------|------|----------|
        | Linear Regression | Regression | Simple linear relationships |
        | Ridge Regression | Regression | Regularized linear regression |
        | Lasso Regression | Regression | Feature selection via L1 penalty |
        | Logistic Regression | Classification | Binary/multi-class classification |
        | Random Forest | Both | Versatile, handles mixed features |
        | XGBoost | Both | High performance on tabular data |
        | Gradient Boosting | Both | Strong ensemble method |
        | SVM | Both | High-dimensional data |
        | KNN | Both | Small datasets, similarity-based |
        | Decision Tree | Both | Interpretable, simple models |
        """)

    with col_models2:
        st.markdown("""
        #### Time-Series Models
        | Model | Best For |
        |-------|----------|
        | Prophet | Seasonality, holidays, trend changes |
        | ARIMA | Stationary data, short-term forecasts |
        | SARIMAX | Seasonal + external variables |

        #### How Tasks Are Detected
        - **Regression** — Target has continuous numerical values (e.g., price, temperature).
        - **Classification** — Target has discrete categories (e.g., yes/no, species name).
        - **Time-Series** — Toggled manually when your data has a date/time dimension.
        """)

    st.divider()

    # ---- SECTION 6: SYSTEM REQUIREMENTS ----
    st.markdown("### 🔗 System & Environment")

    with st.expander("Python Packages (requirements.txt)", expanded=False):
        st.markdown("""
        | Package | Purpose |
        |---------|---------|
        | `streamlit` | Web UI framework |
        | `pandas`, `numpy` | Data manipulation |
        | `scikit-learn` | Standard ML models & utilities |
        | `xgboost` | Gradient boosting library |
        | `prophet` | Facebook's time-series forecaster |
        | `statsmodels` | ARIMA / SARIMAX / statistical tests |
        | `plotly` | Interactive charts |
        | `optuna` | Bayesian hyperparameter optimization |
        | `langchain`, `langchain-groq` | LLM orchestration + Groq API |
        | `langchain-huggingface` | Embedding models |
        | `faiss-cpu` | Vector store for RAG |
        | `python-dotenv` | Environment variable management |
        """)

    with st.expander("Environment Setup", expanded=False):
        st.markdown("""
        1. **Create a virtual environment:**
           ```bash
           python -m venv venv
           source venv/bin/activate  # Linux/Mac
           .\\venv\\Scripts\\activate  # Windows
           ```
        2. **Install dependencies:**
           ```bash
           pip install -r requirements.txt
           ```
        3. **Set up API keys** — Create a `.env` file in the project root:
           ```
           GROQ_API_KEY=your_groq_api_key_here
           HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
           ```
        4. **Build the knowledge base** (first time only):
           ```bash
           python app_backend/llm_rag_core.py
           ```
        5. **Run the app:**
           ```bash
           streamlit run app_frontend/main_ui.py
           ```
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: #64748b; font-size: 0.85rem;">
        AutoML Assistant v1.0 · Built with Streamlit, Scikit-learn, XGBoost & Groq AI<br>
        Need help? Check the troubleshooting section above or review the project documentation.
    </div>
    """, unsafe_allow_html=True)
