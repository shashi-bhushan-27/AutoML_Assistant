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

# --- WORKFLOW TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📂 1. Upload & Clean", "🔍 2. Analysis", "🤖 3. Model Training", "🚀 4. Optimization"])

# ==========================
# TAB 1: UPLOAD & CLEAN
# ==========================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Data Import", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        try:
            df_original = pd.read_csv(uploaded_file, on_bad_lines='skip')
            
            with col2:
                st.markdown("### Quick Look", unsafe_allow_html=True)
                st.dataframe(df_original.head(5), use_container_width=True)
            
            st.divider()
            
            col_clean_1, col_clean_2 = st.columns([1, 1])
            with col_clean_1:
                st.markdown("### Data Cleaning", unsafe_allow_html=True)
                cols_to_drop = st.multiselect("Select columns to remove:", df_original.columns)
            
            if cols_to_drop:
                df = df_original.drop(columns=cols_to_drop)
                st.toast(f"✅ Dropped {len(cols_to_drop)} columns.")
                st.info(f"Analysis will use {len(df.columns)} columns.")
            else:
                df = df_original.copy()
            
            st.session_state.df_clean = df
            
        except Exception as e:
            st.error(f"❌ Error parsing CSV: {e}")
    else:
        with col2:
            st.info("👆 Upload a CSV file to get started.")

# ==========================
# TAB 2: ANALYSIS
# ==========================
with tab2:
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
                        recs = advisor.get_recommendations(stats)
                        st.session_state.recommendations = recs
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
                
                st.markdown("#### AI Recommendations")
                st.success(f"**Strategy:** {', '.join(st.session_state.recommendations)}")
                
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
        
        # Default selection: Intersection of AI Recs and Supported Models
        default_selection = [m for m in st.session_state.recommendations if m in all_supported_models]
        if not default_selection: default_selection = ["XGBoost", "Random Forest"]
        
        selected_models = st.multiselect(
            "Select models to train:",
            options=all_supported_models,
            default=default_selection
        )
        
        if st.button("🚀 Start Training", type="primary"):
            if not selected_models:
                st.error("Select at least one model.")
            else:
                with st.status("Training models...", expanded=True) as status:
                    trainer = st.session_state.trainer
                    results_df = trainer.run_selected_models(selected_models)
                    st.session_state.results_df = results_df
                    status.update(label="✅ Training Complete!", state="complete", expanded=False)
        
        # LEADERBOARD
        if st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            
            st.divider()
            st.markdown("### 🏆 Leaderboard", unsafe_allow_html=True)
            
            if not results_df.empty and "Error" not in results_df.columns:
                sort_metric = "RMSE" if st.session_state.stats['task_type'] == "Regression" else "Accuracy"
                ascending = True if sort_metric == "RMSE" else False
                
                if sort_metric in results_df.columns:
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
                n_iter = st.slider("Number of Iterations", min_value=5, max_value=50, value=15, 
                                   help="How many random parameter combinations to try.")
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
                with st.spinner(f"Tuning {model_to_tune} with {n_iter} iterations and {cv_folds}-fold CV..."):
                    tuner = ModelTuner(st.session_state.trainer)
                    tuned_res = tuner.tune_model(model_to_tune, n_iter=n_iter, cv_folds=cv_folds, custom_params=custom_params)
                    
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
                            except Exception as e:
                                st.warning(f"Serialization failed: {e}")
        else:
            st.warning("No valid models to tune.")
    else:
        st.info("Train a model in Step 3 first.")
