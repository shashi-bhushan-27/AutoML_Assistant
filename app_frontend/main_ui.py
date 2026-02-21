import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from app_backend.shap_explainer import SHAPExplainer

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
if 'reasoning' not in st.session_state: st.session_state.reasoning = []
if 'trainer' not in st.session_state: st.session_state.trainer = None
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'df_clean' not in st.session_state: st.session_state.df_clean = None
if 'preprocessor' not in st.session_state: st.session_state.preprocessor = None
if 'preprocess_report' not in st.session_state: st.session_state.preprocess_report = None
if 'X_train' not in st.session_state: st.session_state.X_train = None
if 'X_test' not in st.session_state: st.session_state.X_test = None
if 'y_train' not in st.session_state: st.session_state.y_train = None
if 'y_test' not in st.session_state: st.session_state.y_test = None
if 'current_workspace' not in st.session_state: st.session_state.current_workspace = None
if 'workspace_manager' not in st.session_state: st.session_state.workspace_manager = WorkspaceManager()
if 'view_mode' not in st.session_state: st.session_state.view_mode = "home"  # "home" or "workspace"

# === WORKSPACE-FIRST UI FLOW ===
# Hot-reload guard: re-instantiate if WorkspaceManager is stale (missing new methods)
if not hasattr(st.session_state.workspace_manager, 'save_training_data'):
    st.session_state.workspace_manager = WorkspaceManager()
wm = st.session_state.workspace_manager
workspaces = wm.list_workspaces()

# Show workspace selector if no active workspace
if st.session_state.current_workspace is None:
    home_tab_workspaces, home_tab_docs = st.tabs(["📁 Workspaces", "📖 Docs & Help"])
    
    with home_tab_workspaces:
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
                            
                            # --- FULL STATE RESTORATION ---
                            # 1. Restore dataset
                            saved_df = wm.load_dataset(ws_data['workspace_id'])
                            if saved_df is not None:
                                st.session_state.df_clean = saved_df
                            
                            # 2. Restore session state (stats, recommendations, etc.)
                            saved_state = wm.load_session_state(ws_data['workspace_id'])
                            if saved_state:
                                st.session_state.stats = saved_state.get('stats')
                                st.session_state.recommendations = saved_state.get('recommendations', [])
                                st.session_state.preprocess_report = saved_state.get('preprocess_report')
                                st.session_state.reasoning = saved_state.get('reasoning', [])
                                # Restore results_df
                                saved_results = saved_state.get('results_df')
                                if saved_results is not None:
                                    st.session_state.results_df = pd.DataFrame(saved_results)
                            
                            # 3. Restore preprocessed training data splits
                            saved_training_data = wm.load_training_data(ws_data['workspace_id'])
                            if saved_training_data:
                                st.session_state.X_train = saved_training_data.get('X_train')
                                st.session_state.X_test = saved_training_data.get('X_test')
                                st.session_state.y_train = saved_training_data.get('y_train')
                                st.session_state.y_test = saved_training_data.get('y_test')
                            
                            # 4. Restore preprocessor
                            saved_preprocessor = wm.load_preprocessor(ws_data['workspace_id'])
                            if saved_preprocessor is not None:
                                st.session_state.preprocessor = saved_preprocessor
                            
                            # 5. Rebuild ModelTrainer if we have stats + data
                            if saved_df is not None and st.session_state.stats is not None and loaded_ws.target_col:
                                try:
                                    restored_trainer = ModelTrainer(
                                        saved_df,
                                        loaded_ws.target_col,
                                        st.session_state.stats.get('task_type', 'Regression'),
                                        st.session_state.stats.get('is_time_series', False),
                                        st.session_state.stats.get('time_column')
                                    )
                                    # If we have preprocessed splits, set them on the trainer
                                    if saved_training_data:
                                        restored_trainer.set_preprocessed_data(
                                            saved_training_data['X_train'],
                                            saved_training_data['X_test'],
                                            saved_training_data['y_train'],
                                            saved_training_data['y_test']
                                        )
                                    # Restore trained model objects
                                    saved_models = wm.load_trained_models(ws_data['workspace_id'])
                                    if saved_models:
                                        restored_trainer.trained_models = saved_models
                                    st.session_state.trainer = restored_trainer
                                except Exception:
                                    pass  # Trainer rebuild failed, user can re-analyze
                            
                            st.rerun()
                    
                    with btn_col2:
                        if st.button("🗑️", key=f"del_{ws_data['workspace_id']}", use_container_width=True, help="Delete workspace"):
                            wm.delete_workspace(ws_data['workspace_id'])
                            st.toast(f"Deleted workspace: {ws_data['dataset_name']}")
                            st.rerun()
        else:
            st.info("No workspaces yet. Create one to get started!")
    
    with home_tab_docs:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="font-size: 2.2rem; margin-bottom: 0.3rem;">📖 Documentation & <span style="color: #ec4899">Help Center</span></h2>
            <p style="font-size: 1rem; color: #94a3b8; max-width: 600px; margin: 0 auto;">
                Your complete guide to AutoML Assistant v3.0 — now with SHAP Explainability, FastAPI, LSTM, and Multi-Class Metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ============================================================
        # SECTION 1: GETTING STARTED
        # ============================================================
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        AutoML Assistant guides you through a **5-step workflow** from raw CSV to a trained, explained, deployable ML model:

        | Step | Tab | What Happens |
        |:----:|-----|-------------|
        | **1** | 📂 Upload | Import your CSV dataset and remove unwanted columns |
        | **2** | 🔧 Preprocessing | Auto-clean, encode, and scale your data |
        | **3** | 🔍 Analysis | AI analyzes your data and recommends the best models |
        | **4** | 🤖 Training | Train models, compare on leaderboard, explain with SHAP, download `.pkl` |
        | **5** | ⚡ Optimization | Fine-tune the best model with Bayesian hyperparameter search |

        > **Tip:** Always start by creating a **Workspace** — it saves your entire session so you can come back anytime.
        """)

        st.divider()

        # ============================================================
        # SECTION 2: STEP-BY-STEP GUIDE
        # ============================================================
        st.markdown("### 📋 Step-by-Step Guide")

        with st.expander("📂 Step 1 — Upload & Clean", expanded=False):
            st.markdown("""
            **Goal:** Import your dataset into the platform.

            1. Click **'➕ Create New Workspace'** on the home screen.
            2. In the **Upload** tab, drag-and-drop or browse for a **CSV file**.
            3. A preview of the first 5 rows will appear on the right.
            4. *(Optional)* Use the **'Select columns to remove'** dropdown to drop irrelevant columns (IDs, names, etc.).
            5. Click **'Apply Column Removal'** to confirm.

            | Question | Answer |
            |----------|--------|
            | Supported formats? | **`.csv`** only (comma-delimited, UTF-8 preferred) |
            | Max file size? | No hard limit, but **< 200 MB** recommended |
            | What about encoding issues? | The `chardet` library auto-detects encoding |
            """)

        with st.expander("🔧 Step 2 — Preprocessing", expanded=False):
            st.markdown("""
            **Goal:** Automatically clean and prepare your data for ML training.

            1. Select your **Target Variable** (the column you want to predict).
            2. *(Optional)* Toggle **'Time Series Data?'** and select the date column.
            3. *(Optional)* Toggle **'Apply SMOTE'** for imbalanced classification datasets.
            4. Click **'🚀 Run Preprocessing'**.

            The `AutoPreprocessor` engine performs:
            - **Type Detection**: Numerical, Categorical, DateTime
            - **Missing Value Imputation**: Median (numeric) / Mode (categorical)
            - **Outlier Handling**: IQR-based capping
            - **Encoding**: One-Hot Encoding (low-cardinality) / Label Encoding
            - **Scaling**: Standard Scaler (Z-score normalization)
            - **Train/Test Split**: 80% / 20% stratified

            **Visual Outputs:**
            - 📊 Feature distribution histograms
            - 🔥 Correlation heatmap
            - 🥧 Class balance pie chart

            Click **'💾 Export Pipeline'** to save the fitted preprocessor as a `.pkl` file for reuse in production.
            """)

        with st.expander("🔍 Step 3 — Analysis (AI-Powered)", expanded=False):
            st.markdown("""
            **Goal:** Let the AI analyze your dataset and recommend the best algorithms.

            1. Select your **Target Variable**.
            2. Click **'🔍 Analyze Dataset'**.

            The system will:
            - Generate full statistical profiles (correlation, skewness, class balance, ADF stationarity test)
            - Send the profile to **Groq Llama 3.1** via a FAISS-backed RAG pipeline
            - Return 3–5 model recommendations with detailed reasoning

            **Key UI Elements:**
            | Element | Description |
            |---------|-------------|
            | Data Insights Panel | Row count, feature count, detected task type |
            | AI Recommendations | Recommended models, auto-selected in training UI |
            | 🧠 Why This Model? | Expandable reasoning panel — AI explains its logic in plain language |
            | 💡 Meta-Learning | Badge appears when past experiments are used to bias recommendations |

            > ⚠️ Requires a valid `GROQ_API_KEY` in `.env`. If the API fails, the system falls back to safe defaults (XGBoost + Random Forest).
            """)

        with st.expander("🤖 Step 4 — Training, Evaluation & Explainability", expanded=False):
            st.markdown("""
            **Goal:** Train models, evaluate with rich metrics, download `.pkl` files, and understand *why* the model makes predictions.

            #### Training
            1. AI-recommended models are **pre-selected** in the multi-select dropdown.
            2. Add or remove models as needed.
            3. Click **'🚀 Start Training'**. A progress bar shows live status.

            #### Leaderboard & Downloads
            - 🏆 **Leaderboard** ranks models:
              - Regression → sorted by RMSE (↓ lower is better)
              - Classification → sorted by Accuracy (↑ higher is better)
            - ⬇️ **Per-model download buttons** appear directly below the leaderboard — click to download any trained model as a `.pkl` file instantly.

            #### Evaluation Metrics
            **Regression:** RMSE, MAE, R², MAPE (%)

            **Classification (Binary):** Accuracy, F1, Precision, Recall, MCC, Cohen Kappa, AUC-ROC

            **Classification (Multi-class):** All binary metrics + Per-Class Precision/Recall/F1/Support table + Per-Class F1 bar chart

            > **What is MCC?** Matthews Correlation Coefficient — the most reliable single metric for imbalanced class datasets. Ranges from -1 (worst) to +1 (perfect).

            #### Evaluation Charts
            - Confusion Matrix (interactive heatmap)
            - ROC Curves (AUC per model)
            - Radar Chart (multi-metric comparison)
            - Predicted vs Actual scatter (regression)
            - Residuals distribution (regression)
            - Feature Importance (tree-based models)
            - Training Time comparison bar chart

            #### 🔍 SHAP Explainability (NEW in v3.0)
            After training, scroll down to the **'SHAP Model Explainability'** section:

            1. Select a model from the dropdown.
            2. Click **'Generate SHAP Explanations'** (takes 10–30s).
            3. Four Plotly charts appear:

            | Chart | What It Shows |
            |-------|---------------|
            | 📊 Feature Importance | Mean |SHAP| per feature — global impact ranking |
            | 🐝 Beeswarm | Per-sample SHAP values — direction & magnitude of each feature |
            | 💧 Waterfall | Single-prediction breakdown — why *this* specific prediction was made |
            | 📉 Dependence Plot | How one feature's value shifts the model output (with LOWESS trendline) |

            > SHAP auto-selects the fastest explainer: `TreeExplainer` → `LinearExplainer` → `KernelExplainer`.
            """)

        with st.expander("⚡ Step 5 — Hyperparameter Optimization", expanded=False):
            st.markdown("""
            **Goal:** Squeeze extra performance from your best model via intelligent hyperparameter search.

            1. Select the model to optimize from the dropdown.
            2. Set a **Time Budget** (default: 30s). More time = deeper exploration.
            3. Set **Cross-Validation Folds** (default: 3).
            4. *(Optional)* Expand **'Advanced: Edit Parameter Ranges'** to set custom search bounds.
            5. Click **'✨ Optimize [Model]'**.

            Uses **Optuna** (Bayesian optimization) — intelligently explores the parameter space, pruning unpromising trials early.

            **After Optimization:**
            - Best parameters and score are displayed as JSON.
            - **'💾 Download Tuned Model'** → Export the optimized model as `.pkl`.
            - **'📜 View/Copy Training Code'** → Ready-to-run Python script (copy to Google Colab).

            > **Tip:** Tune for 60–120 seconds on complex models like XGBoost for best results.
            """)

        st.divider()

        # ============================================================
        # SECTION 3: NEW FEATURES (v3.0)
        # ============================================================
        st.markdown("### 🌟 What's New in v3.0")

        col_new1, col_new2 = st.columns(2)
        with col_new1:
            st.markdown("""
            #### 🔍 SHAP Explainability
            - Powered by the `shap` library
            - 4 interactive Plotly charts per model
            - Global (Feature Importance, Beeswarm) & Local (Waterfall, Dependence) explanations
            - **Enable:** `pip install shap` (already in `requirements.txt`)

            #### 📊 Multi-Class Metrics
            - MCC, Cohen Kappa, AUC-ROC (One-vs-Rest)
            - MAPE (%) for regression
            - Per-class F1/Precision/Recall/Support breakdown & chart
            - Handles binary and multi-class automatically
            """)
        with col_new2:
            st.markdown("""
            #### 🌐 FastAPI `/predict` Endpoint
            - REST API for live model inference from any client
            - Endpoints: `/predict`, `/predict/csv`, `/workspaces`, `/health`
            - Auto-applies workspace preprocessor before serving predictions
            - **Run:** `python app_backend/main_api.py`
            - **Docs:** `http://localhost:8000/docs`

            #### 🧠 LSTM Deep Learning (Time-Series)
            - 2-layer Keras LSTM (64→32) with Dropout + EarlyStopping
            - Walk-forward forecasting (no data leakage)
            - Auto-normalizes target; shown only when TensorFlow detected
            - **Enable:** `pip install tensorflow`
            """)

        st.divider()

        # ============================================================
        # SECTION 4: METRICS REFERENCE
        # ============================================================
        st.markdown("### 📏 Metrics Reference")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            #### Regression Metrics
            | Metric | Description | Goal |
            |--------|-------------|------|
            | **RMSE** | Root Mean Squared Error — penalises large errors | ↓ Lower |
            | **MAE** | Mean Absolute Error — average absolute difference | ↓ Lower |
            | **R²** | % variance explained by the model | ↑ Higher (max 1.0) |
            | **MAPE (%)** | Mean Absolute % Error — scale-independent | ↓ Lower |
            """)
        with col_m2:
            st.markdown("""
            #### Classification Metrics
            | Metric | Description | Goal |
            |--------|-------------|------|
            | **Accuracy** | % correct predictions | ↑ Higher |
            | **F1 Score** | Harmonic mean of Precision & Recall | ↑ Higher |
            | **Precision** | True Pos / Predicted Pos | ↑ Higher |
            | **Recall** | True Pos / Actual Pos | ↑ Higher |
            | **MCC** ⭐ | Matthews Corr. Coeff. — best for imbalanced data (-1→1) | ↑ Higher |
            | **Cohen Kappa** ⭐ | Agreement beyond chance (-1→1) | ↑ Higher |
            | **AUC-ROC** ⭐ | Area under ROC curve, OvR for multi-class | ↑ Higher |
            """)

        st.divider()

        # ============================================================
        # SECTION 5: API REFERENCE
        # ============================================================
        st.markdown("### 🌐 FastAPI — Prediction Server")
        st.markdown("""
        The FastAPI server (`app_backend/main_api.py`) lets you query trained models from any language or tool.

        **Start the server:**
        ```bash
        python app_backend/main_api.py
        # → Running at http://localhost:8000
        # → Swagger UI at http://localhost:8000/docs
        ```
        """)

        tab_api1, tab_api2, tab_api3 = st.tabs(["JSON Predict", "CSV Batch Predict", "List Workspaces"])
        with tab_api1:
            st.code("""
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "workspace_id": "your_workspace_id",
    "model_name": "XGBoost",
    "data": [
      {"feature1": 1.5, "feature2": "cat_A", "feature3": 10}
    ]
  }'
            """, language="bash")
            st.caption("Response: `{ predictions, prediction_labels, probabilities }`")

        with tab_api2:
            st.code("""
curl -X POST \\
  "http://localhost:8000/predict/csv/{workspace_id}/{model_name}" \\
  -F "file=@my_test_data.csv"
            """, language="bash")
            st.caption("Response: JSON with `results` array — original rows + prediction column + per-class probabilities.")

        with tab_api3:
            st.code("""
# List all workspaces and their available models
curl http://localhost:8000/workspaces

# List models in a specific workspace
curl http://localhost:8000/workspaces/{workspace_id}/models
            """, language="bash")

        st.divider()

        # ============================================================
        # SECTION 6: TROUBLESHOOTING
        # ============================================================
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
                "fix": "1. Check your `.env` file has a valid `GROQ_API_KEY`.\n2. Run `python test_groq_connection.py` to test the key.\n3. The system still works with fallback models (XGBoost, Random Forest) even if the API fails."
            },
            {
                "error": "Vector Store not found!",
                "cause": "The FAISS vector store (used by the AI advisor) has not been built yet.",
                "fix": "Run once: `python app_backend/llm_rag_core.py` — this builds the FAISS knowledge base index."
            },
            {
                "error": "SMOTE error / Too few samples",
                "cause": "SMOTE requires at least 2 samples per class and only works for classification tasks.",
                "fix": "1. Ensure your target variable is categorical (not continuous).\n2. Ensure each class has at least 2 samples.\n3. For very small datasets (< 50 rows), disable SMOTE."
            },
            {
                "error": "Model training fails / Convergence warning",
                "cause": "Some models (SVM, Logistic Regression) fail to converge on unscaled data.",
                "fix": "1. Always run **Preprocessing** (Step 2) before training.\n2. Try XGBoost or Random Forest — more robust to raw data.\n3. Remove highly correlated or constant columns."
            },
            {
                "error": "SHAP Explainer build failed",
                "cause": "The `shap` library is not installed, or the model type isn't supported.",
                "fix": "Run `pip install shap` and restart the app. For KNet/time-series models, SHAP is not available."
            },
            {
                "error": "LSTM not in model list",
                "cause": "TensorFlow is not installed in the current environment.",
                "fix": "Run `pip install tensorflow` (not in requirements.txt by default — it is a large dependency).\nRestart the app and toggle 'Time Series Data?' mode."
            },
            {
                "error": "FastAPI /predict returns 404 'Workspace not found'",
                "cause": "The workspace ID is incorrect or the workspace has no trained models saved.",
                "fix": "1. Run `GET /workspaces` to list all valid workspace IDs.\n2. Make sure you trained at least one model in that workspace before calling /predict."
            },
            {
                "error": "Workspace data missing after reload",
                "cause": "The Streamlit file uploader resets on page reload, but your data IS saved.",
                "fix": "Look for the green **'✅ Using stored dataset'** banner. Your data is safe — no need to re-upload."
            },
            {
                "error": "Serialization / Pickle error on download",
                "cause": "Some model objects (e.g., Keras LSTM) cannot be pickled with the standard pickle library.",
                "fix": "LSTM models should be saved with `model.save()` from Keras. Standard sklearn/XGBoost models export fine."
            },
            {
                "error": "Prophet / ARIMA / SARIMAX fails",
                "cause": "Time-series models need a valid datetime column and properly ordered data.",
                "fix": "1. Toggle **'Time Series Data?'** in preprocessing.\n2. Ensure your date column is parseable (YYYY-MM-DD).\n3. Sort your data by date before uploading."
            },
            {
                "error": "Per-Class Report not shown for Classification",
                "cause": "This section only renders if models have been trained and the Per-Class Report metadata is present in results.",
                "fix": "Re-train your classification models fresh in the current session. Per-class data is computed in `evaluate()` during training."
            },
        ]

        for item in errors_data:
            with st.expander(f"🔴 {item['error']}"):
                st.markdown(f"**Likely Cause:** {item['cause']}")
                st.markdown(f"**How to Fix:** {item['fix']}")

        st.divider()

        # ============================================================
        # SECTION 7: TIPS & BEST PRACTICES
        # ============================================================
        st.markdown("### 💡 Tips & Best Practices")

        col_tips1, col_tips2 = st.columns(2)
        with col_tips1:
            st.markdown("""
            #### Data Preparation
            - **Clean your CSV** before uploading — fix typos and remove blank rows.
            - **Drop ID/index columns** in Step 1 — they add noise.
            - **Aim for 100+ rows** for meaningful results.
            - **Check class balance** for classification — enable SMOTE if one class dominates.
            - **For Time-Series:** Sort by date and use a consistent time step (daily, monthly).

            #### Model Selection
            - **Start with AI recommendations** — tailored to your data profile.
            - **XGBoost / Random Forest** are reliable general-purpose choices.
            - **Logistic Regression**: works best with linear boundaries and scaled features.
            - **SVM**: slow on large datasets (> 10k rows) — use with caution.
            - **LSTM**: best for sequences with long-range dependencies (stock prices, sensor data).
            """)
        with col_tips2:
            st.markdown("""
            #### Optimization & Tuning
            - **Give more time** (60–120s) for complex models like XGBoost.
            - **Increase CV folds** (5–10) for small datasets to get a reliable score.
            - Always **download the tuned model + training code** for reproducibility.

            #### SHAP Explainability
            - Use the **Beeswarm** chart to spot features with inconsistent impact (wide spread).
            - Use the **Waterfall** chart to debug a specific wrong prediction.
            - The **Dependence** chart reveals non-linear relationships between features and output.
            - For KernelExplainer (fallback), reduce test size for speed (it samples 50 background rows).

            #### Deployment
            - Use **`/predict` via FastAPI** to integrate the trained model into any app or pipeline.
            - Export the **preprocessor `.pkl`** alongside the model `.pkl` — you'll need both for production inference.
            - Tag your workspace with meaningful experiment names for easy retrieval.
            """)

        st.divider()

        # ============================================================
        # SECTION 8: SUPPORTED MODELS
        # ============================================================
        st.markdown("### 📋 Supported Models")
        col_models1, col_models2 = st.columns(2)

        with col_models1:
            st.markdown("""
            #### Standard Models (Regression & Classification)
            | Model | Task | Best For |
            |-------|------|----------|
            | Linear Regression | Regression | Simple linear relationships |
            | Ridge / Lasso / ElasticNet | Regression | Regularized regression |
            | Logistic Regression | Classification | Binary / multi-class |
            | Random Forest | Both | Versatile, robust to outliers |
            | **XGBoost** ⭐ | Both | Best general-purpose model |
            | Gradient Boosting | Both | Strong ensemble method |
            | AdaBoost | Both | Adaptive boosting |
            | Extra Trees | Both | Fast Random Forest alternative |
            | Decision Tree | Both | Interpretable single tree |
            | SVM | Both | High-dimensional data |
            | KNN | Both | Similarity-based, small datasets |
            """)

        with col_models2:
            st.markdown("""
            #### Time-Series Models (toggle 'Time Series Data?')
            | Model | Best For |
            |-------|----------|
            | **Prophet** | Seasonality, holidays, trend changes |
            | **ARIMA** | Stationary data, short-term forecasts |
            | **SARIMAX** | Seasonal data + external variables |
            | **LSTM** ⭐ | Long-range dependencies, deep learning (requires TensorFlow) |

            #### How Task Type Is Detected
            | Task | Trigger |
            |------|---------|
            | **Regression** | Target has continuous numeric values (price, temp) |
            | **Classification** | Target has discrete categories (yes/no, species) |
            | **Time-Series** | Manually toggled in Preprocessing step |
            """)

        st.divider()

        # ============================================================
        # SECTION 9: ENVIRONMENT SETUP
        # ============================================================
        st.markdown("### 🔗 Environment Setup")

        with st.expander("Python Packages (requirements.txt)", expanded=False):
            st.markdown("""
            | Package | Purpose |
            |---------|---------|
            | `streamlit` | Web UI framework |
            | `fastapi` + `uvicorn` | REST API server for `/predict` |
            | `pandas`, `numpy` | Data manipulation |
            | `scikit-learn` | Standard ML models & utilities |
            | `xgboost` | Gradient boosting |
            | `shap` | Model explainability |
            | `prophet` | Time-series forecasting |
            | `statsmodels` | ARIMA / SARIMAX / ADF test |
            | `plotly` | Interactive charts |
            | `optuna` | Bayesian hyperparameter tuning |
            | `langchain`, `langchain-groq` | LLM orchestration + Groq API |
            | `langchain-huggingface` | HuggingFace embedding models |
            | `faiss-cpu` | Vector store for RAG |
            | `chardet` | CSV encoding auto-detection |
            | `scipy` | Statistical functions |
            | `tensorflow` *(optional)* | LSTM deep learning support |
            """)

        with st.expander("Quick Setup Guide", expanded=False):
            st.code("""
# 1. Create virtual environment
python -m venv venv
venv\\Scripts\\activate        # Windows
source venv/bin/activate       # Linux / Mac

# 2. Install all dependencies
pip install -r requirements.txt

# 3. [Optional] Enable LSTM support
pip install tensorflow

# 4. Create .env file
# GROQ_API_KEY=your_groq_api_key_here
# HUGGINGFACEHUB_API_TOKEN=your_hf_token_here

# 5. Build the knowledge base (first time only)
python app_backend/llm_rag_core.py

# 6. Launch the Streamlit app
streamlit run app_frontend/main_ui.py

# 7. [Optional] Launch the FastAPI prediction server
python app_backend/main_api.py
# → http://localhost:8000  |  Docs: http://localhost:8000/docs
            """, language="bash")

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: #64748b; font-size: 0.85rem;">
            AutoML Assistant <b>v3.0</b> · Built with Streamlit, Scikit-learn, XGBoost, SHAP, FastAPI &amp; Groq AI<br>
            Last updated: 2026-02-21 · Branch: <code>feature/high-priority-improvements</code>
        </div>
        """, unsafe_allow_html=True)
    
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
tab1, tab2, tab2b, tab3, tab4 = st.tabs(["📂 1. Upload", "🔧 2. Preprocessing", "🔍 3. Analysis", "🤖 4. Training", "🚀 5. Optimization"])

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
                        
                        # Save preprocessor pipeline
                        st.session_state.workspace_manager.save_preprocessor(ws.workspace_id, preprocessor)
                        
                        # Save training data splits
                        st.session_state.workspace_manager.save_training_data(
                            ws.workspace_id,
                            result["X_train"], result["X_test"],
                            result["y_train"], result["y_test"]
                        )
                        
                        # Save session state for restoration
                        st.session_state.workspace_manager.save_session_state(ws.workspace_id, {
                            'stats': st.session_state.stats,
                            'recommendations': st.session_state.recommendations,
                            'reasoning': st.session_state.get('reasoning', []),
                            'preprocess_report': st.session_state.preprocess_report,
                            'results_df': st.session_state.results_df.to_dict() if st.session_state.results_df is not None else None
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
                
                # ==============================
                # PREPROCESSING VISUALIZATIONS
                # ==============================
                st.divider()
                st.markdown("### 📊 Data Visualizations")
                
                df_viz = st.session_state.df_clean
                
                # --- A2: Feature Distribution Grid ---
                with st.expander("📈 Feature Distributions", expanded=True):
                    num_cols = df_viz.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns.tolist()
                    # Remove target col from viz if present
                    ws_target = st.session_state.current_workspace.target_col if st.session_state.current_workspace else None
                    viz_cols = [c for c in num_cols if c != ws_target][:12]  # Max 12 columns
                    
                    if viz_cols:
                        # Create grid of histograms - 3 per row
                        for i in range(0, len(viz_cols), 3):
                            row_cols = st.columns(3)
                            for j, col_name in enumerate(viz_cols[i:i+3]):
                                with row_cols[j]:
                                    fig = px.histogram(
                                        df_viz, x=col_name, 
                                        nbins=30,
                                        title=col_name,
                                        template="plotly_dark",
                                        color_discrete_sequence=["#8b5cf6"]
                                    )
                                    fig.update_layout(
                                        height=250, 
                                        margin=dict(l=10, r=10, t=35, b=10),
                                        showlegend=False,
                                        title_font_size=13
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No numerical columns to display.")
                
                # --- A3: Correlation Heatmap ---
                with st.expander("🔥 Correlation Heatmap", expanded=False):
                    num_cols_corr = df_viz.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns.tolist()
                    if len(num_cols_corr) >= 2:
                        corr_matrix = df_viz[num_cols_corr].corr()
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title="Feature Correlation Matrix",
                            template="plotly_dark"
                        )
                        fig_corr.update_layout(
                            height=500,
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Need at least 2 numerical columns for correlation heatmap.")
                
                # --- A5: Class Balance Chart ---
                if ws_target and ws_target in df_viz.columns:
                    target_series = df_viz[ws_target]
                    if target_series.dtype == 'object' or target_series.nunique() < 15:
                        with st.expander("⚖️ Target Class Balance", expanded=False):
                            class_counts = target_series.value_counts().reset_index()
                            class_counts.columns = ['Class', 'Count']
                            fig_balance = px.pie(
                                class_counts, 
                                values='Count', 
                                names='Class',
                                title=f"Distribution of '{ws_target}'",
                                template="plotly_dark",
                                hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig_balance.update_layout(
                                height=400,
                                margin=dict(l=10, r=10, t=40, b=10)
                            )
                            st.plotly_chart(fig_balance, use_container_width=True)
    else:
        st.info("Please upload data in Step 1 first.")

# ==========================
# TAB 3: ANALYSIS
# ==========================
with tab2b:
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        # Auto-use target from preprocessing (stored in workspace)
        ws = st.session_state.current_workspace
        saved_target = ws.target_col if ws and ws.target_col else None
        
        if saved_target and saved_target in df.columns:
            target_col = saved_target
        else:
            target_col = None
        
        col_c1, col_c2 = st.columns([1, 3])
        
        with col_c1:
            st.markdown("### Configuration", unsafe_allow_html=True)
            if not df.empty:
                if target_col:
                    st.success(f"🎯 Target Variable: **{target_col}**")
                    st.caption("Set in Preprocessing step.")
                    
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
                                
                                # Save session state after analysis
                                st.session_state.workspace_manager.save_session_state(ws.workspace_id, {
                                    'stats': st.session_state.stats,
                                    'recommendations': st.session_state.recommendations,
                                    'reasoning': st.session_state.get('reasoning', []),
                                    'preprocess_report': st.session_state.preprocess_report,
                                    'results_df': st.session_state.results_df.to_dict() if st.session_state.results_df is not None else None
                                })
                else:
                    st.warning("⚠️ Please complete **Step 2 (Preprocessing)** first to set the target variable.")
        
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
        
        select_all = st.checkbox("✅ Select All Models", value=False)
        
        selected_models = st.multiselect(
            "Select models to train:",
            options=all_supported_models,
            default=all_supported_models if select_all else default_selection,
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
                        
                        # Save trained model objects
                        st.session_state.workspace_manager.save_trained_models(
                            ws.workspace_id, trainer.trained_models
                        )
                        
                        # Save full session state including results
                        st.session_state.workspace_manager.save_session_state(ws.workspace_id, {
                            'stats': st.session_state.stats,
                            'recommendations': st.session_state.recommendations,
                            'reasoning': st.session_state.get('reasoning', []),
                            'preprocess_report': st.session_state.preprocess_report,
                            'results_df': results_df.to_dict()
                        })
                    
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

                # --- Per-Model Download Buttons ---
                st.markdown("#### 💾 Download Trained Models")
                trainer = st.session_state.trainer
                if trainer and trainer.trained_models:
                    dl_cols = st.columns(min(4, len(trainer.trained_models)))
                    for idx, (mname, mobj) in enumerate(trainer.trained_models.items()):
                        with dl_cols[idx % 4]:
                            try:
                                import pickle as _pickle
                                pkl_bytes = _pickle.dumps(mobj)
                                st.download_button(
                                    label=f"⬇️ {mname}",
                                    data=pkl_bytes,
                                    file_name=f"{mname.replace(' ', '_')}_model.pkl",
                                    mime="application/octet-stream",
                                    key=f"dl_{mname}",
                                    use_container_width=True,
                                )
                            except Exception:
                                st.caption(f"⚠️ {mname} not serializable")
            else:
                st.warning("No successful results found.")

            # ==============================
            # DETAILED EVALUATION CHARTS
            # ==============================
            st.divider()
            st.markdown("### 📊 Detailed Evaluation")
            
            trainer = st.session_state.trainer
            task_type = st.session_state.stats.get('task_type', 'Regression') if st.session_state.stats else 'Regression'
            
            # --- B6: Training Time Comparison ---
            if "Time (s)" in results_df.columns:
                with st.expander("⏱️ Training Time Comparison", expanded=False):
                    time_df = results_df.sort_values("Time (s)", ascending=True)
                    fig_time = px.bar(
                        time_df, x="Time (s)", y="Model",
                        orientation='h',
                        title="Training Duration per Model",
                        template="plotly_dark",
                        color="Time (s)",
                        color_continuous_scale="Viridis"
                    )
                    fig_time.update_layout(
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10),
                        yaxis=dict(categoryorder='total ascending')
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
            
            if trainer and hasattr(trainer, 'predictions') and trainer.predictions:
                
                if task_type == "Classification":
                    # --- B1: Confusion Matrix ---
                    with st.expander("🎯 Confusion Matrix", expanded=True):
                        from sklearn.metrics import confusion_matrix
                        model_names = list(trainer.predictions.keys())
                        selected_cm_model = st.selectbox("Select model:", model_names, key="cm_model_select")
                        
                        if selected_cm_model in trainer.predictions:
                            y_true = trainer.y_test
                            y_pred = trainer.predictions[selected_cm_model]
                            
                            labels = sorted(list(set(y_true) | set(y_pred)))
                            cm = confusion_matrix(y_true, y_pred, labels=labels)
                            
                            fig_cm = px.imshow(
                                cm,
                                text_auto=True,
                                x=[str(l) for l in labels],
                                y=[str(l) for l in labels],
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                title=f"Confusion Matrix — {selected_cm_model}",
                                template="plotly_dark",
                                color_continuous_scale="Blues"
                            )
                            fig_cm.update_layout(
                                height=450,
                                margin=dict(l=10, r=10, t=40, b=10)
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # --- B2: ROC Curves ---
                    if trainer.prediction_probas:
                        with st.expander("📉 ROC Curves", expanded=False):
                            from sklearn.metrics import roc_curve, auc
                            from sklearn.preprocessing import label_binarize
                            
                            y_true = trainer.y_test
                            classes = sorted(list(set(y_true)))
                            
                            fig_roc = go.Figure()
                            
                            for model_name, probas in trainer.prediction_probas.items():
                                try:
                                    if len(classes) == 2:
                                        # Binary classification
                                        fpr, tpr, _ = roc_curve(y_true, probas[:, 1])
                                        roc_auc = auc(fpr, tpr)
                                        fig_roc.add_trace(go.Scatter(
                                            x=fpr, y=tpr, mode='lines',
                                            name=f"{model_name} (AUC={roc_auc:.3f})"
                                        ))
                                    else:
                                        # Multi-class: use micro-average
                                        y_bin = label_binarize(y_true, classes=classes)
                                        fpr, tpr, _ = roc_curve(y_bin.ravel(), probas.ravel())
                                        roc_auc = auc(fpr, tpr)
                                        fig_roc.add_trace(go.Scatter(
                                            x=fpr, y=tpr, mode='lines',
                                            name=f"{model_name} (micro-AUC={roc_auc:.3f})"
                                        ))
                                except Exception:
                                    pass
                            
                            # Diagonal line
                            fig_roc.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1], mode='lines',
                                line=dict(dash='dash', color='gray'),
                                name='Random', showlegend=False
                            ))
                            fig_roc.update_layout(
                                title="ROC Curves — All Models",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate",
                                template="plotly_dark",
                                height=450,
                                margin=dict(l=10, r=10, t=40, b=10)
                            )
                            st.plotly_chart(fig_roc, use_container_width=True)
                    
                    # --- B3: Radar Chart for multi-metric comparison ---
                    if all(m in results_df.columns for m in ['Accuracy', 'F1 Score', 'Precision', 'Recall']):
                        with st.expander("🕸️ Multi-Metric Radar", expanded=False):
                            fig_radar = go.Figure()
                            metrics_list = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                            
                            for _, row in results_df.iterrows():
                                if pd.notna(row.get('Accuracy')):
                                    values = [row[m] for m in metrics_list] + [row[metrics_list[0]]]
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=metrics_list + [metrics_list[0]],
                                        fill='toself',
                                        name=row['Model'],
                                        opacity=0.6
                                    ))
                            
                            fig_radar.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                title="Model Comparison — All Metrics",
                                template="plotly_dark",
                                height=500,
                                margin=dict(l=40, r=40, t=60, b=40)
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
                
                else:
                    # --- B4: Residual / Predicted vs Actual Plot (Regression) ---
                    with st.expander("📈 Predicted vs Actual", expanded=True):
                        model_names = list(trainer.predictions.keys())
                        selected_res_model = st.selectbox("Select model:", model_names, key="residual_model_select")
                        
                        if selected_res_model in trainer.predictions:
                            y_true = trainer.y_test
                            y_pred = trainer.predictions[selected_res_model]
                            
                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                # Predicted vs Actual scatter
                                fig_pva = px.scatter(
                                    x=y_true, y=y_pred,
                                    labels={'x': 'Actual', 'y': 'Predicted'},
                                    title=f"Predicted vs Actual — {selected_res_model}",
                                    template="plotly_dark",
                                    color_discrete_sequence=["#06b6d4"]
                                )
                                # 45-degree reference line
                                min_val = min(min(y_true), min(y_pred))
                                max_val = max(max(y_true), max(y_pred))
                                fig_pva.add_trace(go.Scatter(
                                    x=[min_val, max_val], y=[min_val, max_val],
                                    mode='lines', line=dict(dash='dash', color='#ef4444'),
                                    name='Perfect', showlegend=False
                                ))
                                fig_pva.update_layout(
                                    height=400,
                                    margin=dict(l=10, r=10, t=40, b=10)
                                )
                                st.plotly_chart(fig_pva, use_container_width=True)
                            
                            with res_col2:
                                # Residual distribution
                                residuals = y_true - y_pred
                                fig_res = px.histogram(
                                    x=residuals, nbins=30,
                                    labels={'x': 'Residual (Actual - Predicted)'},
                                    title="Residual Distribution",
                                    template="plotly_dark",
                                    color_discrete_sequence=["#f59e0b"]
                                )
                                fig_res.update_layout(
                                    height=400,
                                    margin=dict(l=10, r=10, t=40, b=10)
                                )
                                st.plotly_chart(fig_res, use_container_width=True)
                
                # --- B5: Feature Importance (works for both Classification & Regression) ---
                with st.expander("🏅 Feature Importance", expanded=False):
                    # Find best tree-based model
                    tree_models = {}
                    for name, model in trainer.trained_models.items():
                        if hasattr(model, 'feature_importances_'):
                            tree_models[name] = model
                    
                    if tree_models:
                        fi_model_name = st.selectbox("Select model:", list(tree_models.keys()), key="fi_model_select")
                        model = tree_models[fi_model_name]
                        
                        # Get feature names
                        if hasattr(trainer, 'X_train') and trainer.X_train is not None:
                            if hasattr(trainer.X_train, 'columns'):
                                feature_names = trainer.X_train.columns.tolist()
                            else:
                                feature_names = [f"Feature {i}" for i in range(trainer.X_train.shape[1])]
                        else:
                            feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
                        
                        importances = model.feature_importances_
                        fi_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=True).tail(15)  # Top 15
                        
                        fig_fi = px.bar(
                            fi_df, x='Importance', y='Feature',
                            orientation='h',
                            title=f"Top 15 Features — {fi_model_name}",
                            template="plotly_dark",
                            color='Importance',
                            color_continuous_scale="Plasma"
                        )
                        fig_fi.update_layout(
                            height=450,
                            margin=dict(l=10, r=10, t=40, b=10),
                            yaxis=dict(categoryorder='total ascending')
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.info("Feature importance is available for tree-based models (Random Forest, XGBoost, etc.)")

            # ==============================
            # SHAP EXPLAINABILITY SECTION
            # ==============================
            st.divider()
            st.markdown("### 🔍 SHAP Model Explainability")

            if not SHAPExplainer.is_available():
                st.warning("""
                **SHAP library not installed.** Run the following to enable explainability:
                ```bash
                pip install shap
                ```
                Then restart the app.
                """)
            elif trainer and trainer.trained_models and trainer.X_train is not None:
                shap_model_options = [
                    n for n in trainer.trained_models
                    if n not in ("LSTM", "Prophet", "ARIMA", "SARIMAX")  # SHAP not applicable for these
                ]

                if shap_model_options:
                    shap_col1, shap_col2 = st.columns([1, 3])
                    with shap_col1:
                        sel_shap_model = st.selectbox(
                            "Model to Explain:",
                            shap_model_options,
                            key="shap_model_select"
                        )
                        if st.button("🔮 Generate SHAP Explanations", type="primary", key="shap_run_btn"):
                            with st.spinner("Computing SHAP values (this may take 10-30s)..."):
                                _model_obj   = trainer.trained_models[sel_shap_model]
                                _X_train     = trainer.X_train
                                _X_test      = trainer.X_test
                                _feat_names  = list(_X_train.columns) if hasattr(_X_train, 'columns') else None
                                _task_type   = task_type

                                _explainer = SHAPExplainer(
                                    _model_obj, _X_train, _X_test,
                                    task_type=_task_type, feature_names=_feat_names
                                )
                                ok = _explainer.build_explainer()
                                if ok:
                                    st.session_state['shap_explainer'] = _explainer
                                    st.session_state['shap_model_name'] = sel_shap_model
                                    st.success("✅ SHAP values computed!")
                                else:
                                    st.error("SHAP explainer could not be built for this model.")

                    with shap_col2:
                        shap_exp = st.session_state.get('shap_explainer')
                        if shap_exp and st.session_state.get('shap_model_name') == sel_shap_model:

                            # --- SHAP Plot 1: Mean Absolute Feature Importance Bar ---
                            with st.expander("📊 SHAP Feature Importance (Mean |SHAP|)", expanded=True):
                                shap_fi_df = shap_exp.get_feature_importance_df(top_n=15)
                                if shap_fi_df is not None:
                                    import plotly.express as px
                                    fig_shap_fi = px.bar(
                                        shap_fi_df,
                                        x="SHAP Importance",
                                        y="Feature",
                                        orientation='h',
                                        title=f"SHAP Feature Importance — {sel_shap_model}",
                                        template="plotly_dark",
                                        color="SHAP Importance",
                                        color_continuous_scale="Plasma"
                                    )
                                    fig_shap_fi.update_layout(
                                        height=450,
                                        margin=dict(l=10, r=10, t=40, b=10),
                                        yaxis=dict(categoryorder='total ascending')
                                    )
                                    st.plotly_chart(fig_shap_fi, use_container_width=True)
                                    st.caption("Mean absolute SHAP value per feature. Higher = more impact on predictions.")
                                else:
                                    st.info("Could not compute SHAP feature importance.")

                            # --- SHAP Plot 2: Beeswarm (Impact Direction) ---
                            with st.expander("🐝 SHAP Beeswarm — Impact & Direction", expanded=False):
                                bee_df = shap_exp.get_beeswarm_data(top_n=12)
                                if bee_df is not None:
                                    import plotly.express as px
                                    import numpy as np
                                    # Add random jitter on y-axis to reproduce beeswarm spread
                                    # (px.strip doesn't support color_continuous_scale; px.scatter does)
                                    rng = np.random.default_rng(seed=42)
                                    bee_df = bee_df.copy()
                                    feature_order = bee_df["Feature"].unique().tolist()
                                    bee_df["_y_jitter"] = bee_df["Feature"].map(
                                        {f: i for i, f in enumerate(feature_order)}
                                    ) + rng.uniform(-0.35, 0.35, size=len(bee_df))

                                    fig_bee = px.scatter(
                                        bee_df,
                                        x="SHAP Value",
                                        y="_y_jitter",
                                        color="Feature Value (norm)",
                                        color_continuous_scale="RdBu",
                                        hover_data={"Feature": True, "SHAP Value": ":.4f",
                                                    "Feature Value": ":.4f", "_y_jitter": False},
                                        title=f"SHAP Beeswarm — {sel_shap_model}",
                                        template="plotly_dark",
                                        opacity=0.75,
                                    )
                                    fig_bee.update_traces(marker=dict(size=5))
                                    fig_bee.update_layout(
                                        height=480,
                                        margin=dict(l=10, r=10, t=40, b=10),
                                        coloraxis_colorbar=dict(title="Feature<br>Value<br>(norm)"),
                                        yaxis=dict(
                                            tickmode="array",
                                            tickvals=list(range(len(feature_order))),
                                            ticktext=feature_order,
                                            title="Feature",
                                        ),
                                        xaxis_title="SHAP Value",
                                    )
                                    fig_bee.add_vline(x=0, line_dash="dash", line_color="gray")
                                    st.plotly_chart(fig_bee, use_container_width=True)
                                    st.caption("Each dot = one test sample. Red = high feature value, Blue = low. Right of 0 = positive impact.")

                            # --- SHAP Plot 3: Waterfall (Single Prediction) ---
                            with st.expander("💧 SHAP Waterfall — Single Prediction Explanation", expanded=False):
                                n_test_samples = len(shap_exp.X_test)
                                row_idx = st.slider("Select test sample:", 0, max(0, n_test_samples - 1), 0, key="shap_waterfall_row")
                                wf_data = shap_exp.get_waterfall_data(row_index=row_idx)
                                if wf_data:
                                    import plotly.graph_objects as go
                                    contributions = wf_data["shap_contributions"]
                                    features      = wf_data["features"]
                                    base_val      = wf_data["base_value"]
                                    final_pred    = wf_data["prediction"]

                                    # Build waterfall data
                                    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in contributions]
                                    fig_wf = go.Figure(go.Waterfall(
                                        name="SHAP",
                                        orientation="h",
                                        measure=["relative"] * len(contributions) + ["total"],
                                        y=features + ["f(x)"],
                                        x=contributions + [None],
                                        connector={"line": {"color": "#64748b"}},
                                        increasing={"marker": {"color": "#ef4444"}},
                                        decreasing={"marker": {"color": "#3b82f6"}},
                                        totals={"marker": {"color": "#10b981"}},
                                        base=base_val,
                                    ))
                                    fig_wf.update_layout(
                                        title=f"Prediction Breakdown — Sample #{row_idx}<br>Base: {base_val:.3f} → Prediction: {final_pred:.3f}",
                                        template="plotly_dark",
                                        height=480,
                                        margin=dict(l=10, r=10, t=70, b=10)
                                    )
                                    st.plotly_chart(fig_wf, use_container_width=True)
                                    st.caption("Red bars push prediction higher. Blue bars push prediction lower.")

                            # --- SHAP Plot 4: Dependence Plot ---
                            with st.expander("📉 SHAP Dependence Plot", expanded=False):
                                dep_feature = st.selectbox(
                                    "Select feature:",
                                    shap_exp.feature_names,
                                    key="shap_dep_feature"
                                )
                                dep_df = shap_exp.get_dependence_data(dep_feature)
                                if dep_df is not None:
                                    import plotly.express as px
                                    fig_dep = px.scatter(
                                        dep_df,
                                        x="Feature Value",
                                        y="SHAP Value",
                                        title=f"SHAP Dependence — {dep_feature}",
                                        template="plotly_dark",
                                        color="SHAP Value",
                                        color_continuous_scale="RdBu",
                                        trendline="lowess"
                                    )
                                    fig_dep.update_layout(
                                        height=400,
                                        margin=dict(l=10, r=10, t=40, b=10)
                                    )
                                    st.plotly_chart(fig_dep, use_container_width=True)
                                    st.caption(f"How increasing '{dep_feature}' shifts model predictions.")
                        else:
                            st.info("👆 Click **'Generate SHAP Explanations'** to reveal how this model thinks.")
                else:
                    st.info("SHAP explanations are available after training models (not applicable to time-series models ARIMA/Prophet).")
            else:
                st.info("Train models first to get SHAP explanations.")

            # --- Per-Class Classification Report ---
            if task_type == "Classification" and st.session_state.results_df is not None:
                has_per_class = any(
                    "Per-Class Report" in row
                    for row in st.session_state.results_df.to_dict('records')
                    if isinstance(row.get("Per-Class Report"), dict)
                )
                if has_per_class:
                    st.divider()
                    st.markdown("### 📋 Per-Class Metrics Report")
                    for _, row in st.session_state.results_df.iterrows():
                        per_class = row.get("Per-Class Report")
                        if isinstance(per_class, dict):
                            with st.expander(f"📊 {row['Model']} — Per-Class Breakdown"):
                                report_rows = []
                                for class_name, class_metrics in per_class.items():
                                    if isinstance(class_metrics, dict):
                                        report_rows.append({
                                            "Class": str(class_name),
                                            "Precision": round(class_metrics.get("precision", 0), 4),
                                            "Recall":    round(class_metrics.get("recall",    0), 4),
                                            "F1-Score":  round(class_metrics.get("f1-score",  0), 4),
                                            "Support":   int(class_metrics.get("support",   0)),
                                        })
                                if report_rows:
                                    import plotly.express as px
                                    df_report = pd.DataFrame(report_rows)
                                    st.dataframe(df_report, use_container_width=True)
                                    fig_pcf1 = px.bar(
                                        df_report, x="Class", y="F1-Score",
                                        color="Class", template="plotly_dark",
                                        title=f"Per-Class F1 — {row['Model']}",
                                        color_discrete_sequence=px.colors.qualitative.Set2
                                    )
                                    fig_pcf1.update_layout(height=350, showlegend=False)
                                    st.plotly_chart(fig_pcf1, use_container_width=True)
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

