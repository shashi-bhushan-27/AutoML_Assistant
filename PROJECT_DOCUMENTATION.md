# AutoML Assistant - Project Documentation

**Version:** 1.0 (Groq-Enabled)
**Last Updated:** 2026-01-22

---

## 1. Project Overview

**AutoML Assistant** is an intelligent, user-friendly SaaS platform designed to democratize data science. It enables users—regardless of their coding expertise—to upload raw datasets, automatically preprocess them, discover insights through AI-driven analysis, and train high-performance machine learning models in minutes.

**Core Value Proposition:** "Upload, Clean, Analyze, and Train in minutes."

---

## 2. Technical Architecture

The application follows a modular architecture separating the interactive frontend from the heavy-lifting computational backend.

-   **Frontend**: Built with **Streamlit** (Python), providing a responsive, dashboard-style interface.
-   **Backend**: A robust data processing engine powered by **Pandas**, **Numpy**, and **Scikit-learn**.
-   **AI Intelligence**: Integrated **LangChain** + **Groq (Llama 3.1)** RAG (Retrieval-Augmented Generation) system. It uses a semantic knowledge base (`ml_rules.txt`) to interpret dataset statistics and recommend optimal modeling strategies.
-   **Persistence**: Custom JSON/Pickle-based system (`app_backend/workspace_manager.py`) that ensures workspaces, datasets, and training results are saved and restorable across sessions.

---

## 3. Detailed Feature Breakdown

### 📂 Workspace Management
*   **Sessions**: Users can create isolated workspaces for different experiments.
*   **Persistence**: State is automatically saved. Reloading a workspace restores the exact state of the dataset, preprocessing steps, and analysis results (recently improved to show loaded data availability clearly).
*   **CRUD**: Full Create, Read, Delete capabilities for workspaces.

### 🔌 Data Ingestion
*   **Upload**: Drag-and-drop CSV support.
*   **Validation**: Automatic structural checks and parsing.
*   **Feedback**: Immediate visual confirmation of uploaded data shape and preview.

### 🔧 Preprocessing Engine (`AutoPreprocessor`)
*   **Smart Type Detection**: Automatically identifies Numerical, Categorical, and DateTime columns.
*   **Time-Series Support**: Toggleable mode for temporal data handling.
*   **Handling Imbalance**: Optional **SMOTE** application for classification tasks.
*   **Automated Pipeline**: 
    1.  Missing Value Imputation
    2.  Outlier Removal/Capping
    3.  Encoding (One-Hot/Label)
    4.  Scaling (Standard/MinMax)

### 🤖 AI Analysis & Recommendations
*   **Statistical Profiling**: Generates extensive descriptive statistics (Correlation, Skewness, Class balance).
*   **RAG Advisor**: The system sends dataset profiles to the **Groq Llama 3.1** model, which queries a local knowledge base of ML best practices to suggest the best algorithms (e.g., "Use XGBoost because dataset has mixed features...").
*   **Intelligent Auto-Selection**: Recommendations are fuzzy-matched against the available model list to automatically pre-select the best options in the UI.

### 🚀 Model Training & Optimization
*   **Task Support**: Regression, Classification, and Time-Series Forecasting.
*   **Algorithms**:
    *   **Standard**: Linear/Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM, KNN, Decision Trees.
    *   **Time-Series**: Prophet, ARIMA, SARIMAX.
*   **Visualization**: Live progress bars and status updates.
*   **Evaluation**:
    *   **Leaderboard**: Automatically ranks models by RMSE (Regression) or Accuracy (Classification).
    *   **Charts**: Interactive Plotly bar charts comparing model performance.
*   **Optimization**: Hyperparameter tuner using Cross-Validation to squeeze extra performance from selected models.
*   **Export**: Download trained models as `.pkl` files for deployment.

---

## 4. Recent Improvements ("What We Did")

We have actively improved the system stability and user experience. Here is a summary of recent changes:

### ✅ 1. Migration to Groq API
Switched the underlying AI provider from Google Gemini to **Groq** for faster, low-latency inference. Updated `llm_rag_core.py` to use `ChatGroq` with Llama-3.1.

### ✅ 2. Fixed AI Model Auto-Selection
**Problem**: The AI would recommend "XGBoost Classifier" or "Random Forest", but the UI checkbox expected "XGBoost" or "Random Forest Classifier" (exact string mismatch), leading to no models being selected by default.
**Solution**:
*   Updated the Backend to feed the *exact* list of supported models to the AI prompt.
*   Implemented a **Fuzzy Matching** algorithm in the Frontend. It now intelligently matches recommendations (e.g., "randomforest") to the correct UI option ("Random Forest"), ensuring the "Start Training" button is ready to go immediately after analysis.

### ✅ 3. Fixed Workspace Persistence
**Problem**: Users reported that uploaded CSV files "disappeared" when reopening a workspace.
**Solution**:
*   Verified backend integrity (data was safe).
*   Updated the UI logic in `main_ui.py`. The system now explicitly checks for a stored dataset even if the file uploader widget is empty (which resets on reload).
*   Added a clear banner: **"✅ Using stored dataset: [Filename]"** to verify data presence.

### ✅ 4. Enhanced UX
*   **Data Split Visibility**: Added safeguards to display the exact number of Training vs. Testing samples before training starts.
*   **State Management**: Improved session state clearing when re-running analysis to preventing stale recommendations from lingering.

### 🌟 5. Major Feature Upgrades (The "High Impact" Tier)
We have implemented three game-changing features to elevate the platform:

#### 🔥 True Automated Hyperparameter Optimization (Optuna)
Replaced the basic random search with **Optuna**, a Bayesian optimization framework.
-   **Old Way**: "Run 50 iterations" (blind guessing).
-   **New Way**: "Spend 60 seconds optimizing" (time-budget based).
-   **Benefit**: The system intelligently explores the parameter space, "pruning" bad trials early and finding better models faster.

#### 🧠 Learning Workspace (Meta-Learning)
The system now has "memory". It looks at your past successful experiments to improve future recommendations.
-   **How it works**: When you analyze a dataset, the system checks if it has processed similar data before (based on size, task, and complexity).
-   **Impact**: If `XGBoost` won last time, the AI will bias its recommendation towards `XGBoost` for similar new tasks.
-   **Indicator**: Look for the **"💡 Meta-Learning Active"** badge in the Analysis tab.

#### 🎓 Explainability Layer ("Why This Model?")
We added a reasoning engine to the AI Advisor.
-   **Feature**: A new **"Why This Model?"** panel in the Analysis step.
-   **Detail**: Instead of just listing models, the AI explains its logic (e.g., *"Selected Random Forest because your data contains outliers which tree-based models handle well"*).
-   **Trust**: Transforms the AI from a black box into a transparent partner.

---

## 5. File Structure
```
📂 /
├── 📂 app_frontend/
│   └── main_ui.py            # Main Streamlit Dashboard
├── 📂 app_backend/
│   ├── preprocessing_engine/ # Data cleaning logic
│   ├── model_trainer.py      # Scikit-learn/XGBoost wrappers
│   ├── workspace_manager.py  # JSON storage & persistence
│   └── llm_rag_core.py       # LangChain + Groq Integration
├── 📂 knowledge_base/        # ML Rules text file & Vector Store
├── 📂 workspaces/            # Saved user sessions (JSON)
└── 📂 data/uploads/          # Persisted raw datasets
```
