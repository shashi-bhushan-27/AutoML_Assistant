# AutoML Assistant - Project Documentation

**Version:** 3.0 (High-Priority Feature Release)
**Last Updated:** 2026-02-21

---

## 1. Project Overview

**AutoML Assistant** is an intelligent, user-friendly SaaS platform designed to democratize data science. It enables users—regardless of their coding expertise—to upload raw datasets, automatically preprocess them, discover insights through AI-driven analysis, and train high-performance machine learning models in minutes.

**Core Value Proposition:** *"Upload, Clean, Analyze, Train, Explain, and Deploy — in minutes."*

---

## 2. Technical Architecture

The application follows a modular architecture separating the interactive frontend from the heavy-lifting computational backend.

- **Frontend**: Built with **Streamlit** (Python), providing a responsive, dashboard-style interface with interactive Plotly charts.
- **Backend**: A robust data processing engine powered by **Pandas**, **NumPy**, **Scikit-learn**, **XGBoost**, and **TensorFlow (optional)**.
- **REST API**: A **FastAPI** server (`app_backend/main_api.py`) exposing trained models as live prediction endpoints — enabling external applications to call trained models over HTTP.
- **AI Intelligence**: Integrated **LangChain** + **Groq (Llama 3.1)** RAG (Retrieval-Augmented Generation) system. It uses a semantic FAISS knowledge base (`ml_rules.txt`) to interpret dataset statistics and recommend optimal modeling strategies.
- **Explainability Engine**: A **SHAP**-powered module (`app_backend/shap_explainer.py`) that generates model-agnostic explanations using TreeExplainer, LinearExplainer, or KernelExplainer depending on the model type.
- **Persistence**: Custom JSON/Pickle-based system (`app_backend/workspace_manager.py`) that ensures workspaces, datasets, and training results are saved and restorable across sessions.

---

## 3. Detailed Feature Breakdown

### 📂 Workspace Management
- **Sessions**: Users can create isolated workspaces for different experiments.
- **Persistence**: State is automatically saved. Reloading a workspace restores the exact state of the dataset, preprocessing steps, analysis results, and trained models.
- **CRUD**: Full Create, Read, Delete capabilities for workspaces.

### 🔌 Data Ingestion
- **Upload**: Drag-and-drop CSV support with automatic encoding detection (`chardet`).
- **Validation**: Automatic structural checks and parsing.
- **Feedback**: Immediate visual confirmation of uploaded data shape and preview.

### 🔧 Preprocessing Engine (`AutoPreprocessor`)
- **Smart Type Detection**: Automatically identifies Numerical, Categorical, and DateTime columns.
- **Time-Series Support**: Toggleable mode for temporal data handling.
- **Handling Imbalance**: Optional **SMOTE** application for classification tasks.
- **Automated Pipeline**:
  1. Missing Value Imputation
  2. Outlier Removal/Capping
  3. Encoding (One-Hot/Label)
  4. Scaling (Standard/MinMax)
- **Visualizations**: Feature distribution histograms, Correlation heatmap, Class balance pie chart.

### 🤖 AI Analysis & Recommendations
- **Statistical Profiling**: Generates extensive descriptive statistics (Correlation, Skewness, Class balance, Stationarity tests).
- **RAG Advisor**: The system sends dataset profiles to the **Groq Llama 3.1** model, which queries a local knowledge base of ML best practices to suggest the best algorithms.
- **Meta-Learning**: Checks past successful experiments to bias recommendations towards proven winners.
- **Intelligent Auto-Selection**: Recommendations are fuzzy-matched against the available model list to automatically pre-select the best options.
- **Explainability**: "Why This Model?" panel shows the AI's reasoning in plain language.

### 🚀 Model Training & Evaluation
- **Task Support**: Regression, Classification (binary & multi-class), and Time-Series Forecasting.
- **Algorithms**:
  - **Standard**: Linear/Logistic Regression, Ridge, Lasso, ElasticNet, Random Forest, XGBoost, Gradient Boosting, AdaBoost, Extra Trees, SVM, KNN, Decision Trees.
  - **Time-Series**: Prophet, ARIMA, SARIMAX, **LSTM** (new — requires TensorFlow).
- **Rich Evaluation Metrics**:
  - **Regression**: RMSE, MAE, R², MAPE (%)
  - **Classification (Binary)**: Accuracy, F1, Precision, Recall, MCC, Cohen Kappa, AUC-ROC
  - **Classification (Multi-class)**: Accuracy, Weighted F1/Precision/Recall, MCC, Cohen Kappa, AUC-ROC (One-vs-Rest), Per-class F1/Precision/Recall/Support table
- **Leaderboard**: Automatically ranks models. Highlights best model.
- **Per-Model Downloads**: Download any trained model as a `.pkl` file directly from the leaderboard.
- **Detailed Charts**: Confusion Matrix, ROC Curves, Radar Chart, Predicted vs. Actual, Residuals, Training Time comparison, Feature Importance.

### 🔍 SHAP Explainability (NEW — High Priority)
A dedicated explainability engine powered by the **SHAP** library. Available after training any non-time-series model.

- **SHAP Feature Importance Bar**: Mean absolute SHAP values per feature — which features matter most globally.
- **SHAP Beeswarm Plot**: Shows the direction and magnitude of each feature's impact across all test samples. Red = high feature value, Blue = low.
- **SHAP Waterfall Plot**: Explains a single prediction step-by-step — shows exactly why the model made *that* specific prediction.
- **SHAP Dependence Plot**: Shows how a specific feature's value shifts model predictions, with a LOWESS trendline.
- **Auto-Explainer Selection**: Uses `TreeExplainer` for tree/ensemble models, `LinearExplainer` for regression/logistic models, and `KernelExplainer` as a universal fallback.

### 🌐 REST API (`/predict` Endpoint) (NEW — High Priority)
A **FastAPI** server that serves trained workspace models over HTTP.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check |
| `/workspaces` | GET | List all workspaces and their trained models |
| `/workspaces/{id}/models` | GET | List models for a specific workspace |
| `/predict` | POST | JSON-based inference (single row or batch) |
| `/predict/csv/{workspace_id}/{model_name}` | POST | CSV-file-based batch inference |

- Automatically applies saved preprocessor before inference.
- Returns prediction probabilities for classification models.
- Start with: `python app_backend/main_api.py` → API at `http://localhost:8000`
- Interactive docs at: `http://localhost:8000/docs`

### ⚡ Hyperparameter Optimization (Optuna)
- **Bayesian Optimization**: Uses Optuna to intelligently search the hyperparameter space.
- **Time-budget control**: "Spend 60 seconds" instead of "run N iterations".
- **Cross-Validation**: Configurable K-fold validation during tuning.
- **Export**: Download the tuned model + reproducible Python training code (Colab-ready).

---

## 4. Recent Improvements History

### ✅ v3.0 — High-Priority Feature Release (2026-02-21)
Breaking new capabilities on the `feature/high-priority-improvements` branch:

#### 🔮 Task 1: FastAPI `/predict` Endpoint
- Wired up the previously empty `app_backend/main_api.py` into a full REST API.
- Exposes `/predict` (JSON) and `/predict/csv` (file upload) for live inference.
- Loads models and preprocessors directly from workspace storage.

#### 🎯 Task 2: SHAP Explainability
- New `app_backend/shap_explainer.py` module with 4 Plotly-compatible chart types.
- Added "🔍 SHAP Model Explainability" section to the Training tab in the UI.
- Gracefully degrades if `shap` is not installed.

#### 📊 Task 3: Enhanced Multi-Class Metrics
- `evaluate()` in `model_trainer.py` now computes: **MCC**, **Cohen Kappa**, **AUC-ROC (OvR)**, **MAPE**, per-class F1 report.
- UI shows per-class breakdown table and F1 bar chart per model.
- Binary vs. multi-class is handled automatically.

#### 🧠 Task 4: LSTM Deep Learning for Time-Series
- 2-layer Keras LSTM (64→32 units) with Dropout and EarlyStopping.
- Walk-forward forecasting (no data leakage).
- Auto-normalizes target values; appears in model list only if TensorFlow is installed.

#### 💾 Bonus: Per-Model Download Buttons
- Every trained model now has an inline download button on the Leaderboard.

---

### ✅ v2.0 — Visualizations & Documentation (2026-02-15)
- Added preprocessing visualizations: Feature distribution histograms, Correlation heatmap, Class balance chart.
- Added training/evaluation charts: Confusion Matrix, ROC Curves, Radar Chart, Predicted vs. Actual, Residuals, Feature Importance.
- Added "Docs & Help" tab with step-by-step guide and troubleshooting.
- Bumped version to 2.0.

### ✅ v1.5 — AI Auto-Selection & Meta-Learning (2026-01-22)
- Auto-selects AI-recommended models in the training UI.
- Meta-learning: Uses past experiments to bias future recommendations.
- Explainability layer: "Why This Model?" reasoning panel.

### ✅ v1.2 — Workspace Persistence Fix (2026-01-18)
- Fixed CSV files disappearing on workspace reload.
- Added workspace deletion button.
- "✅ Using stored dataset" banner.

### ✅ v1.0 — Groq Migration (2026-01-22)
- Migrated AI provider from Google Gemini to Groq (Llama 3.1).
- Implemented fuzzy matching for AI model name recommendations.

---

## 5. File Structure

```
📂 AutoML_SaaS_Project/
├── 📂 app_frontend/
│   ├── main_ui.py                  # Main Streamlit Dashboard (5-tab workflow)
│   └── assets/
│       └── style.css               # Custom dark-mode CSS
├── 📂 app_backend/
│   ├── main_api.py                 # ⭐ FastAPI REST API (/predict, /health)
│   ├── model_trainer.py            # ⭐ Model training + enhanced metrics + LSTM
│   ├── model_tuner.py              # Optuna hyperparameter tuning
│   ├── shap_explainer.py           # ⭐ SHAP explainability module (NEW)
│   ├── workspace_manager.py        # JSON/Pickle workspace persistence
│   ├── statistical_engine.py       # Dataset profiling & statistics
│   ├── llm_rag_core.py             # LangChain + Groq + FAISS RAG
│   ├── report_generator.py         # LLM-based report & code export
│   ├── code_generator.py           # Reproducible Python code generator
│   ├── preprocessing.py            # AdvancedPreprocessor entry point
│   └── preprocessing_engine/       # Modular preprocessing sub-system
│       ├── engine.py               # AutoPreprocessor orchestrator
│       ├── imputer.py              # Missing value strategies
│       ├── encoder.py              # Categorical encoding
│       ├── scaler.py               # Feature scaling
│       ├── transformer.py          # Feature transformations
│       ├── selector.py             # Feature selection
│       ├── balancer.py             # SMOTE / class balancing
│       ├── profiler.py             # Data profiling
│       ├── ingestion.py            # CSV loading & validation
│       └── splitter.py             # Train/test splitting
├── 📂 knowledge_base/
│   ├── ml_rules.txt                # ML best practices knowledge base
│   └── faiss_index/                # Vector store (auto-generated)
├── 📂 workspaces/                  # Saved user sessions (JSON + Pickle)
├── 📂 data/uploads/                # Persisted raw datasets
├── 📂 tests/                       # Unit tests
├── .env                            # API keys (GROQ_API_KEY, HF_TOKEN)
├── requirements.txt                # Python dependencies
└── Dockerfile                      # Container deployment
```

---

## 6. Setup & Running

### Prerequisites
- Python 3.9+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. [Optional] Enable LSTM support
pip install tensorflow

# 4. Configure API keys
# Create .env file:
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here

# 5. Build the knowledge base (first time only)
python app_backend/llm_rag_core.py

# 6. Run the Streamlit UI
streamlit run app_frontend/main_ui.py

# 7. [Optional] Run the FastAPI prediction server
python app_backend/main_api.py
# API available at: http://localhost:8000
# Swagger docs at: http://localhost:8000/docs
```

### Docker Deployment

```bash
docker build -t automl-assistant .
docker run -p 8501:8501 --env-file .env automl-assistant
```

---

## 7. API Reference (FastAPI)

### Example: Predict via JSON

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "abc12345",
    "model_name": "XGBoost",
    "data": [
      {"feature1": 1.5, "feature2": "cat_A", "feature3": 10}
    ]
  }'
```

**Response:**
```json
{
  "workspace_id": "abc12345",
  "model_name": "XGBoost",
  "predictions": [1],
  "prediction_labels": [1],
  "probabilities": [[0.12, 0.88]]
}
```

### Example: Batch Predict via CSV

```bash
curl -X POST "http://localhost:8000/predict/csv/abc12345/XGBoost" \
  -F "file=@my_test_data.csv"
```

---

## 8. Supported Models Reference

| Model | Task | Notes |
|-------|------|-------|
| Linear Regression | Regression | Fast, interpretable |
| Ridge / Lasso / ElasticNet | Regression | Regularized linear models |
| Logistic Regression | Classification | Good for low-dimensional data |
| Random Forest | Both | Versatile, handles mixed features |
| XGBoost | Both | Best general-purpose (recommended) |
| Gradient Boosting | Both | Strong boosting ensemble |
| AdaBoost | Both | Adaptive boosting |
| Extra Trees | Both | Faster alternative to Random Forest |
| Decision Tree | Both | Interpretable, single tree |
| SVM | Both | Effective for high-dimensional data |
| KNN | Both | Simple, similarity-based |
| Prophet | Time-Series | Seasonality + holidays |
| ARIMA | Time-Series | Stationary univariate forecasting |
| SARIMAX | Time-Series | Seasonal + exogenous variables |
| **LSTM** ⭐ | Time-Series | Deep learning (requires TensorFlow) |

---

## 9. Metrics Reference

### Regression Metrics
| Metric | Description | Goal |
|--------|-------------|------|
| RMSE | Root Mean Squared Error | ↓ Lower is better |
| MAE | Mean Absolute Error | ↓ Lower is better |
| R² | Coefficient of Determination | ↑ Higher is better (max 1.0) |
| MAPE (%) | Mean Absolute Percentage Error | ↓ Lower is better |

### Classification Metrics
| Metric | Description | Goal |
|--------|-------------|------|
| Accuracy | % correct predictions | ↑ Higher is better |
| F1 Score | Harmonic mean of Precision & Recall | ↑ Higher is better |
| Precision | True Positives / Predicted Positives | ↑ Higher is better |
| Recall | True Positives / Actual Positives | ↑ Higher is better |
| **MCC** ⭐ | Matthews Correlation Coefficient (-1 to 1) | ↑ Best for imbalanced data |
| **Cohen Kappa** ⭐ | Agreement beyond chance (-1 to 1) | ↑ Higher is better |
| **AUC-ROC** ⭐ | Area Under ROC Curve (OvR for multi-class) | ↑ Higher is better |

---

*AutoML Assistant v3.0 · Built with Streamlit, Scikit-learn, XGBoost, SHAP, FastAPI & Groq AI*
