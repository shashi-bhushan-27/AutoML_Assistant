# AutoML Assistant

AutoML Assistant is an end-to-end machine learning workspace built with Streamlit and Python.  
It helps you upload data, preprocess features, analyze dataset quality, train multiple models, explain predictions with SHAP, and serve trained models through a FastAPI inference API.

---

## ✨ Key Capabilities

- **Workspace-based workflow** with persistent experiment state.
- **Guided 5-step UI**: Upload → Preprocess → Analyze → Train → Optimize.
- **Automated preprocessing** (imputation, encoding, scaling, splitting, optional imbalance handling).
- **Model recommendation engine** using RAG (LangChain + Groq + FAISS + local rules).
- **Multi-model training** for regression, classification, and time-series.
- **Rich evaluation outputs** with leaderboard and detailed metrics.
- **SHAP explainability** (feature importance, beeswarm, waterfall, dependence).
- **Hyperparameter optimization** with Optuna and configurable time budget.
- **FastAPI endpoints** for health checks, workspace/model discovery, and prediction.

---

## 🧱 Architecture

### Frontend
- **Streamlit app**: `/home/runner/work/AutoML_Assistant/AutoML_Assistant/shashi-bhushan-27/AutoML_Assistant/app_frontend/main_ui.py`
- Provides interactive tabs for the complete AutoML flow.

### Backend
- **Core ML pipeline**: preprocessing, analytics, model training, tuning.
- **Explainability**: SHAP wrapper for supported model types.
- **Persistence**: workspace metadata + pickled artifacts on disk.
- **Inference API**: FastAPI service in `app_backend/main_api.py`.

### Knowledge Layer
- Rules source: `knowledge_base/ml_rules.txt`
- Vector index: `knowledge_base/faiss_index/` (FAISS)

---

## 📁 Project Structure

```text
AutoML_Assistant/
├── app_frontend/
│   ├── main_ui.py
│   └── assets/style.css
├── app_backend/
│   ├── main_api.py
│   ├── model_trainer.py
│   ├── model_tuner.py
│   ├── shap_explainer.py
│   ├── preprocessing.py
│   ├── statistical_engine.py
│   ├── workspace_manager.py
│   ├── llm_rag_core.py
│   ├── report_generator.py
│   ├── code_generator.py
│   └── preprocessing_engine/
├── knowledge_base/
│   ├── ml_rules.txt
│   └── faiss_index/
├── workspaces/
├── requirements.txt
└── PROJECT_DOCUMENTATION.md
```

---

## 🛠️ Requirements

- Python 3.9+
- Pip
- Groq API key (`GROQ_API_KEY`) for LLM-powered recommendations

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (for LSTM time-series model):

```bash
pip install tensorflow
```

---

## 🔐 Environment Variables

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

---

## 🚀 Quick Start

### 1) Build the RAG vector store (first run only)

```bash
python app_backend/llm_rag_core.py
```

### 2) Start the Streamlit application

```bash
streamlit run app_frontend/main_ui.py
```

### 3) (Optional) Start the FastAPI inference server

```bash
python app_backend/main_api.py
```

- API base URL: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

---

## 🤖 Supported Modeling Modes

- **Regression**
- **Classification** (binary and multi-class)
- **Time-Series Forecasting**

Examples of model families included in the trainer/tuner stack:
- Linear models, tree ensembles, boosting, SVM, KNN
- Time-series models (Prophet, ARIMA, SARIMAX)
- Optional deep learning model (LSTM, when TensorFlow is installed)

---

## 📊 Explainability

After model training (non-time-series models), SHAP views are available from the UI:
- Feature importance (global)
- Beeswarm (sample-level impact distribution)
- Waterfall (single prediction explanation)
- Dependence plot (feature value vs contribution)

If SHAP is missing, install it from `requirements.txt` dependencies and restart the app.

---

## 🌐 FastAPI Endpoints

### Health
- `GET /health`

### Workspace Discovery
- `GET /workspaces`
- `GET /workspaces/{workspace_id}/models`

### Inference
- `POST /predict` (JSON rows)
- `POST /predict/csv/{workspace_id}/{model_name}` (CSV upload)

### Example `POST /predict` payload

```json
{
  "workspace_id": "abc12345",
  "model_name": "XGBoost",
  "data": [
    {"feature1": 1.5, "feature2": "cat_A", "feature3": 10}
  ]
}
```

---

## 💾 Persistence Model

Workspace data is stored under `workspaces/` and `data/uploads/`:
- Workspace metadata (`*.json`)
- Session state and preprocessing artifacts (`*.pkl`)
- Trained model bundles (`*_trained_models.pkl`)
- Uploaded datasets (`data/uploads/*_data.csv`)

This enables reopening prior experiments and reusing trained artifacts through both UI and API.

---

## 🧪 Validation Notes

In this repository snapshot:
- `python -m pytest` fails because `pytest` is not installed (`No module named pytest`).
- `python -m compileall app_backend app_frontend` succeeds.

---

## 📚 Additional Documentation

For a deeper feature-by-feature write-up, see:
- `PROJECT_DOCUMENTATION.md`

