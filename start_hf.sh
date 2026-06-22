#!/bin/bash
# Start the FastAPI backend in the background
uvicorn app_backend.main_api:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend in the foreground on port 7860 (Hugging Face default)
streamlit run app_frontend/main_ui.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false

