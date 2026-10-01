"""Cached backend services and the sample datasets offered on the Workspaces page."""
import os
from typing import Dict, List

import pandas as pd
import streamlit as st

from app_backend.llm_rag_core import ModelAdvisor, check_llm_health, get_llm_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLES = [
    {"key": "adult", "name": "Adult income", "path": os.path.join(ROOT, "samples", "adult_income_sample.csv"),
     "target": "class", "about": "Classification. 4,000-row stratified sample of UCI Adult (OpenML 1590)."},
    {"key": "california", "name": "California housing",
     "path": os.path.join(ROOT, "samples", "california_housing_sample.csv"), "target": "MedHouseVal",
     "about": "Regression. 3,000-row sample of the scikit-learn California Housing data (target in $100k)."},
    {"key": "ev", "name": "EV charging & grid optimisation",
     "path": os.path.join(ROOT, "app_backend", "EV_Charging_Grid_Optimization_Categorical.csv"), "target": None,
     "about": "Local sample file (not part of the repository)."},
]


def available_samples() -> List[Dict]:
    return [s for s in SAMPLES if os.path.exists(s["path"])]


@st.cache_data(ttl=300, show_spinner=False)
def llm_health(model: str = None) -> Dict:
    return check_llm_health(model or get_llm_model())


@st.cache_resource(show_spinner="Loading the retrieval index and embedding model...")
def advisor(model: str) -> ModelAdvisor:
    return ModelAdvisor(model_name=model)


@st.cache_data(show_spinner=False)
def sample_shape(path: str):
    return pd.read_csv(path).shape
