"""
Report Generator Module
Generates exportable reports using Groq LLM.
"""
import os
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class ReportGenerator:
    """Generates analysis reports using Groq LLM."""
    
    def __init__(self):
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize Groq LLM."""
        try:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                temperature=0.3,
                model_name="llama-3.1-8b-instant"
            )
        except Exception as e:
            print(f"LLM initialization failed: {e}")
            self.llm = None
    
    def generate_summary_report(self, workspace_data: Dict) -> str:
        """Generate a summary report using LLM."""
        if not self.llm:
            return self._generate_fallback_report(workspace_data)

        is_multi = workspace_data.get('is_multi_output', False)
        target_display = workspace_data.get('target_col', 'Unknown')
        target_cols    = workspace_data.get('target_cols', [])

        multi_section = ""
        if is_multi:
            multi_section = f"""
        Prediction Mode: MULTI-OUTPUT (predicting {len(target_cols)} targets simultaneously)
        Target Columns: {', '.join(target_cols)}
        Note: Models were wrapped with MultiOutputRegressor / MultiOutputClassifier.
        Reported metrics (RMSE, R², Accuracy, F1) are macro-averaged across all targets.
        """
        else:
            multi_section = f"Target Variable: {target_display}"

        prompt = f"""
        Generate a professional AutoML analysis report based on the following experiment:

        Dataset: {workspace_data.get('dataset_name', 'Unknown')}
        Shape: {workspace_data.get('dataset_shape', 'Unknown')}
        Task Type: {workspace_data.get('task_type', 'Unknown')}
        {multi_section}
        Best Model: {workspace_data.get('best_model', 'N/A')}
        Best Score: {workspace_data.get('best_score', 'N/A')}

        Preprocessing Steps Applied:
        {workspace_data.get('preprocessing_steps', [])}

        Model Recommendations:
        {workspace_data.get('recommendations', [])}

        Generate a concise, professional report with:
        1. Executive Summary
        2. Data Overview
        3. Preprocessing Applied
        4. Model Performance{' (include a Multi-Output Prediction section explaining averaged metrics)' if is_multi else ''}
        5. Recommendations

        Format in Markdown.
        """

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"LLM report generation failed: {e}")
            return self._generate_fallback_report(workspace_data)
    
    def _generate_fallback_report(self, workspace_data: Dict) -> str:
        """Generate report without LLM."""
        is_multi    = workspace_data.get('is_multi_output', False)
        target_cols = workspace_data.get('target_cols', [])
        target_disp = workspace_data.get('target_col', 'Unknown')

        if is_multi:
            target_section = (
                f"- **Prediction Mode:** Multi-Output ({len(target_cols)} targets simultaneously)\n"
                + "\n".join(f"  - `{t}`" for t in target_cols)
            )
        else:
            target_section = f"- **Target Variable:** {target_disp}"

        return f"""
# AutoML Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Name:** {workspace_data.get('dataset_name', 'Unknown')}
- **Shape:** {workspace_data.get('dataset_shape', 'Unknown')}
- **Task Type:** {workspace_data.get('task_type', 'Unknown')}
{target_section}

## Best Model
- **Algorithm:** {workspace_data.get('best_model', 'N/A')}
- **Score:** {workspace_data.get('best_score', 'N/A')}
{'> ⚠️ Metrics are **macro-averaged** across all target columns.' if is_multi else ''}

## Preprocessing Steps
{self._format_preprocessing_steps(workspace_data.get('preprocessing_steps', []))}

## Recommendations
{', '.join(workspace_data.get('recommendations', ['No recommendations']))}
"""
    
    def _format_preprocessing_steps(self, steps: List[Dict]) -> str:
        """Format preprocessing steps as markdown list."""
        if not steps:
            return "- No preprocessing steps recorded"
        
        lines = []
        for step in steps[:15]:
            lines.append(f"- **{step.get('step', 'Unknown')}**: {step.get('action', '')}")
        return "\n".join(lines)
    
    def generate_code_export(self, workspace_data: Dict) -> str:
        """Generate reproducible Python code for Kaggle/Colab."""
        dataset_name = workspace_data.get('dataset_name', 'dataset.csv')
        target_col = workspace_data.get('target_col', 'target')
        task_type = workspace_data.get('task_type', 'classification')
        best_model = workspace_data.get('best_model', 'Random Forest')
        
        # Map model names to sklearn imports
        model_imports = {
            "Random Forest": ("sklearn.ensemble", "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor"),
            "XGBoost": ("xgboost", "XGBClassifier" if task_type == "classification" else "XGBRegressor"),
            "Gradient Boosting": ("sklearn.ensemble", "GradientBoostingClassifier" if task_type == "classification" else "GradientBoostingRegressor"),
            "Logistic Regression": ("sklearn.linear_model", "LogisticRegression"),
            "Linear Regression": ("sklearn.linear_model", "LinearRegression"),
        }
        
        model_info = model_imports.get(best_model, ("sklearn.ensemble", "RandomForestClassifier"))
        
        code = f'''# AutoML Generated Code
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Task: {task_type.title()}
# Best Model: {best_model}

# ===== INSTALLATION =====
# !pip install pandas scikit-learn xgboost

# ===== IMPORTS =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from {model_info[0]} import {model_info[1]}

# ===== LOAD DATA =====
# Update path to your dataset
df = pd.read_csv("{dataset_name}")
print(f"Dataset shape: {{df.shape}}")

# ===== PREPROCESSING =====
# Separate features and target
X = df.drop(columns=["{target_col}"])
y = df["{target_col}"]

# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Handle missing values
X = X.fillna(X.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===== MODEL TRAINING =====
model = {model_info[1]}()
model.fit(X_train, y_train)

# ===== EVALUATION =====
y_pred = model.predict(X_test)
'''
        
        if task_type == "classification":
            code += '''
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score:.4f}")
'''
        else:
            code += '''
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
'''
        
        code += '''
# ===== SAVE MODEL =====
import pickle
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to trained_model.pkl")
'''
        
        return code
