
import json

def generate_training_code(dataset_name, target_col, model_name, best_params=None, task_type="Regression"):
    """
    Generates a standalone Python script for training the model.
    """
    
    # Handle potentially numpy types in params
    if best_params:
        try:
            # Simple sanitization
            best_params = str(best_params).replace("nan", "None")
        except:
            best_params = "{}"
    else:
        best_params = "{}"

    # Template for the code
    code_template = f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# --- 1. LOAD DATA ---
# Replace with your actual dataset path
dataset_path = "{dataset_name}" 
print(f"Loading dataset from {{dataset_path}}...")
# df = pd.read_csv(dataset_path) 
# For demo purposes, we assume df is loaded. 
# In Colab, you would upload the file and read it:
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv(next(iter(uploaded)))

# --- 2. PREPROCESSING ---
target_col = "{target_col}"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. MODEL INITIALIZATION ---
model_name = "{model_name}"
best_params = {best_params}

print(f"Initializing {{model_name}} with params: {{best_params}}")

"""

    # Model specific imports and init
    model_init = ""
    if "Random Forest" in model_name:
        if task_type == "Regression":
            model_init = "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(**best_params)"
        else:
            model_init = "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(**best_params)"
            
    elif "XGBoost" in model_name:
        model_init = "import xgboost as xgb\n"
        if task_type == "Regression":
            model_init += "model = xgb.XGBRegressor(**best_params)"
        else:
            model_init += "model = xgb.XGBClassifier(**best_params)"
            
    elif "Linear Regression" in model_name:
        model_init = "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression(**best_params)"
    
    elif "Logistic Regression" in model_name:
        model_init = "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(**best_params)"
        
    elif "SVM" in model_name:
        if task_type == "Regression":
            model_init = "from sklearn.svm import SVR\nmodel = SVR(**best_params)"
        else:
            model_init = "from sklearn.svm import SVC\nmodel = SVC(**best_params)"
            
    elif "Gradient Boosting" in model_name:
        if task_type == "Regression":
            model_init = "from sklearn.ensemble import GradientBoostingRegressor\nmodel = GradientBoostingRegressor(**best_params)"
        else:
            model_init = "from sklearn.ensemble import GradientBoostingClassifier\nmodel = GradientBoostingClassifier(**best_params)"
            
    else:
        # Fallback
        model_init = "# Model import logic for specific type goes here\n# model = GenericModel(**best_params)"

    code_template += model_init + "\n\n"
    
    code_template += """
# --- 4. TRAINING PIPELINE ---
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

print("Training model...")
clf.fit(X_train, y_train)

# --- 5. EVALUATION ---
print("Evaluating...")
score = clf.score(X_test, y_test)
print(f"Test Score: {score:.4f}")

# Example Prediction
# preds = clf.predict(X_test.iloc[:5])
# print("Predictions:", preds)
"""

    return code_template
