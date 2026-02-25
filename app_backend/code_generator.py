"""
Code Generator — AutoML Assistant
Generates a standalone, reproducible Python training script that faithfully
mirrors every preprocessing step applied by the AutoPreprocessor engine,
followed by model training and evaluation.

Supports single-output and multi-output (MultiOutputRegressor / Classifier).
"""

import json
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj):
    """Convert params to a clean JSON-serialisable dict."""
    if not obj:
        return {}
    try:
        return json.loads(
            json.dumps(obj, default=lambda o: int(o) if isinstance(o, np.integer)
                                              else float(o) if isinstance(o, np.floating)
                                              else str(o))
        )
    except Exception:
        return {}


def _normalise_targets(target_col):
    """Always return a list of target column names."""
    if isinstance(target_col, (list, tuple)):
        return [str(c) for c in target_col if c]
    return [str(target_col)] if target_col else []


def _model_import_and_init(model_name: str, task_type: str, best_params: dict,
                           is_multi_output: bool = False) -> str:
    """Return the import + instantiation lines for the requested model."""
    p = best_params or {}
    params_str = json.dumps(p, indent=4) if p else "{}"

    lines = [f"best_params = {params_str}", ""]

    mn = model_name  # shorthand
    tt = (task_type or "Regression").lower()

    if "Random Forest" in mn:
        cls = "RandomForestRegressor" if tt == "regression" else "RandomForestClassifier"
        lines += [f"from sklearn.ensemble import {cls}",
                  f"base_model = {cls}(**best_params)"]
    elif "XGBoost" in mn:
        cls = "XGBRegressor" if tt == "regression" else "XGBClassifier"
        lines += ["import xgboost as xgb",
                  f"base_model = xgb.{cls}(**best_params)"]
    elif "Linear Regression" in mn:
        lines += ["from sklearn.linear_model import LinearRegression",
                  "base_model = LinearRegression(**best_params)"]
    elif "Logistic Regression" in mn:
        lines += ["from sklearn.linear_model import LogisticRegression",
                  "base_model = LogisticRegression(max_iter=1000, **best_params)"]
    elif "Ridge" in mn:
        lines += ["from sklearn.linear_model import Ridge",
                  "base_model = Ridge(**best_params)"]
    elif "Lasso" in mn:
        lines += ["from sklearn.linear_model import Lasso",
                  "base_model = Lasso(**best_params)"]
    elif "ElasticNet" in mn:
        lines += ["from sklearn.linear_model import ElasticNet",
                  "base_model = ElasticNet(**best_params)"]
    elif "SVM" in mn:
        cls = "SVR" if tt == "regression" else "SVC"
        lines += [f"from sklearn.svm import {cls}",
                  f"base_model = {cls}(**best_params)"]
    elif "Gradient Boosting" in mn:
        cls = "GradientBoostingRegressor" if tt == "regression" else "GradientBoostingClassifier"
        lines += [f"from sklearn.ensemble import {cls}",
                  f"base_model = {cls}(**best_params)"]
    elif "AdaBoost" in mn:
        cls = "AdaBoostRegressor" if tt == "regression" else "AdaBoostClassifier"
        lines += [f"from sklearn.ensemble import {cls}",
                  f"base_model = {cls}(**best_params)"]
    elif "Extra Trees" in mn:
        cls = "ExtraTreesRegressor" if tt == "regression" else "ExtraTreesClassifier"
        lines += [f"from sklearn.ensemble import {cls}",
                  f"base_model = {cls}(**best_params)"]
    elif "Decision Tree" in mn:
        cls = "DecisionTreeRegressor" if tt == "regression" else "DecisionTreeClassifier"
        lines += [f"from sklearn.tree import {cls}",
                  f"base_model = {cls}(**best_params)"]
    elif "KNN" in mn or "K-Nearest" in mn:
        cls = "KNeighborsRegressor" if tt == "regression" else "KNeighborsClassifier"
        lines += [f"from sklearn.neighbors import {cls}",
                  f"base_model = {cls}(**best_params)"]
    else:
        lines += [f"# ⚠️  No specific import found for '{mn}'. Add it manually.",
                  "# base_model = YourModelClass(**best_params)",
                  "base_model = None  # REPLACE THIS"]

    # Wrap in MultiOutput if needed
    if is_multi_output:
        wrapper = "MultiOutputRegressor" if tt == "regression" else "MultiOutputClassifier"
        lines += [
            "",
            f"# Multi-output wrapper — trains one base model per target column",
            f"from sklearn.multioutput import {wrapper}",
            f"model = {wrapper}(base_model)",
        ]
    else:
        lines += ["", "model = base_model"]

    return "\n".join(lines)


def _eval_block(task_type: str, is_multi_output: bool = False,
                target_cols: list = None) -> str:
    """Return the evaluation code block."""
    tt = (task_type or "Regression").lower()

    if is_multi_output and target_cols:
        cols_repr = repr(target_cols)
        if tt == "regression":
            return f"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test)
TARGET_COLS = {cols_repr}

print("  Per-target metrics:")
rmse_list, mae_list, r2_list = [], [], []
for i, col in enumerate(TARGET_COLS):
    yt = np.array(y_test)[:, i] if hasattr(y_test, 'iloc') else y_test[:, i]
    yp = y_pred[:, i]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    rmse_list.append(rmse); mae_list.append(mae); r2_list.append(r2)
    print(f"    {{col}}: RMSE={{rmse:.4f}}  MAE={{mae:.4f}}  R²={{r2:.4f}}")

print(f"\\n  Averaged across {len(target_cols)} targets:")
print(f"    Avg RMSE : {{np.mean(rmse_list):.4f}}")
print(f"    Avg MAE  : {{np.mean(mae_list):.4f}}")
print(f"    Avg R²   : {{np.mean(r2_list):.4f}}")
"""
        else:
            return f"""
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

y_pred = model.predict(X_test)
TARGET_COLS = {cols_repr}

print("  Per-target metrics:")
acc_list, f1_list = [], []
for i, col in enumerate(TARGET_COLS):
    yt = np.array(y_test)[:, i] if hasattr(y_test, 'iloc') else y_test[:, i]
    yp = y_pred[:, i]
    acc = accuracy_score(yt, yp)
    f1  = f1_score(yt, yp, average='weighted', zero_division=0)
    acc_list.append(acc); f1_list.append(f1)
    print(f"    {{col}}: Accuracy={{acc:.4f}}  F1={{f1:.4f}}")

print(f"\\n  Averaged across {len(target_cols)} targets:")
print(f"    Avg Accuracy : {{np.mean(acc_list):.4f}}")
print(f"    Avg F1 Score : {{np.mean(f1_list):.4f}}")
"""

    # ── Single-output ──────────────────────────────────────────────────────
    if tt == "regression":
        return """
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test)
rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)
mape  = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

print(f"  RMSE  : {rmse:.4f}")
print(f"  MAE   : {mae:.4f}")
print(f"  R²    : {r2:.4f}")
print(f"  MAPE  : {mape:.2f}%")
"""
    else:
        return """
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)

y_pred = model.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average='weighted')
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"  Accuracy  : {acc:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
"""


# ---------------------------------------------------------------------------
# Public API — main function called from the UI
# ---------------------------------------------------------------------------

def generate_training_code(
    dataset_name: str,
    target_col,                 # str OR list[str]
    model_name: str,
    best_params: dict = None,
    task_type: str = "Regression",
    preprocessor=None           # AutoPreprocessor instance (fitted)
) -> str:
    """
    Generate a standalone, end-to-end Python training script.

    Parameters
    ----------
    dataset_name  : Name / path of the original CSV.
    target_col    : Target column name (str) OR list of names (multi-output).
    model_name    : Model name string (e.g. "XGBoost").
    best_params   : Best hyperparameters dict (from Optuna tuning).
    task_type     : "Regression" or "Classification".
    preprocessor  : Fitted AutoPreprocessor instance.
    """
    best_params = _safe_json(best_params)

    # Normalise target_col → always a list internally
    target_cols   = _normalise_targets(target_col)
    is_multi      = len(target_cols) > 1
    target_display = ", ".join(target_cols)

    # ------------------------------------------------------------------
    # SECTION 1 — Header + imports
    # ------------------------------------------------------------------
    multi_note = f" ({len(target_cols)} targets)" if is_multi else ""
    script = f'''\
# =============================================================================
#  AutoML Assistant — Auto-Generated Training Script
#  Model      : {model_name}
#  Task       : {task_type}{"  [MULTI-OUTPUT]" if is_multi else ""}
#  Target(s)  : {target_display}{multi_note}
#  Dataset    : {dataset_name}
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
# Replace the path below with your actual CSV file path.
# In Google Colab:
#   from google.colab import files
#   uploaded = files.upload()
#   df = pd.read_csv(next(iter(uploaded)))

dataset_path = "{dataset_name}"
df = pd.read_csv(dataset_path)
print(f"Loaded {{len(df)}} rows × {{len(df.columns)}} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SEPARATE TARGET{"S" if is_multi else ""}
# ─────────────────────────────────────────────────────────────────────────────
'''

    if is_multi:
        cols_repr = repr(target_cols)
        script += f'''TARGET_COLS = {cols_repr}
# Multi-output prediction: y is a DataFrame with one column per target
X = df.drop(columns=TARGET_COLS)
y = df[TARGET_COLS]
print(f"Features: {{X.shape[1]}}  |  Targets: {{len(TARGET_COLS)}}")
'''
    else:
        script += f'''TARGET = "{target_cols[0]}"
X = df.drop(columns=[TARGET])
y = df[TARGET]
'''

    # ------------------------------------------------------------------
    # SECTION 2 — Preprocessing (derived from the actual fitted state)
    # ------------------------------------------------------------------
    if preprocessor is not None:
        script += _build_preprocessing_block(preprocessor, target_cols)
    else:
        script += _generic_preprocessing_block()

    # ------------------------------------------------------------------
    # SECTION 3 — Train / Test split (only if not already done inside
    #              the preprocessing block)
    # ------------------------------------------------------------------
    if preprocessor is None:
        test_size = 0.2
        script += f'''
# ─────────────────────────────────────────────────────────────────────────────
# STEP — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=42
)
print(f"Train size: {{len(X_train)}} | Test size: {{len(X_test)}}")
'''

    # ------------------------------------------------------------------
    # SECTION 4 — Model initialisation
    # ------------------------------------------------------------------
    script += f'''
# ─────────────────────────────────────────────────────────────────────────────
# STEP — MODEL INITIALISATION{"  (Multi-Output Wrapper)" if is_multi else ""}
# ─────────────────────────────────────────────────────────────────────────────
{_model_import_and_init(model_name, task_type, best_params, is_multi_output=is_multi)}
'''

    # ------------------------------------------------------------------
    # SECTION 5 — Training
    # ------------------------------------------------------------------
    script += '''
# ─────────────────────────────────────────────────────────────────────────────
# STEP — TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("Training model...")
model.fit(X_train, y_train)
print("Training complete!")
'''

    # ------------------------------------------------------------------
    # SECTION 6 — Evaluation
    # ------------------------------------------------------------------
    script += f'''
# ─────────────────────────────────────────────────────────────────────────────
# STEP — EVALUATION{"  (per-target + averaged)" if is_multi else ""}
# ─────────────────────────────────────────────────────────────────────────────
print("\\nEvaluation Results:")
{_eval_block(task_type, is_multi_output=is_multi, target_cols=target_cols)}'''

    # ------------------------------------------------------------------
    # SECTION 7 — Save model
    # ------------------------------------------------------------------
    script += f'''
# ─────────────────────────────────────────────────────────────────────────────
# STEP — SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
import pickle

model_path = "{model_name.replace(" ", "_").lower()}_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {{model_path}}")

# To load later:
# with open(model_path, "rb") as f:
#     loaded_model = pickle.load(f)
'''

    return script


# ---------------------------------------------------------------------------
# Preprocessing block builders
# ---------------------------------------------------------------------------

def _build_preprocessing_block(preprocessor, target_cols) -> str:
    """
    Inspect the fitted AutoPreprocessor and emit Python code that
    reproduces every step that was actually applied.
    """
    # target_cols may be a list; we need it as list for the drop statement
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    lines = ["""
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREPROCESSING  (mirrors the AutoML pipeline applied in-app)
# ─────────────────────────────────────────────────────────────────────────────
"""]

    # ── 3a. Feature Transformer ────────────────────────────────────────────
    transformer = getattr(preprocessor, "transformer", None)

    # Boolean normalisation
    lines.append("# 3a. Boolean normalisation (Yes/No → 1/0)")
    lines.append("""
bool_map_patterns = [
    ({'yes','no'}, {'yes':1,'no':0}),
    ({'true','false'}, {'true':1,'false':0}),
    ({'y','n'}, {'y':1,'n':0}),
]
for col in X.select_dtypes(include=['object']).columns:
    vals = set(X[col].dropna().str.lower().unique())
    for pattern, mapping in bool_map_patterns:
        if vals == pattern:
            X[col] = X[col].str.lower().map(mapping)
            print(f"  Boolean normalised: {col}")
            break
""")

    # Type correction
    lines.append("# 3b. Type correction (convert stringified numbers to numeric)")
    lines.append("""
for col in X.select_dtypes(include=['object']).columns:
    converted = pd.to_numeric(X[col], errors='coerce')
    if converted.notna().sum() > len(X) * 0.9:
        X[col] = converted
        print(f"  Converted to numeric: {col}")
""")

    # DateTime extraction
    dt_cols = getattr(transformer, "datetime_cols", []) if transformer else []
    if dt_cols:
        lines.append("# 3c. DateTime feature extraction")
        for col in dt_cols:
            lines.append(f"""
if "{col}" in X.columns:
    _parsed = pd.to_datetime(X["{col}"], errors='coerce')
    X["{col}_year"]      = _parsed.dt.year
    X["{col}_month"]     = _parsed.dt.month
    X["{col}_day"]       = _parsed.dt.day
    X["{col}_dayofweek"] = _parsed.dt.dayofweek
    X["{col}_hour"]      = _parsed.dt.hour.fillna(0).astype(int)
    X = X.drop(columns=["{col}"])
    print(f"  Extracted datetime features from: {col}")
""")
    else:
        lines.append("# 3c. No datetime columns detected\n")

    # Outlier capping
    outlier_caps = getattr(transformer, "outlier_caps", {}) if transformer else {}
    if outlier_caps:
        lines.append("# 3d. Outlier capping (IQR-based, fitted values from training data)")
        lines.append("OUTLIER_CAPS = {")
        for col, (lo, hi) in outlier_caps.items():
            lines.append(f'    "{col}": ({lo!r}, {hi!r}),')
        lines.append("}")
        lines.append("""
for col, (lo, hi) in OUTLIER_CAPS.items():
    if col in X.columns:
        X[col] = X[col].clip(lower=lo, upper=hi)
        print(f"  Outlier caps applied: {col} → [{lo:.3f}, {hi:.3f}]")
""")
    else:
        lines.append("# 3d. No outlier capping was applied\n")

    # Skewness correction
    skew_transforms = getattr(transformer, "transforms", {}) if transformer else {}
    if skew_transforms:
        lines.append("# 3e. Skewness correction (log1p / sqrt applied during training)")
        lines.append("SKEW_TRANSFORMS = {")
        for col, fn in skew_transforms.items():
            lines.append(f'    "{col}": "{fn}",')
        lines.append("}")
        lines.append("""
for col, fn in SKEW_TRANSFORMS.items():
    if col in X.columns:
        if fn == "log1p":
            X[col] = np.log1p(X[col].clip(lower=0))
        elif fn == "sqrt":
            X[col] = np.sqrt(X[col].clip(lower=0))
        print(f"  Skewness fix ({fn}): {col}")
""")
    else:
        lines.append("# 3e. No skewness corrections were applied\n")

    # ── 3b. Imputation ────────────────────────────────────────────────────
    imputer  = getattr(preprocessor, "imputer", None)
    imputers = getattr(imputer, "imputers", {}) if imputer else {}

    if imputers:
        lines.append("# 3f. Missing value imputation (strategy derived from training data)")
        lines.append("IMPUTE_CONFIG = {")
        for col, info in imputers.items():
            strat = info.get("strategy", "median")
            val   = info.get("value", 0)
            if strat in ("median", "mode"):
                lines.append(f'    "{col}": {{"strategy": "{strat}", "value": {val!r}}},')
            else:
                lines.append(f'    "{col}": {{"strategy": "median", "value": {val!r}}},  # KNN at training, median fallback here')
        lines.append("}")
        lines.append("""
for col, cfg in IMPUTE_CONFIG.items():
    if col in X.columns:
        X[col] = X[col].fillna(cfg["value"])
        print(f"  Imputed: {col} ({cfg['strategy']} = {cfg['value']})")
""")
    else:
        lines.append("# 3f. No missing-value imputation was needed\n")

    # ── 3c. Encoding ──────────────────────────────────────────────────────
    encoder  = getattr(preprocessor, "encoder", None)
    encoders = getattr(encoder, "encoders", {}) if encoder else {}

    if encoders:
        lines.append("# 3g. Feature encoding (strategy determined by cardinality at fit time)")
        lines.append("ENCODING_CONFIG = {")
        for col, info in encoders.items():
            enc_type = info.get("type", "label")
            if enc_type == "label":
                classes = list(info["encoder"].classes_) if "encoder" in info else []
                lines.append(f'    "{col}": {{"type": "label", "classes": {classes!r}}},')
            elif enc_type == "onehot":
                lines.append(f'    "{col}": {{"type": "onehot"}},')
            elif enc_type in ("target", "frequency"):
                mapping = info.get("mapping", {})
                default = info.get("default", 0)
                lines.append(f'    "{col}": {{"type": "{enc_type}", "mapping": {mapping!r}, "default": {default!r}}},')
        lines.append("}")
        lines.append(r"""
for col, cfg in ENCODING_CONFIG.items():
    if col not in X.columns:
        continue
    X[col] = X[col].fillna("__MISSING__").astype(str)

    if cfg["type"] == "label":
        classes = cfg["classes"]
        label_map = {c: i for i, c in enumerate(classes)}
        X[col] = X[col].map(label_map).fillna(-1).astype(int)
        print(f"  Label encoded: {col}")

    elif cfg["type"] == "onehot":
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        print(f"  One-hot encoded: {col} → {len(dummies.columns)} columns")

    elif cfg["type"] in ("target", "frequency"):
        mapping = cfg["mapping"]
        default = cfg.get("default", 0)
        X[col] = X[col].map(mapping).fillna(default)
        print(f"  {cfg['type'].capitalize()} encoded: {col}")
""")
    else:
        lines.append("# 3g. No categorical encoding was needed\n")

    # ── 3d. Feature Selection ─────────────────────────────────────────────
    selector     = getattr(preprocessor, "selector", None)
    dropped_var  = getattr(selector, "dropped_variance",    []) if selector else []
    dropped_corr = getattr(selector, "dropped_correlation", []) if selector else []

    if dropped_var or dropped_corr:
        all_dropped = list(set(dropped_var + dropped_corr))
        lines.append("# 3h. Feature selection (low-variance and high-correlation columns removed)")
        lines.append(f"DROPPED_FEATURES = {all_dropped!r}")
        lines.append("""
cols_to_drop = [c for c in DROPPED_FEATURES if c in X.columns]
if cols_to_drop:
    X = X.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} low-variance / high-correlation features")
""")
    else:
        lines.append("# 3h. No features were dropped by feature selection\n")

    # ── 3e. Train / Test Split ────────────────────────────────────────────
    test_size = getattr(preprocessor, "test_size", 0.2)
    is_ts     = getattr(preprocessor, "is_time_series", False)
    stratify  = "None" if (is_ts or (getattr(preprocessor, "task_type", "regression") == "regression")) else "y"

    lines.append(f"""# 3i. Train / test split (test_size={test_size}, same as used in app)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=42, shuffle={not is_ts},
    {f'stratify={stratify}' if stratify != "None" else '# stratify=None (regression / time-series / multi-output)'}
)
print(f"Train size: {{len(X_train)}} | Test size: {{len(X_test)}}")
""")

    # ── 3f. Scaling ───────────────────────────────────────────────────────
    scaler_obj  = getattr(preprocessor, "scaler", None)
    scaler_cls  = type(getattr(scaler_obj, "scaler", None)).__name__ if scaler_obj and scaler_obj.scaler else "StandardScaler"
    scaled_cols = getattr(scaler_obj, "columns", []) if scaler_obj else []

    if scaled_cols:
        lines.append(f"# 3j. Feature scaling — {scaler_cls} (fitted on train only, then applied to test)")
        lines.append(f"from sklearn.preprocessing import {scaler_cls}")
        lines.append(f"""
SCALED_COLS = {scaled_cols!r}
scaled_cols_present = [c for c in SCALED_COLS if c in X_train.columns]

scaler = {scaler_cls}()
X_train[scaled_cols_present] = scaler.fit_transform(X_train[scaled_cols_present])
X_test[scaled_cols_present]  = scaler.transform(X_test[scaled_cols_present])
print(f"  {scaler_cls} applied to {{len(scaled_cols_present)}} numeric columns")
""")
    else:
        lines.append("# 3j. No scaling was applied (no numeric columns found)\n")

    return "\n".join(lines)


def _generic_preprocessing_block() -> str:
    """Fallback generic block when no fitted preprocessor is provided."""
    return """
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREPROCESSING  (generic — no fitted preprocessor was available)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: This is a generic pipeline. For a fully accurate script, export the
# code AFTER running preprocessing inside the AutoML Assistant app.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor_pipe = ColumnTransformer([
    ('num', numeric_pipeline,     numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
], remainder='passthrough')

"""
