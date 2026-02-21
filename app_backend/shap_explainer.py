"""
SHAP Explainability Module — AutoML Assistant
Generates SHAP explanations for trained ML models.
Supports: tree-based models (XGBoost, RandomForest, GradientBoosting),
          linear models (LogisticRegression, LinearRegression, Ridge, Lasso),
          and generic KernelExplainer fallback for others.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# ── SHAP is an optional dependency ──────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    Wraps SHAP explainer selection, value computation, and data preparation
    for Plotly-compatible visualisation in Streamlit.
    """

    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 task_type: str = "Regression", feature_names: Optional[list] = None):
        """
        Parameters
        ----------
        model       : Trained scikit-learn / XGBoost model object.
        X_train     : Training features (used to build background for KernelExplainer).
        X_test      : Test features (used to compute SHAP values on).
        task_type   : "Regression" or "Classification".
        feature_names : Optional list of feature names.
        """
        self.model = model
        self.X_train = self._ensure_dataframe(X_train, feature_names)
        self.X_test = self._ensure_dataframe(X_test, feature_names)
        self.task_type = task_type
        self.feature_names = list(self.X_train.columns)
        self._explainer = None
        self._shap_values = None  # Cached

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_dataframe(X, feature_names=None) -> pd.DataFrame:
        """Convert input to DataFrame and cast all columns to float32.
        This prevents NumPy 2.x boolean-subtract errors on one-hot encoded columns.
        """
        if isinstance(X, pd.DataFrame):
            df = X.reset_index(drop=True)
        elif isinstance(X, np.ndarray):
            cols = feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
        else:
            df = pd.DataFrame(X)
        # Cast every column to float32 — safe for SHAP and avoids bool arithmetic errors
        return df.astype(np.float32, errors='ignore')

    def _get_model_type(self) -> str:
        model_class = type(self.model).__name__.lower()
        tree_keywords = ["forest", "xgb", "gradient", "tree", "boost", "extra"]
        linear_keywords = ["linear", "logistic", "ridge", "lasso", "elasticnet"]

        for kw in tree_keywords:
            if kw in model_class:
                return "tree"
        for kw in linear_keywords:
            if kw in model_class:
                return "linear"
        return "kernel"

    # ── Core: Build Explainer ────────────────────────────────────────────────

    def build_explainer(self) -> bool:
        """Build the appropriate SHAP explainer. Returns True on success."""
        if not SHAP_AVAILABLE:
            return False
        try:
            model_type = self._get_model_type()

            if model_type == "tree":
                self._explainer = shap.TreeExplainer(self.model)
            elif model_type == "linear":
                self._explainer = shap.LinearExplainer(
                    self.model,
                    self.X_train,
                    feature_perturbation="correlation_dependent"
                )
            else:
                # KernelExplainer: use a small background sample for speed
                background = shap.sample(self.X_train, min(50, len(self.X_train)))
                self._explainer = shap.KernelExplainer(
                    self.model.predict, background
                )
            return True
        except Exception as e:
            print(f"[SHAP] Explainer build failed: {e}")
            return False

    # ── Core: Compute SHAP Values ────────────────────────────────────────────

    def compute_shap_values(self, max_rows: int = 200) -> Optional[np.ndarray]:
        """
        Compute SHAP values on X_test (limited to max_rows for speed).
        Results are cached.
        """
        if self._shap_values is not None:
            return self._shap_values

        if self._explainer is None:
            success = self.build_explainer()
            if not success:
                return None

        X_sample = self.X_test.head(max_rows)

        try:
            raw = self._explainer.shap_values(X_sample)

            # For binary classification TreeExplainer returns a list [neg_class, pos_class]
            if isinstance(raw, list):
                if len(raw) == 2:
                    shap_vals = raw[1]  # Positive class
                else:
                    shap_vals = raw[0]
            else:
                shap_vals = raw

            # Ensure 2D (rows x features)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]  # Multi-class: take class-1 slice

            self._shap_values = shap_vals
            return self._shap_values

        except Exception as e:
            print(f"[SHAP] Value computation failed: {e}")
            return None

    # ── Plotly-compatible outputs ────────────────────────────────────────────

    def get_feature_importance_df(self, top_n: int = 15) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame with mean absolute SHAP values per feature.
        Ready for direct use in px.bar().
        """
        shap_vals = self.compute_shap_values()
        if shap_vals is None:
            return None

        mean_abs = np.abs(shap_vals).mean(axis=0)
        df = pd.DataFrame({
            "Feature": self.feature_names[:len(mean_abs)],
            "SHAP Importance": mean_abs
        }).sort_values("SHAP Importance", ascending=True).tail(top_n)

        return df

    def get_beeswarm_data(self, top_n: int = 15) -> Optional[pd.DataFrame]:
        """
        Returns long-format DataFrame for a dot/beeswarm-style plot.
        Columns: Feature, SHAP Value, Feature Value (normalised)
        """
        shap_vals = self.compute_shap_values()
        if shap_vals is None:
            return None

        X_sample = self.X_test.head(shap_vals.shape[0])
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-top_n:]

        rows = []
        for idx in top_idx:
            feat = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
            shap_col = shap_vals[:, idx]
            raw_col = X_sample.iloc[:, idx].values if idx < X_sample.shape[1] else np.zeros(len(shap_col))

            # Cast to float64 explicitly — prevents NumPy 2.x bool subtract error
            feat_col = raw_col.astype(np.float64)

            # Normalise feature values 0-1 for colour encoding
            mn, mx = feat_col.min(), feat_col.max()
            norm = (feat_col - mn) / (mx - mn + 1e-8)

            for sv, fv, fnorm in zip(shap_col, feat_col, norm):
                rows.append({
                    "Feature": feat,
                    "SHAP Value": float(sv),
                    "Feature Value": float(fv),
                    "Feature Value (norm)": float(fnorm),
                })

        return pd.DataFrame(rows)

    def get_waterfall_data(self, row_index: int = 0) -> Optional[Dict[str, Any]]:
        """
        Returns data for a waterfall chart for a single prediction (row_index).
        Includes: base_value (expected model output), shap contributions per feature.
        """
        shap_vals = self.compute_shap_values()
        if shap_vals is None:
            return None

        if row_index >= shap_vals.shape[0]:
            row_index = 0

        contributions = shap_vals[row_index]

        # Safely extract a scalar expected value regardless of explainer type:
        #   - Regression / LinearExplainer  → scalar float
        #   - Binary TreeExplainer          → list/array of 2  → take [-1] (positive class)
        #   - Single-value array (size 1)   → index [1] would crash → use [-1] instead
        _ev = self._explainer.expected_value
        if isinstance(_ev, (list, np.ndarray)):
            expected_value = float(np.asarray(_ev).flat[-1])
        else:
            expected_value = float(_ev)

        X_row = self.X_test.iloc[row_index]

        # Sort by absolute SHAP and take top 15
        sorted_idx = np.argsort(np.abs(contributions))[::-1][:15]

        feat_names = [self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                      for i in sorted_idx]
        feat_vals = [X_row.iloc[i] if i < len(X_row) else 0.0 for i in sorted_idx]
        shap_contr = [float(contributions[i]) for i in sorted_idx]

        return {
            "base_value": expected_value,
            "features": feat_names,
            "feature_values": feat_vals,
            "shap_contributions": shap_contr,
            "prediction": expected_value + float(contributions.sum()),
        }

    def get_dependence_data(self, feature_name: str) -> Optional[pd.DataFrame]:
        """
        Returns data for a dependence plot: Feature value vs SHAP value.
        Useful for showing how one feature affects the model output.
        """
        shap_vals = self.compute_shap_values()
        if shap_vals is None or feature_name not in self.feature_names:
            return None

        feat_idx = self.feature_names.index(feature_name)
        X_sample = self.X_test.head(shap_vals.shape[0])

        # Cast to float64 to avoid NumPy 2.x bool arithmetic errors
        feat_values = X_sample.iloc[:, feat_idx].values.astype(np.float64)

        return pd.DataFrame({
            "Feature Value": feat_values,
            "SHAP Value": shap_vals[:, feat_idx].astype(np.float64),
        })

    @staticmethod
    def is_available() -> bool:
        """Check if shap library is installed."""
        return SHAP_AVAILABLE
