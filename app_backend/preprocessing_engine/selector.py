"""
Feature Selector Module
Removes low variance and highly correlated features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    """Handles feature selection based on variance and correlation."""
    
    def __init__(self, variance_threshold: float = 0.01, correlation_threshold: float = 0.95):
        self.log: List[Dict[str, Any]] = []
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.dropped_variance: List[str] = []
        self.dropped_correlation: List[str] = []
        self.selected_features: List[str] = []
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    def remove_low_variance(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Remove features with variance below threshold."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if not numeric_cols:
            return df
        
        selector = VarianceThreshold(threshold=self.variance_threshold)
        
        try:
            selector.fit(df[numeric_cols])
            mask = selector.get_support()
            low_var_cols = [col for col, keep in zip(numeric_cols, mask) if not keep]
            
            if low_var_cols:
                self.dropped_variance = low_var_cols
                df = df.drop(columns=low_var_cols)
                self._log("Variance Filter", f"Dropped {len(low_var_cols)} columns", f"Variance < {self.variance_threshold}")
            else:
                self._log("Variance Filter", "No low variance columns", "All features above threshold", status="skipped")
        except:
            pass
        
        return df
    
    def remove_high_correlation(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Remove one of each pair of highly correlated features."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) < 2:
            return df
        
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = []
        for col in upper.columns:
            if any(upper[col] > self.correlation_threshold):
                to_drop.append(col)
        
        if to_drop:
            self.dropped_correlation = to_drop
            df = df.drop(columns=to_drop)
            self._log("Correlation Filter", f"Dropped {len(to_drop)} columns", f"Correlation > {self.correlation_threshold}")
        else:
            self._log("Correlation Filter", "No highly correlated pairs", "All correlations acceptable", status="skipped")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Apply all feature selection steps."""
        df = self.remove_low_variance(df, target_col)
        df = self.remove_high_correlation(df, target_col)
        self.selected_features = df.columns.tolist()
        return df
    
    def get_dropped_features(self) -> Dict[str, List[str]]:
        """Return all dropped features with reasons."""
        return {
            "low_variance": self.dropped_variance,
            "high_correlation": self.dropped_correlation
        }
