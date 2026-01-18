"""
Scaler Module
Feature scaling with intelligent strategy selection.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class SmartScaler:
    """Handles feature scaling with intelligent strategy selection."""
    
    def __init__(self, strategy: str = "auto"):
        """
        Args:
            strategy: 'auto', 'standard', 'minmax', or 'robust'
        """
        self.log: List[Dict[str, Any]] = []
        self.strategy = strategy
        self.scaler = None
        self.columns: List[str] = []
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    def _detect_outliers(self, df: pd.DataFrame) -> bool:
        """Check if data has significant outliers."""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
                if outliers > len(df) * 0.05:  # More than 5% outliers
                    return True
        return False
    
    def _select_strategy(self, df: pd.DataFrame) -> str:
        """Select scaling strategy based on data characteristics."""
        if self.strategy != "auto":
            return self.strategy
        
        if self._detect_outliers(df):
            return "robust"
        else:
            return "standard"
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Fit and transform numeric features."""
        df = df.copy()
        
        # Get numeric columns
        self.columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in self.columns:
            self.columns.remove(target_col)
        
        if not self.columns:
            self._log("Scaling", "No numeric columns to scale", "Skipped", status="skipped")
            return df
        
        # Select strategy
        strategy = self._select_strategy(df[self.columns])
        
        if strategy == "standard":
            self.scaler = StandardScaler()
            reason = "No significant outliers, using StandardScaler"
        elif strategy == "minmax":
            self.scaler = MinMaxScaler()
            reason = "Using MinMaxScaler (0-1 range)"
        elif strategy == "robust":
            self.scaler = RobustScaler()
            reason = "Outliers detected, using RobustScaler"
        else:
            self.scaler = StandardScaler()
            reason = "Default StandardScaler"
        
        # Fit and transform
        df[self.columns] = self.scaler.fit_transform(df[self.columns])
        self._log("Scaling", f"{strategy.capitalize()} scaling on {len(self.columns)} columns", reason)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler."""
        if self.scaler is None:
            return df
        
        df = df.copy()
        cols_to_scale = [c for c in self.columns if c in df.columns]
        
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data."""
        if self.scaler is None:
            return df
        
        df = df.copy()
        cols_to_scale = [c for c in self.columns if c in df.columns]
        
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.inverse_transform(df[cols_to_scale])
        
        return df
