"""
Imputer Module
Intelligent missing value handling with strategy selection.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.impute import SimpleImputer, KNNImputer


class SmartImputer:
    """Handles missing values with intelligent strategy selection."""
    
    def __init__(self):
        self.log: List[Dict[str, Any]] = []
        self.imputers: Dict[str, Any] = {}
        self.strategies: Dict[str, str] = {}
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    def _select_strategy(self, series: pd.Series, missing_pct: float) -> str:
        """Select imputation strategy based on missing percentage and data type."""
        if missing_pct > 50:
            return "drop"
        
        if pd.api.types.is_numeric_dtype(series):
            if missing_pct < 5:
                return "median"
            elif missing_pct < 30:
                return "knn"
            else:
                return "median"  # KNN too slow for high missing
        else:
            return "mode"
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Fit and transform missing values."""
        df = df.copy()
        cols_to_drop = []
        
        for col in df.columns:
            if col == target_col:
                continue
                
            missing_pct = df[col].isnull().sum() / len(df) * 100
            
            if missing_pct == 0:
                continue
            
            strategy = self._select_strategy(df[col], missing_pct)
            self.strategies[col] = strategy
            
            if strategy == "drop":
                cols_to_drop.append(col)
                self._log("Imputation", f"Dropping column: {col}", f"Missing: {missing_pct:.1f}% (too high)")
                
            elif strategy == "median":
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.imputers[col] = {"strategy": "median", "value": median_val}
                self._log("Imputation", f"Median impute: {col}", f"Missing: {missing_pct:.1f}%, Value: {median_val:.2f}")
                
            elif strategy == "mode":
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "unknown"
                df[col] = df[col].fillna(mode_val)
                self.imputers[col] = {"strategy": "mode", "value": mode_val}
                self._log("Imputation", f"Mode impute: {col}", f"Missing: {missing_pct:.1f}%, Value: {mode_val}")
                
            elif strategy == "knn":
                # KNN only works on numeric
                try:
                    knn = KNNImputer(n_neighbors=5)
                    df[[col]] = knn.fit_transform(df[[col]])
                    self.imputers[col] = {"strategy": "knn", "imputer": knn}
                    self._log("Imputation", f"KNN impute: {col}", f"Missing: {missing_pct:.1f}%, k=5")
                except:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self._log("Imputation", f"Fallback median: {col}", "KNN failed")
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted imputers."""
        df = df.copy()
        
        for col, imputer_info in self.imputers.items():
            if col not in df.columns:
                continue
            
            if imputer_info["strategy"] in ["median", "mode"]:
                df[col] = df[col].fillna(imputer_info["value"])
            elif imputer_info["strategy"] == "knn":
                df[[col]] = imputer_info["imputer"].transform(df[[col]])
        
        return df
