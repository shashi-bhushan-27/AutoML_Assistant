"""
Encoder Module
Feature encoding with strategy selection based on cardinality.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


class SmartEncoder:
    """Handles feature encoding with intelligent strategy selection."""
    
    def __init__(self):
        self.log: List[Dict[str, Any]] = []
        self.encoders: Dict[str, Any] = {}
        self.strategies: Dict[str, str] = {}
        self.ohe_columns: List[str] = []
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    def _select_strategy(self, series: pd.Series) -> str:
        """Select encoding strategy based on cardinality."""
        n_unique = series.nunique()
        
        if n_unique <= 2:
            return "label"  # Binary
        elif n_unique <= 10:
            return "onehot"  # Low cardinality
        elif n_unique <= 100:
            return "target"  # Medium cardinality - target encoding
        else:
            return "frequency"  # High cardinality
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform categorical features."""
        df = df.copy()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        
        for col in cat_cols:
            strategy = self._select_strategy(df[col])
            self.strategies[col] = strategy
            
            # Fill NaN before encoding
            df[col] = df[col].fillna("__MISSING__").astype(str)
            
            if strategy == "label":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = {"type": "label", "encoder": le}
                self._log("Encoding", f"Label encode: {col}", f"Binary/Low cardinality ({df[col].nunique()} unique)")
                
            elif strategy == "onehot":
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.ohe_columns.extend(dummies.columns.tolist())
                self.encoders[col] = {"type": "onehot", "categories": df[col].unique().tolist() if col in df.columns else []}
                self._log("Encoding", f"One-hot encode: {col}", f"Low cardinality, created {len(dummies.columns)} columns")
                
            elif strategy == "target":
                # Target encoding (mean of target per category)
                if y is not None and pd.api.types.is_numeric_dtype(y):
                    means = df.groupby(col)[target_col].mean() if target_col in df.columns else y.groupby(df[col]).mean()
                    global_mean = y.mean() if y is not None else 0
                    mapping = means.to_dict()
                    df[col] = df[col].map(mapping).fillna(global_mean)
                    self.encoders[col] = {"type": "target", "mapping": mapping, "default": global_mean}
                    self._log("Encoding", f"Target encode: {col}", f"Medium cardinality ({len(mapping)} categories)")
                else:
                    # Fallback to frequency
                    freq = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq).fillna(0)
                    self.encoders[col] = {"type": "frequency", "mapping": freq}
                    self._log("Encoding", f"Frequency encode: {col}", "Target not numeric, using frequency")
                    
            elif strategy == "frequency":
                freq = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq).fillna(0)
                self.encoders[col] = {"type": "frequency", "mapping": freq}
                self._log("Encoding", f"Frequency encode: {col}", f"High cardinality ({len(freq)} categories)")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders."""
        df = df.copy()
        
        for col, encoder_info in self.encoders.items():
            if col not in df.columns:
                continue
            
            df[col] = df[col].fillna("__MISSING__").astype(str)
            
            if encoder_info["type"] == "label":
                # Handle unseen categories
                le = encoder_info["encoder"]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                
            elif encoder_info["type"] == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                
            elif encoder_info["type"] in ["target", "frequency"]:
                mapping = encoder_info["mapping"]
                default = encoder_info.get("default", 0)
                df[col] = df[col].map(mapping).fillna(default)
        
        return df
