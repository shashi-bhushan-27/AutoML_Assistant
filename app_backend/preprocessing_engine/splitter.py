"""
Data Splitter Module
Train-test split with stratification and time-awareness.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Handles train-test split with data leakage prevention."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.log: List[Dict[str, Any]] = []
        self.test_size = test_size
        self.random_state = random_state
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    def split(self, X: pd.DataFrame, y: pd.Series, 
              task_type: str = "classification",
              is_time_series: bool = False,
              date_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data with appropriate strategy.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        
        if is_time_series:
            # Time-based split (no shuffle)
            split_idx = int(len(X) * (1 - self.test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            self._log("Data Split", f"Time-based split at index {split_idx}", 
                     f"Train: {len(X_train)}, Test: {len(X_test)}")
            
        elif task_type == "classification":
            # Stratified split for classification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=self.test_size, 
                    stratify=y,
                    random_state=self.random_state
                )
                self._log("Data Split", "Stratified split applied", 
                         f"Train: {len(X_train)}, Test: {len(X_test)}")
            except:
                # Fallback if stratification fails
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=self.test_size, 
                    random_state=self.random_state
                )
                self._log("Data Split", "Random split (stratification failed)", 
                         f"Train: {len(X_train)}, Test: {len(X_test)}")
        else:
            # Random split for regression
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            self._log("Data Split", "Random split for regression", 
                     f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
