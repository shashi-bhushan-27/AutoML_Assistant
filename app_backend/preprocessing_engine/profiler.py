"""
Dataset Profiler Module
Generates comprehensive statistics and analysis of the dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


class DatasetProfiler:
    """Analyzes dataset and generates profiling report."""
    
    def __init__(self):
        self.log: List[Dict[str, Any]] = []
        self._profile_data: Dict[str, Any] = {}
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics."""
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        self._log("Basic Stats", f"{stats['rows']} rows, {stats['columns']} columns", f"Memory: {stats['memory_mb']} MB")
        return stats
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze missing value percentages per column."""
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        
        high_missing = [col for col, pct in missing_pct.items() if pct > 30]
        if high_missing:
            self._log("Missing Analysis", f"{len(high_missing)} columns with >30% missing", f"Columns: {high_missing[:5]}")
        else:
            self._log("Missing Analysis", "No severe missing values", "All columns < 30% missing")
        
        return missing_pct
    
    def analyze_numeric_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get summary statistics for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
        
        stats = {}
        for col in numeric_df.columns:
            stats[col] = {
                "mean": round(numeric_df[col].mean(), 4),
                "std": round(numeric_df[col].std(), 4),
                "min": round(numeric_df[col].min(), 4),
                "max": round(numeric_df[col].max(), 4),
                "skewness": round(numeric_df[col].skew(), 4)
            }
        
        skewed = [col for col, s in stats.items() if abs(s['skewness']) > 1]
        if skewed:
            self._log("Numeric Analysis", f"{len(skewed)} skewed columns detected", f"May need transformation")
        
        return stats
    
    def analyze_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze categorical columns."""
        cat_df = df.select_dtypes(include=['object', 'category'])
        stats = {}
        
        for col in cat_df.columns:
            stats[col] = {
                "unique": df[col].nunique(),
                "top": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                "cardinality": "high" if df[col].nunique() > 100 else "medium" if df[col].nunique() > 10 else "low"
            }
        
        high_card = [col for col, s in stats.items() if s['cardinality'] == 'high']
        if high_card:
            self._log("Categorical Analysis", f"{len(high_card)} high-cardinality columns", "Consider target encoding")
        
        return stats
    
    def compute_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return pd.DataFrame()
        
        corr = numeric_df.corr()
        
        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.95:
                    high_corr.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))
        
        if high_corr:
            self._log("Correlation Analysis", f"{len(high_corr)} highly correlated pairs", "Consider removing redundant features")
        
        return corr
    
    def detect_target_type(self, df: pd.DataFrame, target_col: str) -> str:
        """Detect if target is classification or regression."""
        if target_col not in df.columns:
            return "unknown"
        
        y = df[target_col]
        
        if y.dtype == 'object' or y.dtype.name == 'category':
            task = "classification"
        elif y.nunique() <= 10:
            task = "classification"
        else:
            task = "regression"
        
        self._log("Target Detection", f"Task type: {task}", f"Target has {y.nunique()} unique values")
        return task
    
    def detect_class_imbalance(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Detect class imbalance in classification targets."""
        if target_col not in df.columns:
            return {}
        
        y = df[target_col]
        value_counts = y.value_counts(normalize=True)
        
        if len(value_counts) < 2:
            return {"imbalanced": False}
        
        min_ratio = value_counts.min()
        is_imbalanced = min_ratio < 0.1
        
        if is_imbalanced:
            self._log("Class Imbalance", f"Minority class: {min_ratio:.2%}", "Recommend SMOTE or class weights")
        
        return {"imbalanced": is_imbalanced, "min_ratio": round(min_ratio, 4), "distribution": value_counts.to_dict()}
    
    def generate_profile(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Generate complete dataset profile."""
        corr_result = self.compute_correlation(df)
        
        self._profile_data = {
            "basic_stats": self.get_basic_stats(df),
            "missing_values": self.analyze_missing_values(df),
            "numeric_stats": self.analyze_numeric_columns(df),
            "categorical_stats": self.analyze_categorical_columns(df),
            "correlation": corr_result.to_dict() if not corr_result.empty else {},
        }
        
        if target_col:
            self._profile_data["task_type"] = self.detect_target_type(df, target_col)
            self._profile_data["class_imbalance"] = self.detect_class_imbalance(df, target_col)
        
        return self._profile_data

