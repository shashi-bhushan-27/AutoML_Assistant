"""
Advanced Data Preprocessing Pipeline
Handles type detection, encoding, imputation, and outlier flagging.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Simple cleaning: strip column names, drop fully-empty cols, fill NaNs"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")
    return df


class AdvancedPreprocessor:
    """
    Intelligent preprocessing pipeline that:
    1. Detects true column types (numeric vs mixed/string).
    2. Applies Label Encoding for categorical columns.
    3. Handles missing values.
    4. Reports all transformations made.
    """
    
    def __init__(self):
        self.encoders: Dict[str, Any] = {}
        self.report: Dict[str, Any] = {
            "columns_analyzed": 0,
            "numeric_columns": [],
            "encoded_columns": [],
            "imputed_columns": [],
            "outlier_columns": [],
            "transformations": []
        }
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """
        Check if ALL non-null values in a column can be converted to numeric.
        Returns True only if the entire column is numeric.
        """
        try:
            # Drop NaNs and try converting
            non_null = series.dropna()
            if len(non_null) == 0:
                return True  # Empty column, treat as numeric
            pd.to_numeric(non_null, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _detect_outliers(self, series: pd.Series, threshold: float = 3.0) -> bool:
        """
        Detect if a numeric column has outliers using Z-score method.
        """
        try:
            if not pd.api.types.is_numeric_dtype(series):
                return False
            z_scores = np.abs((series - series.mean()) / series.std())
            return (z_scores > threshold).any()
        except:
            return False
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Main processing function. Analyzes and transforms the dataframe.
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.impute import SimpleImputer
        
        df_processed = df.copy()
        self.report["columns_analyzed"] = len(df.columns)
        
        columns_to_process = [c for c in df.columns if c != target_col]
        
        for col in columns_to_process:
            series = df_processed[col]
            
            # Step 1: Type Detection
            if self._is_numeric_column(series):
                # It's truly numeric
                self.report["numeric_columns"].append(col)
                
                # Convert to numeric (in case it was object dtype with numeric strings)
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Step 2: Impute missing values in numeric columns
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_val)
                    self.report["imputed_columns"].append(col)
                    self.report["transformations"].append(f"Imputed '{col}' with median ({median_val:.2f})")
                
                # Step 3: Check for outliers
                if self._detect_outliers(df_processed[col]):
                    self.report["outlier_columns"].append(col)
                    
            else:
                # It's categorical/mixed - needs encoding
                self.report["encoded_columns"].append(col)
                
                # Fill missing with 'unknown'
                df_processed[col] = df_processed[col].fillna("unknown").astype(str)
                
                # Label Encode
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.encoders[col] = le
                
                n_unique = len(le.classes_)
                self.report["transformations"].append(f"Label Encoded '{col}' ({n_unique} unique values)")
        
        return df_processed
    
    def get_report(self) -> Dict[str, Any]:
        """Returns the preprocessing report."""
        return self.report
    
    def get_summary_text(self) -> str:
        """Returns a human-readable summary."""
        lines = []
        lines.append(f"📊 **Analyzed {self.report['columns_analyzed']} columns**")
        lines.append(f"- Numeric: {len(self.report['numeric_columns'])}")
        lines.append(f"- Encoded: {len(self.report['encoded_columns'])}")
        if self.report['imputed_columns']:
            lines.append(f"- Imputed: {len(self.report['imputed_columns'])}")
        if self.report['outlier_columns']:
            lines.append(f"⚠️ Outliers detected in: {', '.join(self.report['outlier_columns'])}")
        return "\n".join(lines)
