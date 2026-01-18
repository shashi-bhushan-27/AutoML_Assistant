"""
Transformer Module
Feature transformations, outlier handling, and feature engineering.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats


class FeatureTransformer:
    """Handles feature transformations, outliers, and engineering."""
    
    def __init__(self):
        self.log: List[Dict[str, Any]] = []
        self.transforms: Dict[str, str] = {}
        self.outlier_caps: Dict[str, Tuple[float, float]] = {}
        self.datetime_cols: List[str] = []
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    # ========== OUTLIER HANDLING ==========
    def handle_outliers(self, df: pd.DataFrame, method: str = "iqr", target_col: str = None) -> pd.DataFrame:
        """Detect and handle outliers using IQR or Z-score."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for col in numeric_cols:
            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
            else:  # z-score
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - 3 * std
                upper = mean + 3 * std
            
            outliers_before = ((df[col] < lower) | (df[col] > upper)).sum()
            
            if outliers_before > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                self.outlier_caps[col] = (lower, upper)
                self._log("Outlier Handling", f"Capped {col}", f"{outliers_before} outliers capped to [{lower:.2f}, {upper:.2f}]")
        
        if not self.outlier_caps:
            self._log("Outlier Handling", "No outliers detected", "All values within normal range", status="skipped")
        
        return df
    
    # ========== SKEWNESS CORRECTION ==========
    def fix_skewness(self, df: pd.DataFrame, threshold: float = 1.0, target_col: str = None) -> pd.DataFrame:
        """Apply log transform to highly skewed features."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for col in numeric_cols:
            skewness = df[col].skew()
            
            if abs(skewness) > threshold:
                # Check if all values are positive
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
                    self.transforms[col] = "log1p"
                    self._log("Skewness Fix", f"Log transform: {col}", f"Skewness was {skewness:.2f}")
                elif (df[col] >= 0).all():
                    df[col] = np.sqrt(df[col])
                    self.transforms[col] = "sqrt"
                    self._log("Skewness Fix", f"Sqrt transform: {col}", f"Skewness was {skewness:.2f}, has zeros")
        
        return df
    
    # ========== DATETIME FEATURES ==========
    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        df = df.copy()
        
        for col in df.columns:
            # Try to detect datetime columns
            if df[col].dtype == 'object':
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() > len(df) * 0.8:  # 80% valid dates
                        self.datetime_cols.append(col)
                        
                        df[f"{col}_year"] = parsed.dt.year
                        df[f"{col}_month"] = parsed.dt.month
                        df[f"{col}_day"] = parsed.dt.day
                        df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                        df[f"{col}_hour"] = parsed.dt.hour.fillna(0).astype(int)
                        
                        df = df.drop(columns=[col])
                        self._log("DateTime Features", f"Extracted from: {col}", "Created year, month, day, dayofweek, hour")
                except:
                    continue
        
        return df
    
    # ========== BOOLEAN NORMALIZATION ==========
    def normalize_booleans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize boolean columns to 0/1."""
        df = df.copy()
        
        for col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            
            # Check for boolean-like values
            bool_patterns = [
                {'yes', 'no'}, {'true', 'false'}, {'t', 'f'}, 
                {'y', 'n'}, {'1', '0'}, {1, 0}, {True, False}
            ]
            
            for pattern in bool_patterns:
                if unique_vals.issubset(pattern) and len(unique_vals) == 2:
                    mapping = {}
                    for val in unique_vals:
                        if str(val).lower() in ['yes', 'true', 't', 'y', '1', 1, True]:
                            mapping[val] = 1
                        else:
                            mapping[val] = 0
                    
                    df[col] = df[col].map(mapping)
                    self._log("Boolean Normalize", f"Converted: {col}", f"Mapped to 0/1")
                    break
        
        return df
    
    # ========== TYPE CORRECTION ==========
    def correct_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert object columns to numeric where possible."""
        df = df.copy()
        
        for col in df.select_dtypes(include=['object']).columns:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.9:  # 90% convertible
                    df[col] = converted
                    self._log("Type Correction", f"Converted to numeric: {col}", f"{converted.notna().sum()}/{len(df)} values converted")
            except:
                continue
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Apply all transformations."""
        df = self.normalize_booleans(df)
        df = self.correct_types(df)
        df = self.extract_datetime_features(df)
        df = self.handle_outliers(df, target_col=target_col)
        df = self.fix_skewness(df, target_col=target_col)
        return df
