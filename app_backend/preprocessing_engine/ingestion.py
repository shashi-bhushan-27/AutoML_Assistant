"""
Data Ingestion Module
Handles data loading, validation, encoding detection, and deduplication.
"""
import pandas as pd
import chardet
from typing import Tuple, List, Dict, Any


class DataIngestor:
    """Handles data loading with automatic encoding detection and validation."""
    
    def __init__(self):
        self.log: List[Dict[str, Any]] = []
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        encoding = result.get('encoding', 'utf-8')
        self._log("Encoding Detection", f"Detected: {encoding}", f"Confidence: {result.get('confidence', 0):.2%}")
        return encoding
    
    def load_csv(self, file_path: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
        """Load CSV with automatic encoding detection."""
        if df is not None:
            self._log("Data Loading", "DataFrame provided directly", "Skipping file load")
            return df.copy()
        
        if file_path:
            encoding = self.detect_encoding(file_path)
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                self._log("Data Loading", f"Loaded {len(df)} rows", f"Encoding: {encoding}")
            except Exception as e:
                df = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')
                self._log("Data Loading", "Fallback to latin-1", f"Original encoding failed: {str(e)}")
            return df
        
        raise ValueError("Either file_path or df must be provided")
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_len = len(df)
        df = df.drop_duplicates()
        removed = original_len - len(df)
        
        if removed > 0:
            self._log("Deduplication", f"Removed {removed} duplicates", f"{removed/original_len:.2%} of data")
        else:
            self._log("Deduplication", "No duplicates found", "Data is unique", status="skipped")
        
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset schema and detect issues."""
        issues = []
        
        # Check for empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Empty columns: {empty_cols}")
            df = df.drop(columns=empty_cols)
            self._log("Schema Validation", f"Dropped {len(empty_cols)} empty columns", "All values were null")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")
            self._log("Schema Validation", f"Flagged {len(constant_cols)} constant columns", "Single unique value")
        
        # Check for mixed types
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(100)
            numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
            if 0 < numeric_count < len(sample):
                issues.append(f"Mixed types in: {col}")
        
        if not issues:
            self._log("Schema Validation", "No schema issues", "Dataset is clean", status="skipped")
        
        return {"issues": issues, "df": df}
    
    def ingest(self, file_path: str = None, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """Full ingestion pipeline."""
        df = self.load_csv(file_path=file_path, df=df)
        df = self.remove_duplicates(df)
        result = self.validate_schema(df)
        return result["df"], self.log
