"""
Main AutoPreprocessor Engine
Orchestrates all preprocessing components with full explainability.
"""
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from .ingestion import DataIngestor
from .profiler import DatasetProfiler
from .imputer import SmartImputer
from .encoder import SmartEncoder
from .scaler import SmartScaler
from .transformer import FeatureTransformer
from .selector import FeatureSelector
from .balancer import ImbalanceHandler
from .splitter import DataSplitter


class AutoPreprocessor:
    """
    Complete automated preprocessing engine.
    Handles any tabular dataset with full explainability.
    """
    
    def __init__(self, 
                 target_col: str = None,
                 task_type: str = "auto",
                 is_time_series: bool = False,
                 date_col: str = None,
                 test_size: float = 0.2,
                 apply_smote: bool = False,
                 verbose: bool = True):
        """
        Args:
            target_col: Target variable column name
            task_type: 'classification', 'regression', or 'auto'
            is_time_series: Whether data is time-series
            date_col: Date column for time-series
            test_size: Test split ratio
            apply_smote: Whether to apply SMOTE for imbalanced data
            verbose: Print progress logs
        """
        self.target_col = target_col
        self.task_type = task_type
        self.is_time_series = is_time_series
        self.date_col = date_col
        self.test_size = test_size
        self.apply_smote = apply_smote
        self.verbose = verbose
        
        # Initialize components
        self.ingestor = DataIngestor()
        self.profiler = DatasetProfiler()
        self.imputer = SmartImputer()
        self.encoder = SmartEncoder()
        self.scaler = SmartScaler()
        self.transformer = FeatureTransformer()
        self.selector = FeatureSelector()
        self.balancer = ImbalanceHandler()
        self.splitter = DataSplitter(test_size=test_size)
        
        # Results storage
        self.profile: Dict = {}
        self.full_log: List[Dict] = []
        self.is_fitted = False
        
        # Data storage
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
    
    def _collect_logs(self):
        """Collect logs from all components."""
        self.full_log = []
        self.full_log.extend(self.ingestor.log)
        self.full_log.extend(self.profiler.log)
        self.full_log.extend(self.imputer.log)
        self.full_log.extend(self.transformer.log)
        self.full_log.extend(self.encoder.log)
        self.full_log.extend(self.scaler.log)
        self.full_log.extend(self.selector.log)
        self.full_log.extend(self.balancer.log)
        self.full_log.extend(self.splitter.log)
    
    def _print(self, msg: str):
        if self.verbose:
            print(f"[AutoPreprocessor] {msg}")
    
    def fit_transform(self, df: pd.DataFrame = None, file_path: str = None) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Returns:
            Dict with X_train, X_test, y_train, y_test, profile, and logs
        """
        self._print("Starting preprocessing pipeline...")
        
        # ========== 1. INGESTION ==========
        self._print("Step 1: Data Ingestion")
        df, _ = self.ingestor.ingest(file_path=file_path, df=df)
        
        # ========== 2. PROFILING ==========
        self._print("Step 2: Dataset Profiling")
        self.profile = self.profiler.generate_profile(df, target_col=self.target_col)
        
        # Auto-detect task type
        if self.task_type == "auto" and self.target_col:
            self.task_type = self.profile.get("task_type", "regression")
        
        # ========== 3. SEPARATE X AND Y ==========
        if self.target_col and self.target_col in df.columns:
            y = df[self.target_col].copy()
            X = df.drop(columns=[self.target_col])
        else:
            y = None
            X = df.copy()
        
        # ========== 4. TRANSFORMATIONS (Before split to extract features) ==========
        self._print("Step 3: Feature Transformations")
        X = self.transformer.fit_transform(X, target_col=None)
        
        # ========== 5. IMPUTATION ==========
        self._print("Step 4: Missing Value Imputation")
        X = self.imputer.fit_transform(X, target_col=None)
        
        # ========== 6. ENCODING ==========
        self._print("Step 5: Feature Encoding")
        X = self.encoder.fit_transform(X, target_col=None, y=y)
        
        # ========== 7. FEATURE SELECTION ==========
        self._print("Step 6: Feature Selection")
        X = self.selector.fit_transform(X, target_col=None)
        
        # ========== 8. TRAIN-TEST SPLIT ==========
        self._print("Step 7: Train-Test Split")
        if y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = self.splitter.split(
                X, y, 
                task_type=self.task_type,
                is_time_series=self.is_time_series,
                date_col=self.date_col
            )
        else:
            self.X_train = X
            self.X_test = None
            self.y_train = None
            self.y_test = None
        
        # ========== 9. SCALING (Fit on train only!) ==========
        self._print("Step 8: Feature Scaling")
        self.X_train = self.scaler.fit_transform(self.X_train)
        if self.X_test is not None:
            self.X_test = self.scaler.transform(self.X_test)
        
        # ========== 10. CLASS IMBALANCE ==========
        if self.task_type == "classification" and self.y_train is not None:
            self._print("Step 9: Class Imbalance Check")
            imbalance_info = self.balancer.analyze(self.y_train)
            
            if self.apply_smote and imbalance_info.get("imbalanced"):
                self.X_train, self.y_train = self.balancer.apply_smote(self.X_train, self.y_train)
        
        # ========== FINALIZE ==========
        self._collect_logs()
        self.is_fitted = True
        self._print("Preprocessing complete!")
        
        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "profile": self.profile,
            "logs": self.full_log
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Apply same transformations
        X = self.transformer.correct_types(df)
        X = self.imputer.transform(X)
        X = self.encoder.transform(X)
        X = self.scaler.transform(X)
        
        return X
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing report."""
        applied = [log for log in self.full_log if log.get("status") == "applied"]
        skipped = [log for log in self.full_log if log.get("status") == "skipped"]
        
        return {
            "summary": {
                "total_steps": len(self.full_log),
                "applied": len(applied),
                "skipped": len(skipped),
                "task_type": self.task_type,
                "features_final": len(self.X_train.columns) if self.X_train is not None else 0,
                "train_samples": len(self.X_train) if self.X_train is not None else 0,
                "test_samples": len(self.X_test) if self.X_test is not None else 0
            },
            "applied_steps": applied,
            "skipped_steps": skipped,
            "profile": self.profile,
            "class_weights": self.balancer.get_class_weights()
        }
    
    def get_markdown_report(self) -> str:
        """Generate markdown report for UI display."""
        report = self.get_report()
        lines = []
        
        lines.append("## 📊 Preprocessing Summary")
        lines.append(f"- **Task Type**: {report['summary']['task_type']}")
        lines.append(f"- **Final Features**: {report['summary']['features_final']}")
        lines.append(f"- **Train Samples**: {report['summary']['train_samples']}")
        lines.append(f"- **Test Samples**: {report['summary']['test_samples']}")
        lines.append("")
        
        lines.append("### ✅ Applied Steps")
        for step in report['applied_steps']:
            lines.append(f"- **{step['step']}**: {step['action']}")
            lines.append(f"  - *Reason*: {step['reason']}")
        
        lines.append("")
        lines.append("### ⏭️ Skipped Steps")
        for step in report['skipped_steps']:
            lines.append(f"- **{step['step']}**: {step['action']}")
            lines.append(f"  - *Reason*: {step['reason']}")
        
        return "\n".join(lines)
    
    def save(self, path: str):
        """Save fitted preprocessor to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self._print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'AutoPreprocessor':
        """Load preprocessor from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def export_data(self, train_path: str, test_path: str = None):
        """Export processed datasets."""
        if self.X_train is not None:
            train_df = self.X_train.copy()
            if self.y_train is not None:
                train_df[self.target_col] = self.y_train.values
            train_df.to_csv(train_path, index=False)
            self._print(f"Train data saved to {train_path}")
        
        if test_path and self.X_test is not None:
            test_df = self.X_test.copy()
            if self.y_test is not None:
                test_df[self.target_col] = self.y_test.values
            test_df.to_csv(test_path, index=False)
            self._print(f"Test data saved to {test_path}")
