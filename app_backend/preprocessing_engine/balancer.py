"""
Class Imbalance Handler Module
Detects and handles class imbalance in classification tasks.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


class ImbalanceHandler:
    """Detects and recommends handling for class imbalance."""
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Minimum class ratio to trigger imbalance handling
        """
        self.log: List[Dict[str, Any]] = []
        self.threshold = threshold
        self.is_imbalanced = False
        self.class_distribution: Dict = {}
        self.recommendation: str = None
    
    def _log(self, step: str, action: str, reason: str, status: str = "applied"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status})
    
    def analyze(self, y: pd.Series) -> Dict[str, Any]:
        """Analyze class distribution and detect imbalance."""
        if y is None or len(y) == 0:
            return {"imbalanced": False}
        
        value_counts = y.value_counts(normalize=True)
        self.class_distribution = value_counts.to_dict()
        
        if len(value_counts) < 2:
            self._log("Imbalance Check", "Single class detected", "Cannot apply balancing", status="skipped")
            return {"imbalanced": False, "distribution": self.class_distribution}
        
        min_ratio = value_counts.min()
        self.is_imbalanced = min_ratio < self.threshold
        
        if self.is_imbalanced:
            if min_ratio < 0.05:
                self.recommendation = "SMOTE + class_weight"
                self._log("Imbalance Detection", f"Severe imbalance ({min_ratio:.2%})", "Recommend SMOTE + class_weight='balanced'")
            else:
                self.recommendation = "class_weight"
                self._log("Imbalance Detection", f"Moderate imbalance ({min_ratio:.2%})", "Recommend class_weight='balanced'")
        else:
            self._log("Imbalance Check", "Classes are balanced", f"Min ratio: {min_ratio:.2%}", status="skipped")
        
        return {
            "imbalanced": self.is_imbalanced,
            "min_ratio": min_ratio,
            "distribution": self.class_distribution,
            "recommendation": self.recommendation
        }
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to balance classes."""
        try:
            from imblearn.over_sampling import SMOTE
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            self._log("SMOTE Applied", f"Resampled from {len(X)} to {len(X_resampled)}", "Synthetic minority oversampling")
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except ImportError:
            self._log("SMOTE", "imblearn not installed", "Use: pip install imbalanced-learn", status="skipped")
            return X, y
        except Exception as e:
            self._log("SMOTE", f"Failed: {str(e)}", "Falling back to original data", status="skipped")
            return X, y
    
    def get_class_weights(self) -> Dict:
        """Return class weights for sklearn models."""
        if not self.is_imbalanced:
            return None
        return "balanced"
