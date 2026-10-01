"""
Data Ingestion Module
Handles data loading, encoding detection, validation and deduplication.

Deduplication and the removal of all-empty columns run on *all* rows, before the
train/test split. Both are structural (no statistic is learned from the data), but
deduplication can remove many rows (9,144 on Credit Card Fraud), so the count is
always logged and shown in the UI.
"""
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import chardet
import pandas as pd

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 200


class DataIngestor:
    """Loads CSV data with encoding detection, removes duplicate rows and empty columns."""

    def __init__(self, drop_duplicates: bool = True):
        self.log: List[Dict[str, Any]] = []
        self.drop_duplicates = drop_duplicates
        self.duplicates_removed = 0
        self.empty_columns_dropped: List[str] = []
        self.encoding: Optional[str] = None

    def _log(self, step: str, action: str, reason: str, status: str = "applied", fitted_on: str = "all rows"):
        self.log.append({"step": step, "action": action, "reason": reason, "status": status,
                         "fitted_on": fitted_on})

    # ── encoding ────────────────────────────────────────────────────────────
    def detect_encoding(self, file_path: str = None, raw: bytes = None) -> str:
        """Detect the text encoding of a file (or raw bytes) with chardet on the first 100 kB."""
        if raw is None:
            with open(file_path, "rb") as f:
                raw = f.read(100_000)
        result = chardet.detect(raw[:100_000])
        encoding = result.get("encoding") or "utf-8"
        # chardet reports pure-ASCII files as "ascii"; utf-8 is a strict superset.
        if encoding.lower() == "ascii":
            encoding = "utf-8"
        self.encoding = encoding
        self._log("Encoding Detection", f"Detected: {encoding}",
                  f"Confidence: {result.get('confidence', 0) or 0:.0%}", fitted_on="n/a")
        return encoding

    def read_csv_bytes(self, raw: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse an uploaded CSV. Returns (df, info) where info records the encoding and any skipped lines."""
        encoding = self.detect_encoding(raw=raw)
        info: Dict[str, Any] = {"encoding": encoding, "size_mb": len(raw) / 1e6, "warnings": []}
        for enc in dict.fromkeys([encoding, "utf-8", "latin-1"]):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                info["encoding"] = enc
                return df, info
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError as exc:
                # Malformed lines: parse leniently, but say so.
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
                info["encoding"] = enc
                info["warnings"].append(f"Some malformed lines were skipped ({str(exc).splitlines()[0]}).")
                return df, info
        raise ValueError("Could not decode the file as UTF-8 or Latin-1 text.")

    def load_csv(self, file_path: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is not None:
            self._log("Data Loading", "DataFrame provided directly", f"{len(df)} rows", fitted_on="n/a")
            return df.copy()
        if file_path:
            with open(file_path, "rb") as f:
                data, info = self.read_csv_bytes(f.read())
            self._log("Data Loading", f"Loaded {len(data)} rows", f"Encoding: {info['encoding']}", fitted_on="n/a")
            return data
        raise ValueError("Either file_path or df must be provided")

    # ── structural cleaning (all rows) ─────────────────────────────────────
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.drop_duplicates:
            self._log("Deduplication", "Disabled", "Duplicate rows kept", status="skipped")
            return df
        original_len = len(df)
        df = df.drop_duplicates()
        removed = original_len - len(df)
        self.duplicates_removed = removed
        if removed > 0:
            self._log("Deduplication", f"Removed {removed:,} duplicate rows",
                      f"{removed / original_len:.2%} of rows; runs before the train/test split so no "
                      "row can appear in both partitions")
        else:
            self._log("Deduplication", "No duplicates found", "All rows are unique", status="skipped")
        return df

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = []
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Empty columns: {empty_cols}")
            df = df.drop(columns=empty_cols)
            self.empty_columns_dropped = empty_cols
            self._log("Schema Validation", f"Dropped {len(empty_cols)} empty columns", "All values were null")
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")
            self._log("Schema Validation", f"Flagged {len(constant_cols)} constant columns",
                      f"Single unique value: {constant_cols[:5]}", status="skipped")
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(100)
            numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
            if 0 < numeric_count < len(sample):
                issues.append(f"Mixed types in: {col}")
        if not issues:
            self._log("Schema Validation", "No schema issues", "Dataset is clean", status="skipped")
        return {"issues": issues, "df": df}

    def ingest(self, file_path: str = None, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, List[Dict]]:
        df = self.load_csv(file_path=file_path, df=df)
        df = self.remove_duplicates(df)
        result = self.validate_schema(df)
        return result["df"], self.log
