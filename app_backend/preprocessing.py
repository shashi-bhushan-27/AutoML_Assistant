"""
Data preprocessing utilities: cleaning, missing values, type inference
"""
from typing import Any
import pandas as pd


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
