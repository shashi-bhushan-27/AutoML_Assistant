import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis

def get_column_types(df):
    """Separates columns into numerical, categorical, and datetime."""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    num_cols = df.select_dtypes(include=numerics).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Attempt to identify datetime columns automatically
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif 'date' in col.lower() or 'time' in col.lower():
            try:
                # specific check to avoid false positives
                pd.to_datetime(df[col], errors='raise') 
                date_cols.append(col)
            except:
                pass
                
    return num_cols, cat_cols, date_cols

def check_stationarity(series):
    """
    Performs Augmented Dickey-Fuller test to check if Time Series is stationary.
    Returns: 'Stationary' or 'Non-Stationary'
    """
    try:
        # Drop NAs just for the test
        clean_series = series.dropna()
        if len(clean_series) < 20: # ADF requires some data
            return "Unknown (Too little data)"
            
        result = adfuller(clean_series)
        p_value = result[1]
        
        if p_value < 0.05:
            return "Stationary (Good for ARIMA)"
        else:
            return "Non-Stationary (Needs Differencing/Transformation)"
    except:
        return "Check Failed"

def analyze_dataset(df, target_col=None):
    """
    Main function to extract stats for the LLM.
    """
    stats = {}
    
    # 1. Basic Dimensions
    stats['rows'] = df.shape[0]
    stats['columns'] = df.shape[1]
    stats['missing_values'] = df.isnull().sum().sum()
    
    # 2. Column Types
    num_cols, cat_cols, date_cols = get_column_types(df)
    stats['numerical_columns'] = num_cols
    stats['categorical_columns'] = cat_cols
    stats['datetime_columns'] = date_cols
    
    # 3. Time Series Logic (Crucial for your Vehicle Data)
    if len(date_cols) > 0:
        stats['is_time_series'] = True
        stats['time_column'] = date_cols[0] # Assume first date col is the index
        
        # Try to infer frequency (Hz, Daily, etc.)
        try:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            sorted_df = df.sort_values(by=date_cols[0])
            time_diff = sorted_df[date_cols[0]].diff().median()
            stats['time_frequency'] = str(time_diff)
        except:
            stats['time_frequency'] = "Irregular"
    else:
        stats['is_time_series'] = False
        stats['time_frequency'] = "None"

    # 4. Target Variable Analysis (If user selected one)
    if target_col and target_col in df.columns:
        y = df[target_col]
        
        # Determine Task Type
        if y.dtype == 'object' or y.nunique() < 10:
            stats['task_type'] = "Classification"
        else:
            stats['task_type'] = "Regression"
            
        # Stats for Regression
        if stats['task_type'] == "Regression" and pd.api.types.is_numeric_dtype(y):
            stats['target_mean'] = round(y.mean(), 2)
            stats['target_std'] = round(y.std(), 2)
            stats['target_skewness'] = round(skew(y.dropna()), 2)
            
            # Check linearity (Correlation with other num cols)
            correlations = df[num_cols].corrwith(y).abs().mean()
            if correlations > 0.5:
                stats['linearity'] = "High (Linear Models might work)"
            else:
                stats['linearity'] = "Low (Non-linear/Tree models needed)"
                
            # Check Stationarity (Only if Time Series)
            if stats['is_time_series']:
                stats['stationarity'] = check_stationarity(y)
                
    return stats

# --- TEST BLOCK (Run this file independently to check) ---
if __name__ == "__main__":
    # Create dummy data to test
    data = {
        'Date': pd.date_range(start='1/1/2024', periods=100, freq='D'),
        'Speed': np.random.normal(60, 10, 100), # Continuous
        'Fault_Code': np.random.choice(['A', 'B'], 100) # Categorical
    }
    df = pd.DataFrame(data)
    
    print("--- ANALYZING DUMMY VEHICLE DATA ---")
    summary = analyze_dataset(df, target_col='Speed')
    
    import json
    print(json.dumps(summary, indent=4, default=str))