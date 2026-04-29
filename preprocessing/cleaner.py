import pandas as pd
import numpy as np
from scipy import stats

def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Impute missing values in medical time-series.
    Typically forward-fill then backward-fill is used, or linear interpolation.
    """
    if method == 'ffill':
        return df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        return df.interpolate(method='linear', limit_direction='both')
    else:
        return df.fillna(df.mean())

def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Removes extreme outliers based on z-score threshold.
    Replaces them with the median of that column.
    """
    cleaned_df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Don't touch categorical/ID columns if they exist
    if 'Patient_ID' in numeric_cols:
        numeric_cols = numeric_cols.drop(['Patient_ID', 'Time_Step'], errors='ignore')
        
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(cleaned_df[col].dropna()))
        outliers = z_scores > threshold
        
        median_val = cleaned_df[col].median()
        # For simplicity in logic, using index directly on boolean mask
        invalid_idx = cleaned_df[col].dropna()[outliers].index
        cleaned_df.loc[invalid_idx, col] = median_val
        
    return cleaned_df

def clean_medical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.
    """
    df = handle_missing_values(df, method='interpolate')
    df = remove_outliers(df)
    return df
