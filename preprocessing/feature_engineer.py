import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame, time_col: str = 'Time_Step') -> pd.DataFrame:
    """
    Adds rolling statistics based on existing features for the DataFrame.
    Assumes data is grouped by 'Patient_ID'.
    """
    if 'Patient_ID' not in df.columns:
        return df
        
    engineered_df = df.copy()
    numeric_cols = [c for c in df.columns if c not in ['Patient_ID', time_col]]
    
    # Example: add rolling mean for HR/BP
    for col in numeric_cols:
        engineered_df[f'{col}_roll_mean_3'] = df.groupby('Patient_ID')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        engineered_df[f'{col}_roll_std_3'] = df.groupby('Patient_ID')[col].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        
    return engineered_df
