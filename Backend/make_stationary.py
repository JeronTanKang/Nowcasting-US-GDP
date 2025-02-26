import pandas as pd
from statsmodels.tsa.stattools import adfuller


def is_stationary(series, significance_level=0.05):
    if series.dropna().shape[0] < 2:  # Ensure there are enough data points
        return False
    
    adf_test = adfuller(series.dropna(), autolag="AIC")
    p_value = adf_test[1]
    
    return p_value <= significance_level  # Stationary if p-value <= threshold

def make_stationary(df, max_diff=2):
    df_stationary = df.copy()
    differenced_counts = {}  # Track how many times differencing was applied

    for col in df_stationary.columns:
        diff_count = 0
        temp_series = df_stationary[col].copy()

        while diff_count < max_diff:
            if is_stationary(temp_series):
                break  # Stop if already stationary
            
            temp_series = temp_series.diff()  # Apply differencing
            diff_count += 1
        
        df_stationary[col] = temp_series  # Assign transformed series
        differenced_counts[col] = diff_count  # Store differencing count

    return df_stationary, differenced_counts 

