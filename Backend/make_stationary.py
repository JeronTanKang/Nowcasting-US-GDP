import pandas as pd
from statsmodels.tsa.stattools import adfuller

def make_stationary(df, max_diff=2):
    df_stationary = df.copy()
    differenced_counts = {}  # Track how many times differencing was applied

    for col in df_stationary.columns:
        diff_count = 0
        temp_series = df_stationary[col].copy()

        while diff_count < max_diff:
            if temp_series.dropna().shape[0] < 2:  # Ensure enough data points for ADF test
                break

            adf_test = adfuller(temp_series.dropna(), autolag="AIC")

            if adf_test[1] > 0.05:  # Non-stationary
                temp_series = temp_series.diff()  # Apply differencing
                diff_count += 1
            else:
                break  # Stop if stationary

        df_stationary[col] = temp_series  # Assign the final transformed series
        differenced_counts[col] = diff_count  # Store differencing count

    return df_stationary, differenced_counts 

