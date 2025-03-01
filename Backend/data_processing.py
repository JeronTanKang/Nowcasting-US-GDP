import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np


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


def aggregate_indicators(df):
    """
    Function that takes in df with monthly frequency indicators and GDP.
    - Converts indicators to quarterly frequency using specified aggregation rules.
    - GDP remains unchanged (takes the only available value per quarter).

    Returns:
    - DataFrame with quarterly frequency.
    """

    # Convert 'date' column to datetime format if not already
    
    #df['date'] = pd.to_datetime(df['date'], format="%Y-%m")
    #df = df.set_index('date')


    # Define aggregation rules for each column
    aggregation_rule = {
        "Industrial_Production": "mean",  # Average industrial output over the period
        "Retail_Sales": "sum",  # Total sales should be summed
        "Nonfarm_Payrolls": "sum",  # Employment-related numbers are usually summed
        "Trade_Balance": "sum",  # Trade surplus/deficit is accumulated over the period
        "Core_PCE": "exp_almon",  # Applying an exponential Almon lag model for smoothing
        "Unemployment": "mean",  # Unemployment rate is an average
        "Interest_Rate": "mean",  # Interest rates are typically averaged
        "Three_Month_Treasury_Yield": "mean",  # Treasury yields are averaged
        "Construction_Spending": "sum",  # Total spending should be summed
        "Housing_Starts": "sum",  # Count data should be summed
        "Capacity_Utilization": "mean",  # Utilization rate is an average
    }

    # Separate GDP column from indicators
    gdp_data = df[['GDP']].resample('Q').last()  # Takes the last available GDP value per quarter

    # Initialize an empty DataFrame for indicators
    indicators_data = pd.DataFrame()

    # Apply different aggregation methods for each indicator
    for col, method in aggregation_rule.items():
        if method == "mean":
            indicators_data[col] = df[col].resample('Q').mean()  # Standard mean
        elif method == "sum":
            indicators_data[col] = df[col].resample('Q').sum()  # Summation for flow variables
        elif method == "exp_almon":
            indicators_data[col] = df[col].resample('Q').apply(exp_almon_weighted)  # Apply Almon weighting

    # Merge back GDP and aggregated indicators
    quarterly_df = gdp_data.merge(indicators_data, left_index=True, right_index=True, how='left')
    quarterly_df = quarterly_df.reset_index()
    quarterly_df['date'] = quarterly_df['date'].dt.strftime('%Y-%m')

    return quarterly_df


def exp_almon_weighted(series, alpha=0.9):
    """
    Applies an Exponential Almon transformation for weighted aggregation.
    - Recent values get higher weights.

    Args:
    - series (pd.Series): Time series data to transform.
    - alpha (float): Decay factor (0 < alpha < 1, closer to 1 gives more weight to recent values).

    Returns:
    - float: Weighted aggregated value.
    """
    weights = np.array([(alpha ** i) for i in range(len(series))][::-1])
    return np.sum(series * weights) / np.sum(weights)