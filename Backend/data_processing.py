"""
This file contains utility functions for handling time series data, including operations to check for stationarity, make data stationary, 
aggregate monthly data to quarterly data, and create lag features for modeling purposes. It also includes functions to handle missing months 
and preprocess data by adding missing months to complete the current quarter.

Functions:
1. `aggregate_indicators`: Aggregates monthly frequency indicators to quarterly frequency, using predefined aggregation rules for each column.
2. `create_lag_features`: Creates lagged features for time series data up to a specified maximum lag.
3. `get_missing_months`: Calculates the number of months needed to complete the current quarter and adds missing months to the DataFrame.
4. `add_missing_months`: Adds missing months to a DataFrame to complete the current quarter, ensuring monthly continuity.
"""

import pandas as pd
import numpy as np


def aggregate_indicators(df):
    """
    Aggregates monthly frequency indicators to quarterly frequency.

    The function takes in a DataFrame with monthly frequency indicators and GDP, and converts them to quarterly frequency using 
    predefined aggregation rules for each column. GDP remains unchanged, taking the only available value per quarter.

    Args:
        df (pd.DataFrame): DataFrame with monthly frequency data, including 'GDP' and other economic indicators.

    Returns:
        pd.DataFrame: DataFrame with quarterly frequency data, including aggregated indicators and GDP.
    """


    df = df.set_index('date')

    aggregation_rule = {
        "GDP": "sum",  # GDP should be aggregated using mean
        "gdp_growth": "sum",  # GDP growth, as a rate, should be averaged
        "gdp_growth_lag1": "sum",  # Lagged GDP growth, average
        "gdp_growth_lag2": "sum",  # Lagged GDP growth, average
        "gdp_growth_lag3": "sum",  # Lagged GDP growth, average
        "gdp_growth_lag4": "sum",  # Lagged GDP growth, average
        "Nonfarm_Payrolls": "sum",  # Sum of non-farm payrolls
        "Nonfarm_Payrolls_lag1": "sum", # Non-farm payrolls, lag 1(sum)
        "Construction_Spending": "sum",  # Sum of construction spending
        "Trade_Balance_lag1": "sum",  # Trade balance, lag1 (sum)
        "Industrial_Production_lag1": "mean",  # Industrial production, lag1 (average)
        "Industrial_Production_lag3": "mean",  # Industrial production, lag3 (average)
        "New_Home_Sales_lag1": "sum", # New Home Sales, lag1, sum 
        "Housing_Starts": "sum",  # Housing starts, sum
        "Housing_Starts_lag2": "sum", # Housing starts, lag 2 (sum)
        "Capacity_Utilization": "mean",  # Capacity utilization, average
        "New_Orders_Durable_Goods": "sum",  # New orders for durable goods, sum
        "Interest_Rate_lag1": "mean",  # Interest rate, lag1 (average)
        "Unemployment": "mean",  # Unemployment rate, average
        "junk_bond_spread": "mean",  # Junk bond spread, average
        "junk_bond_spread_lag1": "mean",  # Junk bond spread, lag1 (average)
        "junk_bond_spread_lag2": "mean",  # Junk bond spread, lag2 (average)
        "junk_bond_spread_lag3": "mean",  # Junk bond spread, lag3 (average)
        "junk_bond_spread_lag4": "mean",  # Junk bond spread, lag4 (average)
        "dummy": "mean"  # Dummy variable, sum 
    }

    # Extract the last available GDP value each quarter
    gdp_data = df[['GDP']].resample('QS').last()  

    indicators_data = pd.DataFrame()

    # Remove columns that should not be aggregated
    if "date" in df.columns:
        df_indicators = df.drop(columns=["date", "GDP"], errors="ignore")
    else:
        df_indicators = df.drop(columns=["GDP"], errors="ignore")

    for col in df_indicators.columns:  
        if col in aggregation_rule:
            method = aggregation_rule[col]
            if method == "mean":
                indicators_data[col] = df_indicators[col].resample('QS').mean()
            elif method == "sum":
                indicators_data[col] = df_indicators[col].resample('QS').sum()
        else:
            # Default to mean if there are columns not listed in aggregation_rule
            indicators_data[col] = df_indicators[col].resample('QS').mean()

    quarterly_df = gdp_data.merge(indicators_data, left_index=True, right_index=True, how='left')
    quarterly_df = quarterly_df.reset_index()
    quarterly_df["date"] = pd.to_datetime(quarterly_df["date"], format='%Y-%m')

    # Run line below to check output
    #print("output from aggregate_indicators", quarterly_df)
    return quarterly_df

def create_lag_features(df, exclude_columns, max_lag):
    """
    Creates lag features for time series data.

    This function creates lagged versions of each column in the DataFrame up to a specified maximum lag, 
    excluding columns specified in `exclude_columns`.

    Args:
        df (pd.DataFrame): The input DataFrame containing time series data.
        exclude_columns (list): List of columns to exclude from lagging.
        max_lag (int): The maximum lag to create for each column.

    Returns:
        pd.DataFrame: The DataFrame with lagged features added.
    """
   
    if "date" in df.columns:  
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    for col in df.columns:
        if col not in exclude_columns:
            for lag in range(1, max_lag + 1):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    return df

def get_missing_months(df, date_column="date"):
    """
    Calculates the number of months required to complete the current quarter.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        date_column (str): The name of the date column in the DataFrame.

    Returns:
        int: The number of months required to complete the current quarter.
    """

    df[date_column] = pd.to_datetime(df[date_column])

    latest_date = df[date_column].max()
    
    latest_month = latest_date.month
    
    # Calculate how many months remain to complete the current quarter
    months_to_complete_quarter = (3 - ((latest_month ) % 3)) % 3

    # add 3 more months (1 quarter)
    total_months_to_add = months_to_complete_quarter + 3

    return total_months_to_add

def add_missing_months(df, date_column="date"):
    """
    Adds missing months to a DataFrame to complete the current quarter.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        date_column (str): The name of the date column in the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing months added.
    """

    # Calculate how many months are missing
    num_extra_rows = get_missing_months(df, date_column)

    if num_extra_rows > 0:
        latest_date = pd.to_datetime(df[date_column].max())

        # Generate new dates starting from the next month after the latest date
        new_dates = pd.date_range(
            start=latest_date + pd.DateOffset(months=1),
            periods=num_extra_rows,
            freq='MS'
        )

        new_rows = pd.DataFrame({date_column: new_dates})

        # Append to original dataframe
        df = pd.concat([df, new_rows], ignore_index=True)

    df = df.sort_values(by=date_column).reset_index(drop=True)
    return df

