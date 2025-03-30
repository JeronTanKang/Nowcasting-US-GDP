import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np


def is_stationary(series, significance_level=0.05):
    if series.dropna().shape[0] < 2: # ensure enough data points
        return False
    
    adf_test = adfuller(series.dropna(), autolag="AIC")
    p_value = adf_test[1]
    
    return p_value <= significance_level  

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
    THIS FUNCTION IS DIFF FROM THE REST MUST SET DATE AS INDEX NOOOOTTT RANGE INDEX
    update on 23 march. seems like i can feed df in 

    Function that takes in df with monthly frequency indicators and GDP.
    - Converts indicators to quarterly frequency using specified aggregation rules.
    - GDP remains unchanged (takes the only available value per quarter).

    Returns:
    - DataFrame with quarterly frequency.
    """


    df = df.set_index('date')

    #print("THIS IS WHAT WORKS FOR aggregate_indicators", df)

    aggregation_rule = {
        "GDP": "sum",  # GDP should be aggregated using mean
        "gdp_growth": "sum",  # GDP growth, as a rate, should be averaged
        "gdp_growth_lag1": "sum",  # Lagged GDP growth, average
        "gdp_growth_lag2": "sum",  # Lagged GDP growth, average
        "gdp_growth_lag3": "sum",  # Lagged GDP growth, average
        "gdp_growth_lag4": "sum",  # Lagged GDP growth, average
        "Nonfarm_Payrolls": "sum",  # Sum of non-farm payrolls
        "Construction_Spending": "sum",  # Sum of construction spending
        "Trade_Balance_lag1": "sum",  # Trade balance, lag1 (sum)
        "Industrial_Production_lag1": "mean",  # Industrial production, lag1 (average)
        "Industrial_Production_lag3": "mean",  # Industrial production, lag3 (average)
        "Housing_Starts": "sum",  # Housing starts, sum
        "Capacity_Utilization": "mean",  # Capacity utilization, average
        "New_Orders_Durable_Goods": "sum",  # New orders for durable goods, sum
        "Interest_Rate_lag1": "mean",  # Interest rate, lag1 (average)
        "Unemployment": "mean",  # Unemployment rate, average
        "junk_bond_spread": "mean",  # Junk bond spread, average
        "junk_bond_spread_lag1": "mean",  # Junk bond spread, lag1 (average)
        "junk_bond_spread_lag2": "mean",  # Junk bond spread, lag2 (average)
        "junk_bond_spread_lag3": "mean",  # Junk bond spread, lag3 (average)
        "junk_bond_spread_lag4": "mean",  # Junk bond spread, lag4 (average)
        "dummy": "mean"  # Dummy variable, sum (usually used for counting events)
    }

    gdp_data = df[['GDP']].resample('QS').last()  # extract the last available GDP value per quarter

    indicators_data = pd.DataFrame()

    if "date" in df.columns:
        df_indicators = df.drop(columns=["date", "GDP"], errors="ignore")
    else:
        df_indicators = df.drop(columns=["GDP"], errors="ignore")
    #df_indicators = df.drop(columns=["GDP"])

    for col in df_indicators.columns:  
        if col in aggregation_rule:
            method = aggregation_rule[col]
            if method == "mean":
                indicators_data[col] = df_indicators[col].resample('QS').mean()
            elif method == "sum":
                indicators_data[col] = df_indicators[col].resample('QS').sum()
        else:
            # Default to 'mean' for columns not listed in aggregation_rule
            indicators_data[col] = df_indicators[col].resample('QS').mean()

    quarterly_df = gdp_data.merge(indicators_data, left_index=True, right_index=True, how='left')
    quarterly_df = quarterly_df.reset_index()
    #quarterly_df['date'] = quarterly_df['date'].dt.strftime('%Y-%m')
    quarterly_df["date"] = pd.to_datetime(quarterly_df["date"], format='%Y-%m')

    #quarterly_df = quarterly_df.iloc[::-1].reset_index(drop=True) # reverse row order before returning

    #print("THIS IS WHAT COMES OUT OF aggregate_indicators", quarterly_df)
    return quarterly_df

#Function to create lag features
def create_lag_features(df, exclude_columns, max_lag):
   
    if "date" in df.columns:  
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")  # Convert date if it exists

    for col in df.columns:
        if col not in exclude_columns:
            for lag in range(1, max_lag + 1):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    return df

def get_missing_months(df, date_column="date"):
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Get the latest date in the dataset
    latest_date = df[date_column].max()
    
    # Get the month number (1 to 12)
    latest_month = latest_date.month

    #print("latest_month", latest_month)
    
    # Calculate how many months remain to complete the current quarter
    months_to_complete_quarter = (3 - ((latest_month ) % 3)) % 3

    # Total months to add: remaining months of current quarter + 2 quarters (6 months)
    total_months_to_add = months_to_complete_quarter + 3

    return total_months_to_add

def add_missing_months(df, date_column="date"):
    # Calculate how many months are missing
    num_extra_rows = get_missing_months(df, date_column)

    if num_extra_rows > 0:
        # Get the latest date in the dataset
        latest_date = pd.to_datetime(df[date_column].max())

        # Generate new dates starting from the next month after the latest date
        new_dates = pd.date_range(
            start=latest_date + pd.DateOffset(months=1),
            periods=num_extra_rows,
            freq='MS'
        )

        # Create new rows with NaN values except the date column
        new_rows = pd.DataFrame({date_column: new_dates})

        # Append to the original dataframe
        df = pd.concat([df, new_rows], ignore_index=True)

    # Sort and reset index
    df = df.sort_values(by=date_column).reset_index(drop=True)
    return df

