"""
This file contains functions for forecasting missing values in time series data using AutoRegressive models.
The `record_months_to_forecast` function identifies the months where data is missing and need to be predicted.
The `forecast_indicators` function forecasts the missing values for different predictor variables using the AutoReg model 
with 3 lags, while skipping over columns like GDP and other excluded columns.

Functions:
- `record_months_to_forecast`: Identifies the months with missing data for each predictor variable.
- `forecast_indicators`: Forecasts missing values for the predictor variables using AutoRegressive models (AR) on the time series data.
"""

import pandas as pd
import numpy as np

from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg

import warnings
warnings.simplefilter(action='ignore', category=Warning)

def record_months_to_forecast(df, predictors):
    """
    Identifies months that need forecasting for each predictor variable based on missing data.
    
    This function checks for missing data in the time series for each predictor variable and identifies which months
    need to be forecasted. The months are determined by finding the last known date for each predictor and then 
    identifying the subsequent missing months. It returns a dictionary where keys are the predictor names and the 
    values are lists of months that need forecasting.

    Args:
    - df (pd.DataFrame): DataFrame with dates as index and predictors as columns.
    - predictors (list): List of predictor variable names to check for missing data.

    Returns:
    - dict: A dictionary where the keys are predictor names and values are lists of months with missing data 
            that need forecasting.
    """
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    months_to_forecast = {} 

    for col in predictors:
        months_to_forecast[col] = [] 

        # last_known_index is the last row where data was released for the predictor
        last_known_index = df[col].dropna().index.max() if df[col].dropna().size > 0 else None

        if last_known_index:
            last_known_date = pd.to_datetime(last_known_index) 

            next_date = last_known_date + pd.DateOffset(months=1)

            # Iterate to the end of the dataframe to add all the dates where data is unreleased
            while next_date in df.index and pd.isna(df.loc[next_date, col]):
                months_to_forecast[col].append(next_date)  
                next_date += pd.DateOffset(months=1)

    # Run the line below to see the months_to_forecast
    #print(months_to_forecast)
    return months_to_forecast


def forecast_indicators(df, 
                        exclude=["date", "GDP", "gdp_growth", "gdp_growth_lag1", "gdp_growth_lag2", "gdp_growth_lag3", "gdp_growth_lag4"], # These columns should not be forecasted 
                        # lag_dict stores the optimal lags for each indicator, determined using BIC 
                        # optimal lags is stored in a dictionary and not re-run each time to reduce computation time
                        lag_dict={
                            'Capacity_Utilization': 4,
                            'Construction_Spending': 4,
                            'Housing_Starts': 4,
                            'Industrial_Production_lag1': 4,
                            'Industrial_Production_lag3': 4,
                            'Interest_Rate_lag1': 3,
                            'New_Orders_Durable_Goods': 4,
                            'Nonfarm_Payrolls': 4,
                            'Trade_Balance': 4,
                            'Trade_Balance_lag1': 4,
                            'Unemployment': 4,
                            'junk_bond_spread': 2,
                            'junk_bond_spread_lag1': 2,
                            'junk_bond_spread_lag2': 2,
                            'junk_bond_spread_lag3': 2,
                            'junk_bond_spread_lag4': 2,
                            'yield_spread': 4,
                            'yield_spread_lag1': 4,
                            'yield_spread_lag2': 4,
                            'yield_spread_lag3': 4,
                            'yield_spread_lag4': 4,
                            'Business_Inventories': 4,
                            'CPI': 4,
                            'Consumer_Confidence_Index': 4,
                            'Core_PCE': 4,
                            'Crude_Oil': 4,
                            'Industrial_Production': 4,
                            'Interest_Rate': 3,
                            'New_Home_Sales': 4,
                            'PPI': 4,
                            'Personal_Income': 4,
                            'Retail_Sales': 4,
                            'Three_Month_Treasury_Yield': 4,
                            'Wholesale_Inventories': 4
                        }):

    """
    Forecasts missing values for predictor variables using AutoRegressive models.
    Uses provided lags if lag_dict is given; otherwise, selects optimal lags using AIC.

    Args:
    - df (pd.DataFrame): Time series DataFrame with 'date' and predictors.
    - exclude (list): Columns to exclude from forecasting.
    - lag_dict (dict, optional): Dictionary specifying the lag to use per column. If None, AIC-based selection is used.

    Returns:
    - pd.DataFrame: DataFrame with missing predictor values filled using AR models.
    - dict: Dictionary of optimal (or used) lags per variable.
    """
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    df = df.set_index("date")
    predictors = df.columns.difference(exclude)

    months_to_forecast = record_months_to_forecast(df, predictors)

    final_lag_dict = {} if lag_dict is None else lag_dict.copy()  

    for col in predictors:

        # dummy variable should not be forecasted, fill values with 0
        if col == "dummy":
            df[col].fillna(0, inplace=True)
            continue

        if col in months_to_forecast and months_to_forecast[col]:
            for forecast_date in months_to_forecast[col]:
                data_before_forecast = df.loc[df.index <= forecast_date, col].dropna()

                # Safety condition to ensure series greater than length 10
                if len(data_before_forecast) < 10:
                    continue

                try:
                    if col in final_lag_dict:
                        best_lag = final_lag_dict[col]

                    # If a new indicator is being used, find optimal lags with BIC
                    else: 
                        best_aic = np.inf
                        best_lag = 1
                        max_lags = 4

                        for lag in range(1, max_lags + 1):
                            try:
                                model = AutoReg(data_before_forecast, lags=lag, old_names=False).fit()
                                if model.aic < best_aic:
                                    best_aic = model.aic
                                    best_lag = lag
                            except:
                                continue

                        final_lag_dict[col] = best_lag

                    # AR model prediction
                    final_model = AutoReg(data_before_forecast, lags=best_lag, old_names=False).fit()
                    predicted_value = final_model.predict(start=len(data_before_forecast), end=len(data_before_forecast)).iloc[0]

                    # Store the prediction for the next step of iterated forecasting
                    df.loc[forecast_date, col] = predicted_value

                except Exception as e:
                    pass 

    df = df.reset_index().sort_values(by="date").reset_index(drop=True)
    return df
