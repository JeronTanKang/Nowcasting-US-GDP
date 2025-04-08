"""
This file contains a function to generate GDP nowcasts using an AutoRegressive (AR) model.
The `model_AR` function trains an AR model on macroeconomic indicators and returns the forecasted GDP for the next time step.

Function:
- `model_AR`: Uses AutoRegressive (AR) modeling to nowcast GDP for the next available quarter based on macroeconomic data.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from data_processing import aggregate_indicators

def model_AR(df):
    """
    Generates a GDP nowcast for the next 2 quarters using a direct AutoRegressive (AR) model.
    The model uses gdp_growth_lag2 and gdp_growth_lag3 to directly forecast each quarter in forecast_df.

    Args:
        df (pd.DataFrame): DataFrame containing 'GDP', 'gdp_growth', and its lagged values.

    Returns:
        pd.DataFrame: A DataFrame with the forecasted GDP growth and the nowcasted GDP values for the forecasted period.
    """

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # Keep only the columns needed for AR model
    df = df[["date", "GDP", "gdp_growth", "gdp_growth_lag2", "gdp_growth_lag3"]]


    # Aggregate to quarterly frequency
    df = aggregate_indicators(df)

    # Split the dataframe into the portion for model fitting (train_df) and for forecasting (forecast_df)
    train_df = df.iloc[:-2].copy()
    forecast_df = df.iloc[-2:].copy() # Benchmark AR model will always only forecast the last 2 quarters in the df as specified in the technical documentation.

    # Drop rows with missing lags or gdp_growth
    train_lagged = train_df.dropna(subset=["gdp_growth_lag2", "gdp_growth_lag3", "gdp_growth"]).copy()

    # Prepare data for model fit
    X_train = train_lagged[["gdp_growth_lag2", "gdp_growth_lag3"]]
    y_train = train_lagged["gdp_growth"]
    X_train = sm.add_constant(X_train)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(y_train, errors='coerce')

    # Fit OLS model
    ols_model = sm.OLS(y_train, X_train).fit()

    # Run this line below to check coefficient estimates.
    #print(ols_model.summary())

    # Direct forecast with lag2 and lag3 of gdp growth
    X_forecast = forecast_df[["gdp_growth_lag2", "gdp_growth_lag3"]].copy()
    X_forecast = sm.add_constant(X_forecast)
    X_forecast = X_forecast.apply(pd.to_numeric, errors='coerce')

    gdp_growth_forecast = ols_model.predict(X_forecast).values

    # Assign forecast values
    forecast_df["Nowcasted_GDP_Growth"] = gdp_growth_forecast

    forecast_df = forecast_df.reset_index(drop=True)

    return forecast_df[["date", "Nowcasted_GDP_Growth"]]


if __name__ == "__main__":
    file_path = "../Data/bridge_df.csv"
    
    df = pd.read_csv(file_path)
    next_gdp = model_AR(df)
    
    print("model_AR output:", next_gdp)
