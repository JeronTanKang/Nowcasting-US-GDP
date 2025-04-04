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

    # Initial cleaning (should ideally be done once at the beginning of the pipeline)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # Keep only relevant columns
    df = df[["date", "GDP", "gdp_growth", "gdp_growth_lag2", "gdp_growth_lag3"]]

    #print("model AR before remove", df.tail(10))


    # Aggregate (if needed — assuming your aggregate_indicators handles this properly)
    df = aggregate_indicators(df)

    # Split train and forecast sets
    train_df = df.iloc[:-2].copy()
    forecast_df = df.iloc[-2:].copy() # benchmark model will always only forecast the last 2 quarters in the df

    # Drop rows with missing lags or target
    train_lagged = train_df.dropna(subset=["gdp_growth_lag2", "gdp_growth_lag3", "gdp_growth"]).copy()

    # Prepare training data
    X_train = train_lagged[["gdp_growth_lag2", "gdp_growth_lag3"]]
    y_train = train_lagged["gdp_growth"]
    X_train = sm.add_constant(X_train)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(y_train, errors='coerce')

    # Fit OLS model
    ols_model = sm.OLS(y_train, X_train).fit()

    #print(ols_model.summary())

    # === DIRECT FORECAST === #
    # Use lag2 and lag3 from forecast_df as-is
    X_forecast = forecast_df[["gdp_growth_lag2", "gdp_growth_lag3"]].copy()
    X_forecast = sm.add_constant(X_forecast)
    X_forecast = X_forecast.apply(pd.to_numeric, errors='coerce')

    gdp_growth_forecast = ols_model.predict(X_forecast).values

    # Convert growth to GDP level (starting from last known actual GDP)
    last_actual_gdp = train_df["GDP"].iloc[-1]
    gdp_forecast = []

    for growth in gdp_growth_forecast:
        next_gdp = last_actual_gdp * np.exp(growth / 400)
        gdp_forecast.append(next_gdp)
        last_actual_gdp = next_gdp  # We can keep updating this since it's just for level calc

    # Assign forecast values
    forecast_df["Nowcasted_GDP_Growth"] = gdp_growth_forecast
    forecast_df["Nowcasted_GDP"] = gdp_forecast

    forecast_df = forecast_df.reset_index(drop=True)
    #print(forecast_df[["date", "Nowcasted_GDP_Growth", "Nowcasted_GDP"]])

    return forecast_df[["date", "Nowcasted_GDP_Growth", "Nowcasted_GDP"]]


if __name__ == "__main__":
    file_path = "../Data/bridge_df.csv"
    #file_path = "../Data/tseting_adl.csv"
    
    df = pd.read_csv(file_path)
    next_gdp = model_AR(df)
    
    print("Nowcasted GDP for the most recent quarter:", next_gdp)
