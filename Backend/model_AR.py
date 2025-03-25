import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from data_processing import aggregate_indicators

def model_AR(df, target_variable: str = "GDP"):
    """
    Takes a DataFrame of macroeconomic data, trains an AR (AutoRegressive) model, 
    and returns the GDP nowcast for the next time step.

    1. aggregate data to quarterly
    2. 

    Args:
        df (pd.DataFrame): DataFrame containing macroeconomic indicators.
        target_variable (str): Column name to nowcast (default: "GDP").

    Returns:
        float: Forecasted GDP value for the next available quarter.
    """

    # In the future when refactoring the code make sure that these following 3 lines are done once at the start of data cleaning and preprocessing
    # so that it doesnt have to be run at the start of every model.
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")
    df = df.reset_index(drop=True)
    #df = df.set_index("date")

    # aggregate
    df = aggregate_indicators(df)

    train_df = df[df["gdp_growth"].notna()].copy()
    forecast_df = df[df["gdp_growth"].isna()].copy()

    train_series = train_df["gdp_growth"]

    # Automatically select AR order, forcing MA order to be 0
    model = pm.auto_arima(train_series, seasonal=False, stationary=True, max_order=None, suppress_warnings=True, d=0, q=0)
    ar_order = model.order[0]

    ar_order = 2
    #print(ar_order)
    ar_model = ARIMA(train_series, order=(ar_order, 0, 0)).fit()

    print("AR Model Coefficients:", ar_model.params)
    print(ar_model.summary())  

    # generate forecast
    steps = len(forecast_df)
    gdp_growth_forecast = ar_model.forecast(steps=steps)

    #print("RAW growth rate FORECAST", gdp_growth_forecast)

    last_actual_gdp = train_df["GDP"].iloc[-1]
    gdp_forecast = []

    for growth in gdp_growth_forecast:
        next_gdp = last_actual_gdp * np.exp(growth / 400)
        gdp_forecast.append(next_gdp)
        last_actual_gdp = next_gdp

    # Assign forecasts back to forecast_df
    forecast_df = forecast_df.copy()
    forecast_df["gdp_growth_forecast"] = gdp_growth_forecast.values
    forecast_df["Nowcasted_GDP"] = gdp_forecast

    # Return results with corresponding forecasted dates
    return forecast_df[["date", "gdp_growth_forecast", "Nowcasted_GDP"]]


if __name__ == "__main__":
    file_path = "../Data/bridge_df.csv"
    df = pd.read_csv(file_path)
    next_gdp = model_AR(df)
    
    print("Nowcasted GDP for the most recent quarter:", next_gdp)
