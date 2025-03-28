import numpy as np
import pandas as pd
import statsmodels.api as sm
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


    # drop all other columns, only gdp growth and its lags are needed for AR model
    # keep GDP for calculation purposes
    df = df[["date", "GDP", "gdp_growth", "gdp_growth_lag2", "gdp_growth_lag3"]]

    # aggregate
    df = aggregate_indicators(df)

    train_df = df[df["GDP"].notna()].copy()
    forecast_df = df[df["GDP"].isna()].copy()

    train_lagged = train_df.dropna(subset=["gdp_growth_lag2", "gdp_growth_lag3", "gdp_growth"]).copy()

    X = train_lagged[["gdp_growth_lag2", "gdp_growth_lag3"]]
    y = train_lagged["gdp_growth"]
    X = sm.add_constant(X)

    ols_model = sm.OLS(y, X).fit()
    #print("OLS Coefficients:\n", ols_model.params)
    #print(ols_model.summary())

    gdp_growth_forecast = []

    history = list(train_df["gdp_growth"].dropna().values[-3:])  # we need last 3 for lag2, lag3

    for _ in range(len(forecast_df)):
        #print("history", history)
        lag2 = history[-2]
        lag3 = history[-3]
        # Build x_input with correct column order
        x_input = pd.DataFrame([[1, lag2, lag3]], columns=["const", "gdp_growth_lag2", "gdp_growth_lag3"])
        forecast = ols_model.predict(x_input)[0]
        gdp_growth_forecast.append(forecast)
        history.append(forecast)

    # Convert growth to GDP levels
    last_actual_gdp = train_df["GDP"].iloc[-1]
    gdp_forecast = []

    for growth in gdp_growth_forecast:
        next_gdp = last_actual_gdp * np.exp(growth / 400)
        gdp_forecast.append(next_gdp)
        last_actual_gdp = next_gdp

    # Assign forecasts
    forecast_df = forecast_df.copy()
    forecast_df["gdp_growth_forecast"] = gdp_growth_forecast
    forecast_df["Nowcasted_GDP"] = gdp_forecast

    return forecast_df[["date", "gdp_growth_forecast", "Nowcasted_GDP"]]


if __name__ == "__main__":
    file_path = "../Data/bridge_df.csv"
    #file_path = "../Data/tseting_adl.csv"
    
    df = pd.read_csv(file_path)
    next_gdp = model_AR(df)
    
    print("Nowcasted GDP for the most recent quarter:", next_gdp)
