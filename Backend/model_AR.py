import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

def model_AR(df, target_variable: str = "GDP"):
    """
    Takes a DataFrame of macroeconomic data, trains an AR (AutoRegressive) model, 
    and returns the GDP nowcast for the next time step.

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
    df = df.set_index("date")
    df = df.dropna(subset=["GDP"])

    # for testing
    #df = df.iloc[:-10]

    print(df.tail(10))

    df["GDP_diff"] = df[target_variable].diff()

    # Drop NaN (first row will be NaN due to differencing)
    df = df.dropna(subset=["GDP_diff"])

    # Prepare training data
    train_series = df["GDP_diff"]


    # Automatically select AR order, forcing MA order to be 0
    model = pm.auto_arima(train_series, seasonal=False, stationary=True, max_order=None, suppress_warnings=True, d=0, q=0)
    ar_order = model.order[0]  # Extract AR order

    # Fit AR model
    #ar_order = 4
    print(ar_order)
    ar_model = ARIMA(train_series, order=(ar_order, 0, 0)).fit()

    print("AR Model Coefficients:", ar_model.params)
    print(ar_model.summary())  

    # Nowcast GDP for the next 2 available quarters
    gdp_diff_nowcast = ar_model.forecast(steps=2)

    # Convert differenced predictions back to actual GDP
    last_actual_gdp = df[target_variable].iloc[-1]  
    print("last_actual_gdp:", last_actual_gdp)
    gdp_nowcast = last_actual_gdp + gdp_diff_nowcast.cumsum() # reverse differencing

    gdp_nowcast_df = gdp_nowcast.to_frame(name="Nowcasted_GDP")
    gdp_nowcast_df = gdp_nowcast_df.reset_index()
    gdp_nowcast_df.rename(columns={"index": "date"}, inplace=True)

    return gdp_nowcast_df


if __name__ == "__main__":
    file_path = "../Data/lasso_indicators.csv"
    df = pd.read_csv(file_path)
    next_gdp = model_AR(df)
    
    print("Nowcasted GDP for the most recent quarter:", next_gdp)
