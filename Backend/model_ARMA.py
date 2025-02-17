import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

def arma_nowcast(file_path: str, target_variable: str = "GDP"):
    """
    Reads macroeconomic data, trains an ARMA model, and returns the GDP nowcast for the next time step.

    Args:
        file_path (str): Path to the macroeconomic dataset.
        target_variable (str): Column name to nowcast (default: "GDP").

    Returns:
        float: Forecasted GDP value for the next available quarter.
    """
    # Load dataset
    data = pd.read_csv(file_path, parse_dates=["date"])
    data = data.sort_values(by="date")
    data.set_index("date", inplace=True)

    # Filter only rows where GDP is available (no interpolation)
    gdp_data = data.dropna(subset=[target_variable])

    # Time series for training
    train_series = gdp_data[target_variable]

    # Fit ARMA model
    model = pm.auto_arima(train_series, seasonal=False, stationary=True)
    ar_order, ma_order = model.order[0], model.order[2]  # Extract AR and MA orders

    arma_model = ARIMA(train_series, order=(ar_order, 0, ma_order)).fit()

    # Forecast GDP for the next available quarter
    next_gdp_nowcast = arma_model.forecast(steps=1)[0]

    return next_gdp_nowcast


if __name__ == "__main__":
    file_path = "../Data/test_macro_data.csv"
    next_gdp = arma_nowcast(file_path)
    
    print("Nowcasted GDP for the most recent quarter:", next_gdp)