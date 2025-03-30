import pandas as pd
import numpy as np

from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg

import warnings
warnings.simplefilter(action='ignore', category=Warning)

def record_months_to_forecast(df, predictors):
    """
    Identifies months that need forecasting for each predictor.

    Args:
    - df (pd.DataFrame): DataFrame with dates as index and predictors as columns.
    - predictors (list): List of predictor variable names.

    Returns:
    - dict: A dictionary where keys are predictor names and values are lists of months to forecast.
    """
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    #print("input to record months:", df)

    months_to_forecast = {}  # Dict to store missing months for each predictor

    for col in predictors:
        months_to_forecast[col] = [] 

        last_known_index = df[col].dropna().index.max() if df[col].dropna().size > 0 else None

        if last_known_index:
            last_known_date = pd.to_datetime(last_known_index)  # Ensure it's datetime

            next_date = last_known_date + pd.DateOffset(months=1)

            while next_date in df.index and pd.isna(df.loc[next_date, col]):
                months_to_forecast[col].append(next_date)  # Store as datetime
                next_date += pd.DateOffset(months=1)

    return months_to_forecast

def forecast_indicators(df, exclude=["date","GDP","gdp_growth","gdp_growth_lag2","gdp_growth_lag3","gdp_growth_lag4"]):
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    df = df.set_index("date") 


    """ 
    Handles missing values only for predictor variables (not GDP).
    - Starts 3 months before the current month.
    - If a month has missing data, predict using AR(p).
    - If a month has data, use it to predict the next month.
    - Works column by column, excluding GDP (Y variable).
    """

    predictors = df.columns.difference(exclude)

    # Identify which months need to be forecasted for each predictor
    months_to_forecast = record_months_to_forecast(df, predictors)
    #print("Months to forecast:", months_to_forecast)

    #print("LOOOOOOKKK HERREEEE", df)

    for col in predictors:
        if col == "dummy":
            df[col].fillna(0, inplace=True)
            continue  # skip forecasting for dummy
        if col in months_to_forecast and months_to_forecast[col]:
            indic_data = df[col].dropna().sort_index()

            # Iterate over each month to forecast one by one
            for forecast_date in months_to_forecast[col]:
                # Get the data before the current forecast_date
                data_before_forecast = df.loc[df.index <= forecast_date, col].dropna()

                try:
                    # Fit the model to the data before the current forecast date
                    final_model = AutoReg(data_before_forecast, lags=3, old_names=False).fit()

                    # Predict the missing value for the current date
                    predicted_value = final_model.predict(start=len(data_before_forecast), end=len(data_before_forecast)).iloc[0]

                    # Update the DataFrame with the predicted value at the missing date
                    df.loc[forecast_date, col] = predicted_value

                    # Optionally, print the predicted value for debugging
                    #print(f"Predicted {col} for {forecast_date}: {predicted_value}")

                except Exception as e:
                    pass
                    #print(f"Could not forecast {col} for {forecast_date} due to: {e}")

    df = df.reset_index()
    df = df.sort_values(by="date").reset_index(drop=True)
    #print("Returned by forecast_indicators:", df)
    #print("prediction df", df.tail(13))
    return df
