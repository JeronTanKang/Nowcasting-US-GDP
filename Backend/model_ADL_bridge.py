"""
This file contains functions for forecasting GDP and related indicators using an Autoregressive Distributed Lag (ADL) bridge model.
The `fit_ols_model` function fits an Ordinary Least Squares (OLS) regression model to GDP growth using a set of predictor variables.
The `model_ADL_bridge` function aggregates monthly indicators, fits an OLS model, and generates nowcasts for GDP growth and GDP levels iteratively.

Functions:
- `fit_ols_model`: Fits an OLS regression model to the given data with GDP as the dependent variable and a set of independent variables.
- `model_ADL_bridge`: Aggregates monthly data to quarterly, fits an OLS model to estimate coefficients, forecasts missing values, and generates nowcasts for GDP growth and GDP levels.
"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from data_processing import aggregate_indicators
from forecast_bridge_indicators import record_months_to_forecast, forecast_indicators


def fit_ols_model(df, variables_to_exclude_from_ADL):
    """
    Fits an Ordinary Least Squares regression model using GDP as the dependent variable 
    and a set of indicators as independent variables.

    This function applies an OLS regression model where the dependent variable is GDP growth, 
    and the independent variables are a selection of economic indicators. It returns the fitted 
    OLS model that can be used for forecasting.

    Args:
    - df (pd.DataFrame): DataFrame containing 'date' as the index, 'GDP' and 'gdp_growth' columns, 
                          along with other economic indicator columns.
    - variables_to_exclude_from_ADL (list): List of column names to exclude from the independent variables in the model (e.g., 'GDP', 'gdp_growth').

    Returns:
    - model: Fitted OLS regression model.
    """

    df = df.set_index('date')
    df = df.iloc[::-1].reset_index(drop=True) # reverse row order


    X = df.drop(columns=variables_to_exclude_from_ADL) 
    Y = df['gdp_growth']  

    model = sm.OLS(Y, X).fit()

    # Run this line below to check coefficient estimates.
    #print(ols_model.summary())
    
    return model


def model_ADL_bridge(df):
    """
    Implements an Autoregressive Distributed Lag (ADL) bridge model for GDP forecasting.

    This function processes monthly data, aggregates it to quarterly frequency, fits an OLS regression 
    model, and forecasts missing values using iterated GDP growth predictions. It generates nowcasts for 
    GDP growth and GDP levels, and returns the nowcasted GDP values.

    The process involves:
    1. Aggregating monthly data to quarterly frequency.
    2. Fitting an OLS regression model on the aggregated data to estimate coefficients.
    3. Forecasting missing indicator values using the `forecast_indicators` function.
    4. Using iterated forecasting to predict GDP growth and GDP levels.

    Args:
    - df (pd.DataFrame): DataFrame with monthly data containing 'date', 'GDP', 'gdp_growth', and other economic indicators.

    Returns:
    - pd.DataFrame: DataFrame with the forecasted GDP growth and GDP levels for the forecasted dates.
    """

    df["date"] = pd.to_datetime(df["date"])

    # Aggregate to quarterly frequency
    df_aggregated = aggregate_indicators(df)

    # Only use quarters where GDP data has been released to fit model
    train_ols = df_aggregated[df_aggregated["GDP"].notna()].copy()

    # to exclude certain variables from being used in the model, add them to variables_to_exclude_from_ADL
    variables_to_exclude_from_ADL = ["GDP","gdp_growth","gdp_growth_lag2", "gdp_growth_lag3", "gdp_growth_lag4",
                    #"gdp_growth_lag1",
                    #"junk_bond_spread",
                    #"junk_bond_spread_lag1",
                    "junk_bond_spread_lag2",
                    "junk_bond_spread_lag3", "junk_bond_spread_lag4",  
                    "Industrial_Production_lag3",
                    "Trade_Balance",
                    #"Trade_Balance_lag1",
                    "Interest_Rate_lag1",
                    #"dummy",
                    #"Construction_Spending",
                    "yield_spread",
                    "yield_spread_lag1",
                    "yield_spread_lag2","yield_spread_lag3","yield_spread_lag4",
                    "Nonfarm_Payrolls"
                     ]

    ols_model = fit_ols_model(train_ols, variables_to_exclude_from_ADL)
    
    # Forecast values for monthly indicators that have not been released 
    monthly_indicators_forecasted = forecast_indicators(df, json_filename=os.path.join(os.path.dirname(__file__), "../Data/results_and_outputs/forecast_months_ADL.json"))

    # Aggregate to quarterly frequency
    quarterly_indicators_forecasted = aggregate_indicators(monthly_indicators_forecasted) 

    variables_to_exclude_from_ADL.append("date")

    # Create prediction data frame for values used to generate forecast
    predictors = quarterly_indicators_forecasted.drop(columns=variables_to_exclude_from_ADL, errors='ignore')

    # rows_to_forecast is the quarters where GDP is na (since they are unreleased) and an additional condition that they must be the 4 most recent quarters is added for safety
    rows_to_forecast = (quarterly_indicators_forecasted["GDP"].isna()) & (quarterly_indicators_forecasted.index >= quarterly_indicators_forecasted.index[-4]) 
    predictors_to_forecast = predictors[rows_to_forecast]
    dates_to_forecast = quarterly_indicators_forecasted["date"][rows_to_forecast]

    # Model prediction
    nowcast_growth = ols_model.predict(predictors_to_forecast)

    # Lists to store results for iterated forecasting of GDP growth 
    nowcast_growth = []

    # Dictionary to store gdp_growth_lag1 
    last_actual_growth = quarterly_indicators_forecasted["gdp_growth"][
        (quarterly_indicators_forecasted["gdp_growth"].notna()) & 
        (quarterly_indicators_forecasted["gdp_growth"] != 0)].iloc[-1]

    predicted_growth_dict = {0: last_actual_growth}
    
    predictors_to_forecast = predictors_to_forecast.reset_index(drop=True)

    # Loop over predictors_to_forecast to generate iterated forecasts
    for idx, row in predictors_to_forecast.iterrows():
        # adjusted_idx is used to retrieve gdp_growth_lag1 from predicted_growth_dict for iterated forecasting
        adjusted_idx = idx + 1 

        # If the current prediction requires gdp_growth_lag1 forecasted in the previous loop, retrieve value from predicted_growth_dict
        if pd.isna(row["gdp_growth_lag1"]) or row["gdp_growth_lag1"] == 0:
            row["gdp_growth_lag1"] = predicted_growth_dict.get(adjusted_idx - 1, last_actual_growth)

        #Predict GDP growth for the current row
        predicted_growth = ols_model.predict([row])[0]

        nowcast_growth.append(predicted_growth)

        #Update the current row's gdp_growth_lag1 with the predicted value from the dictionary
        predicted_growth_dict[adjusted_idx] = predicted_growth

    nowcast_df = pd.DataFrame({
        "date": pd.to_datetime(dates_to_forecast),
        "Nowcasted_GDP_Growth": nowcast_growth
    })

    nowcast_df = nowcast_df.reset_index(drop=True)

    return nowcast_df


# Code below is for testing each model individually
if __name__ == "__main__":
    file_path = "../Data/bridge_df.csv"
    df = pd.read_csv(file_path)

    print(model_ADL_bridge(df))







