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


def fit_ols_model(df, drop_variables):
    """
    Fits an Ordinary Least Squares regression model using GDP as the dependent variable 
    and a set of indicators as independent variables.

    This function applies an OLS regression model where the dependent variable is GDP growth, 
    and the independent variables are a selection of economic indicators. It returns the fitted 
    OLS model that can be used for forecasting.

    Args:
    - df (pd.DataFrame): DataFrame containing 'date' as the index, 'GDP' and 'gdp_growth' columns, 
                          along with other economic indicator columns.
    - drop_variables (list): List of column names to exclude from the independent variables in the model (e.g., 'GDP', 'gdp_growth').

    Returns:
    - model: Fitted OLS regression model.
    """

    #print("XXXXXX", df)
    df = df.set_index('date')
    df = df.iloc[::-1].reset_index(drop=True) # reverse row order



    # do i need this safety check?
    #df = df.dropna(subset=['GDP'])


    # Forward-fill missing values in indicators (just in case)
    df = df.fillna(method='ffill') 

    #print(df)

    Y = df['gdp_growth']  
    X = df.drop(columns=drop_variables) 



    # add constant for the intercept term
    #X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    #print(model.summary())
    
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
    #print("ORIGINAL DF", df)

    train_df = df[df["GDP"].notna()].copy()
    forecast_df = df[df["GDP"].isna()].copy()

    #print("PPPPPPEEEEEEEEKKKKK",df_aggregated)

    # step 1 aggregate
    # only want to pass into fit_ols_model the train data
    df_aggregated = aggregate_indicators(df)
    train_ols = df_aggregated[df_aggregated["GDP"].notna()].copy()

    # step 1: fit ols on quarterly data

    #print("PPPPPPEEEEEEEEKKKKK",train_ols)

    ##########################
    drop_variables = ["GDP","gdp_growth","gdp_growth_lag2", "gdp_growth_lag3", "gdp_growth_lag4",
                      
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
                    "yield_spread_lag2","yield_spread_lag3","yield_spread_lag4"
                     ]
    ##########################

    ols_model = fit_ols_model(train_ols, drop_variables)
    
    # step 2: for loop to forecast values for each indicator
    # OK DONE FOR STEP 2
    monthly_indicators_forecasted = forecast_indicators(df)
    #print("monthly_indicators_forecasted",monthly_indicators_forecasted)

    # step 3: generate nowcast
    #monthly_indicators_forecasted.index = pd.to_datetime(monthly_indicators_forecasted.index)
    quarterly_indicators_forecasted = aggregate_indicators(monthly_indicators_forecasted) # aggregate to quartlerly

    #print("quarterly_indicators_forecasted",quarterly_indicators_forecasted.tail(10))

    drop_variables.append("date")
    predictors = quarterly_indicators_forecasted.drop(columns=drop_variables, errors='ignore')
    #predictors = sm.add_constant(predictors, has_constant='add')  

    # where mask is the dates to forecast
    #print("serious check here", quarterly_indicators_forecasted.tail(10))

    # set -4 as the limit since there may be up to 4 quarters of gdp missing, depending on when this model is run

    mask = (quarterly_indicators_forecasted["GDP"].isna()) & (quarterly_indicators_forecasted.index >= quarterly_indicators_forecasted.index[-4]) 
    #mask = quarterly_indicators_forecasted["GDP"].isna()
    predictors_to_forecast = predictors[mask]
    dates_to_forecast = quarterly_indicators_forecasted["date"][mask]

    nowcast_growth = ols_model.predict(predictors_to_forecast)
    #print("OUTCOMES (gdp_growth forecast):", nowcast_growth)

    # Iterated forecasting for GDP growth and GDP
    nowcast_growth = []
    nowcasted_gdp_levels = []
    
    # Initialize the last known GDP
    last_actual_gdp = train_ols["GDP"].dropna().iloc[-1]  # Last known actual GDP

    # Initialize a dictionary to store predicted growth values (gdp_growth_lag1 and lag2)
    last_actual_growth = quarterly_indicators_forecasted["gdp_growth"][
    (quarterly_indicators_forecasted["gdp_growth"].notna()) & 
    (quarterly_indicators_forecasted["gdp_growth"] != 0)].iloc[-1]

    predicted_growth_dict = {0: last_actual_growth}
    

    predictors_to_forecast = predictors_to_forecast.reset_index(drop=True)
        
    #print("predictors_to_forecast", predictors_to_forecast)

    # Loop over predictors to forecast iteratively
    for idx, row in predictors_to_forecast.iterrows():
        adjusted_idx = idx + 1 

        # Step 1: Update the current row's gdp_growth_lag1 and lag2 with the predicted value from the dictionary
        if pd.isna(row["gdp_growth_lag1"]) or row["gdp_growth_lag1"] == 0:
            row["gdp_growth_lag1"] = predicted_growth_dict.get(adjusted_idx - 1, last_actual_growth)
        #if pd.isna(row["gdp_growth_lag2"]) or row["gdp_growth_lag2"] == 0:
        #    row["gdp_growth_lag2"] = predicted_growth_dict.get(adjusted_idx - 2, last_actual_growth)

        #print("prediction is made on this row \n",row)
        # Step 2: Predict GDP growth for the current row
        predicted_growth = ols_model.predict([row])[0]
        #print(idx, predicted_growth)
        nowcast_growth.append(predicted_growth)

        # Step 3: Convert GDP growth to GDP level for the current period
        next_gdp = last_actual_gdp * np.exp(predicted_growth / 400)  # Convert growth rate to GDP level
        nowcasted_gdp_levels.append(next_gdp)

        # Step 4: Update the last known GDP for the next iteration
        last_actual_gdp = next_gdp  # Update for next forecast

        # Step 5: Store the predicted growth in the dictionary for the next row
        predicted_growth_dict[adjusted_idx] = predicted_growth

        #print("iterated updating",predicted_growth_dict)

    nowcast_df = pd.DataFrame({
        "date": pd.to_datetime(dates_to_forecast),
        "Nowcasted_GDP_Growth": nowcast_growth,
        "Nowcasted_GDP": nowcasted_gdp_levels
    })

    nowcast_df = nowcast_df.reset_index(drop=True)

    return nowcast_df


if __name__ == "__main__":
    file_path = "../Data/bridge_df.csv"
    #file_path = "../Data/tseting_adl.csv"
    #file_path = "../Data/manual_testing.csv"
    df = pd.read_csv(file_path)
    #print("df going in", df)

    print(model_ADL_bridge(df))







