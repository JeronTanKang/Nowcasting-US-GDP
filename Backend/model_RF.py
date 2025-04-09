"""
This file contains the `model_RF` function, which generates GDP forecasts using Random Forest models for multiple forecast horizons (1-step, 2-step forecasts). 
It leverages macroeconomic indicators and their lagged values to predict future GDP growth, and then uses the forecasted GDP growth to calculate the nowcasted GDP levels for the next periods.

The process includes:
1. Aggregating macroeconomic data to quarterly frequency.
2. Creating lagged features for the indicators.
3. Training two Random Forest models for 1-step, 2-step forecasts.
4. Using the trained models to predict GDP growth and convert it to GDP levels.
5. Returning a DataFrame with the forecasted GDP growth and nowcasted GDP for the forecasted periods.

Functions:
- `model_RF`: Generates GDP nowcasts using Random Forest models for different forecast horizons (1-step, 2-step, and 3-step).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import QuarterBegin
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from data_processing import aggregate_indicators, create_lag_features
from model_ADL_bridge import forecast_indicators, record_months_to_forecast #AR(p) model 


def model_RF(df):

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")


    #extract col names for later use
    cols_to_keep = ['date', 'GDP', "gdp_growth", "dummy"]
    contemp_indicators = [col for col in df.columns if col not in cols_to_keep]

    #Sort date in ascending order
    df = df.sort_values(by='date', ascending=True)

    #Aggregate Data
    df_aggregated = aggregate_indicators(df) #gdp_growth is not aggregated 


 
    #lag variables
    exclude_columns = ["date", "GDP", "dummy"] #df to exclude later when laggin indicators
    df_lagged = create_lag_features(df_aggregated, exclude_columns, 6) # 6 lags of each indicator 




    #drop contemporary indicators and all lag1
    df_lagged = df_lagged.drop(columns=contemp_indicators)
    df_lagged = df_lagged.drop(columns=df_lagged.filter(regex='_lag1$').columns) 

    #create a df to store nowcasted values later
    df_to_pred = df_lagged.tail(2)

    #drop rows with no gdp val
    df_lagged = df_lagged.dropna(subset=["GDP"]) 

    #drop lags of each indicator depending on how many steps model is predicting
    def prepare_model_data(df, drop_lags, drop_growth_lag):
        df_model = df.drop(columns=[col for col in df.columns if any(col.endswith(f'_lag{i}') for i in drop_lags)])
        if drop_growth_lag:
            df_model = df_model.drop(columns=drop_growth_lag)
        #Sort date again 
        df_model = df_model.sort_values(by='date', ascending=True)
        #Drop NaN values created by lagging 
        df_model = df_model.dropna()
        #Drop 'Date' before defining X
        X = df_model.drop(columns=["gdp_growth", "GDP", "date"]) 
        y = df_model['gdp_growth']
        return X, y


    #Model 1 (1 step forecast, so take lag2 to lag 5)
    X_1, y_1 = prepare_model_data(df_lagged, [6], None)
    rf_model_1 = RandomForestRegressor(random_state=42).fit(X_1, y_1)

    ##Model 2 (2 step forecast, so take lag3 to lag6)
    X_2, Y_2 = prepare_model_data(df_lagged, [2], None)
    rf_model_2 = RandomForestRegressor(random_state=42).fit(X_2, Y_2)
    
    #ensure that df_pred is sorted in ascending order
    df_to_pred = df_to_pred.reset_index(drop=True).sort_values(by="date", ascending=True)
    df_to_pred_indicators = df_to_pred.drop(columns = ["date", "GDP", "gdp_growth"])
    df_to_pred = df_to_pred[["date", "gdp_growth"]]
    #for loop to fill in gdp_growth
    for index, row in df_to_pred.iterrows():
        # Access values in each row by column name
        date = row['date']
        gdp_growth = row['gdp_growth']
        if index == 0:
            #use model 1
            df_indicators = df_to_pred_indicators.iloc[[index]]
            df_indicators = df_indicators.drop(columns=[col for col in df_indicators.columns if any(col.endswith(f'_lag{i}') for i in [1, 6])])
            gdp_growth = rf_model_1.predict(df_indicators)

        elif index == 1:
            #use model 2
            df_indicators = df_to_pred_indicators.iloc[[index]]
            df_indicators = df_indicators.drop(columns=[col for col in df_indicators.columns if any(col.endswith(f'_lag{i}') for i in [1, 2])])
            gdp_growth = rf_model_2.predict(df_indicators)

        df_to_pred.at[index, 'gdp_growth'] = gdp_growth

    df_to_pred.rename(columns={"gdp_growth": "Nowcasted_GDP_Growth"}, inplace=True)

    return df_to_pred


if __name__ == "__main__":
    file_path = "../Data/tree_df.csv"

    df = pd.read_csv(file_path)
    print(model_RF(df))

