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

def record_months_to_forecast(df, predictors):
    """
    Identifies months that need forecasting for each predictor.

    Args:
    - df (pd.DataFrame): DataFrame with dates as index and predictors as columns.
    - predictors (list): List of predictor variable names.

    Returns:
    - dict: A dictionary where keys are predictor names and values are lists of months to forecast.
    """

    months_to_forecast = {}  # dict to store missing months for each predictor

    for col in predictors:
        months_to_forecast[col] = [] 

        # this identifies the latest row with reported data
        last_known_index = df[col].dropna().index[0] if df[col].dropna().size > 0 else None
        #print(col, last_known_index)

        if last_known_index:
            last_known_date = pd.to_datetime(last_known_index)

            next_date = (last_known_date + pd.DateOffset(months=1)).strftime('%Y-%m')
            #print(next_date)

            # iterate forward until reach the last date in df
            #while next_date in df.index and pd.isna(df.loc[next_date, col]):
            while next_date in df.index.to_list() and pd.isna(df.loc[next_date, col]):
                months_to_forecast[col].append(next_date)
                next_date = (pd.to_datetime(next_date) + pd.DateOffset(months=1)).strftime('%Y-%m')

    return months_to_forecast

def forecast_indicators(df, exclude=["date", "GDP"]):
    df = df.set_index("date") 
    df_temp = df.copy()

    """ 
    Handles missing values only for predictor variables (not GDP).
    - Starts 3 months before the current month.
    - If a month has missing data, predict using AR(p).
    - If a month has data, use it to predict the next month.
    - Works column by column, excluding GDP (Y variable).
    """

    today_dt_object = datetime.today().replace(day=1)
    today = today_dt_object.strftime('%Y-%m')
    #print("PRINTING FROM forecast_indicators",df)
    #print("TODAY MONTH:", today_dt_object.month)

    month_offset = (today_dt_object.month - 1) % 3  
    months_to_add = [today_dt_object + pd.DateOffset(months=i+1) for i in range(2 - month_offset)]
    
    # add months not already in the index to make up the current quarter
    months_to_add = [date.strftime('%Y-%m') for date in months_to_add if date.strftime('%Y-%m') not in df.index]
    
    if months_to_add:
        new_rows = pd.DataFrame(index=months_to_add, columns=df.columns)
        df = pd.concat([df, new_rows]).sort_index(ascending=False)


    #print(f" Added rows: {months_to_add}")

    # months we will be forecasting data with AR(p)
    end_date = today
    start_date = (datetime.today().replace(day=1) - pd.DateOffset(months=3)).strftime('%Y-%m')

    #print("PRINT DATES:", start_date, end_date)

    # exclude gdp from the predictor list
    predictors = df.columns.difference(exclude)

    months_to_forecast = record_months_to_forecast(df, predictors)
    #print(months_to_forecast)

    for col in predictors:
        if col in months_to_forecast and months_to_forecast[col]:
            #end_month = min(months_to_forecast[col])  # earliest missing month 
            #start_month = max(months_to_forecast[col])    # latest missing month 
            #print(start_month, end_month)

            indic_data = df[col].dropna()

            # forecast start and end indices
            forecast_start = 0
            forecast_end = forecast_start + len(months_to_forecast[col]) - 1
            print(f"{col}:  forecasting from {forecast_start} to {forecast_end}")

            # fit AR model
            indic_data = indic_data.sort_index(ascending=True)
            final_model = AutoReg(indic_data, lags=5).fit()

            predicted_values = final_model.predict(start=forecast_start, end=forecast_end)
            #print("predicted_values:", predicted_values)

            predicted_series = pd.Series(predicted_values.values, index=pd.to_datetime(months_to_forecast[col]))
            df.update(predicted_series.to_frame(name=col))
 
    # standardizing the index of the output dataframe for modular usage
    df = df.reset_index()  # move index to a col
    df.rename(columns={"index": "date"}, inplace=True)  # rename that col to date
    df = df.set_index('date') # set date as index
    return df


def fit_ols_model(df):
    """
    Fits an OLS regression model using GDP as the dependent variable
    and the other 10 indicators as independent variables.

    Args:
    - df (pd.DataFrame): DataFrame with 'date' as index, 'GDP' column, and 10 indicators.

    Returns:
    - results: Fitted OLS model
    """

    df = df.set_index('date')

    # do i need this safety check?
    #df = df.dropna(subset=['GDP'])


    # Forward-fill missing values in indicators (just in case)
    df = df.fillna(method='ffill') 

    #print(df)

    Y = df['GDP']  
    X = df.drop(columns=['GDP']) 

    # add constant for the intercept term
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    print(model.summary())
    
    return model


def model_bridge(df):

    """
    indicators in model_bridge will always be monthly frequency. 
    they will only be converted to quarterly frequency when fitting the model to estimate coefficients, and when predicting nowcast (using aggregate_indicators)
    """

    df_temp = df.copy()

    # just ensuring index is datetime object
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m")
    df = df.set_index('date')

    # step 1: fit ols on quarterly data
    ols_model = fit_ols_model(aggregate_indicators(df))

    # step 2: for loop to forecast values for each indicator
    monthly_indicators_forecasted = forecast_indicators(df_temp)

    # step 3: generate nowcast
    monthly_indicators_forecasted.index = pd.to_datetime(monthly_indicators_forecasted.index)
    quarterly_indicators_forecasted = aggregate_indicators(monthly_indicators_forecasted) # aggregate to quartlerly


    predictors = quarterly_indicators_forecasted.drop(columns=['GDP'], errors='ignore')

    #print(ols_model.summary())
    predictors = sm.add_constant(predictors, has_constant='add')  
    #print(predictors.dtypes)


    date_column = predictors["date"].copy()
    predictors = predictors.drop(columns=["date"])

    #print(predictors)
    
    nowcast_gdp = ols_model.predict(predictors)

    nowcast_df = pd.DataFrame({"date": date_column, "Nowcasted_GDP": nowcast_gdp.values})

    # date to datetime format
    #nowcast_df["date"] = pd.to_datetime(nowcast_df["date"], format="%Y-%m")

    #print(predictors)
    return nowcast_df
    pass


if __name__ == "__main__":
    #file_path = "../Data/test_macro_data.csv"
    file_path = "../Data/lasso_indicators.csv"
    df = pd.read_csv(file_path)

    print(model_bridge(df))
    
    #print("\nFinal Nowcasted GDP:", nowcast_gdp)





###### everything below is just an archive

"""
    # Step 2: Rolling Window Bridge Regression (80 quarters)
    rolling_window = 80  # Set rolling window size

    def rolling_regression(y, X, window):
        predictions = []
        for i in range(window, len(y)):
            X_train = X.iloc[i - window:i]
            y_train = y.iloc[i - window:i]
            model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
            y_pred = model.predict(sm.add_constant(X.iloc[i]))
            predictions.append(y_pred.values[0])
        return predictions

    # Estimate GDP using Bridge Regression Model
    gdp_predictions = rolling_regression(y, X, rolling_window)

"""
    

