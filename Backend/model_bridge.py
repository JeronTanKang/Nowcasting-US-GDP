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

"""#pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
"""
# Reset after printing
"""pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
pd.reset_option("display.width")
pd.reset_option("display.max_colwidth")"""

def record_months_to_forecast(df, predictors):
    """
    Identifies months that need forecasting for each predictor.

    Args:
    - df (pd.DataFrame): DataFrame with dates as index and predictors as columns.
    - predictors (list): List of predictor variable names.

    Returns:
    - dict: A dictionary where keys are predictor names and values are lists of months to forecast.
    """
    
    # TRY THIS
    print("input to record months:", df)
    #df = df.set_index('date')

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

    print("input to record months:", df)

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

def forecast_indicators(df, exclude=["date","GDP","gdp_growth","gdp_growth_lag1","gdp_growth_lag2","gdp_growth_lag3","gdp_growth_lag4"]):
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
    print("Months to forecast:", months_to_forecast)

    for col in predictors:
        if col in months_to_forecast and months_to_forecast[col]:
            indic_data = df[col].dropna().sort_index()

            forecast_start = 0
            forecast_end = forecast_start + len(months_to_forecast[col]) - 1
            print(f"{col}: forecasting from {forecast_start} to {forecast_end}")

            try:
                final_model = AutoReg(indic_data, lags=9, old_names=False).fit()
                predicted_values = final_model.predict(start=forecast_start, end=forecast_end)

                predicted_series = pd.Series(predicted_values.values, index=pd.to_datetime(months_to_forecast[col]))
                df.update(predicted_series.to_frame(name=col))
            except Exception as e:
                print(f"Could not forecast {col} due to: {e}")

    df = df.reset_index()
    df = df.sort_values(by="date").reset_index(drop=True)
    print("Returned by forecast_indicators:", df)
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
    df = df.iloc[::-1].reset_index(drop=True) # reverse row order

    # do i need this safety check?
    #df = df.dropna(subset=['GDP'])


    # Forward-fill missing values in indicators (just in case)
    df = df.fillna(method='ffill') 

    #print(df)

    Y = df['gdp_growth']  
    X = df.drop(columns=["GDP","gdp_growth", "gdp_growth_lag1", "gdp_growth_lag2"]) 

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
    df["date"] = pd.to_datetime(df["date"])
    print("ORIGINAL DF", df)

    train_df = df[df["gdp_growth"].notna()].copy()
    forecast_df = df[df["gdp_growth"].isna()].copy()

    #print("PPPPPPEEEEEEEEKKKKK",df_aggregated)

    # step 1 aggregate
    # only want to pass into fit_ols_model the train data
    df_aggregated = aggregate_indicators(df)
    train_ols = df_aggregated[df_aggregated["gdp_growth"].notna()].copy()

    # step 1: fit ols on quarterly data

    #print("PPPPPPEEEEEEEEKKKKK",train_ols)
    ols_model = fit_ols_model(train_ols)
    
    # step 2: for loop to forecast values for each indicator
    # OK DONE FOR STEP 2
    monthly_indicators_forecasted = forecast_indicators(df)
    print("monthly_indicators_forecasted",monthly_indicators_forecasted)

    # step 3: generate nowcast
    #monthly_indicators_forecasted.index = pd.to_datetime(monthly_indicators_forecasted.index)
    quarterly_indicators_forecasted = aggregate_indicators(monthly_indicators_forecasted) # aggregate to quartlerly

    predictors = quarterly_indicators_forecasted.drop(columns=["date","GDP","gdp_growth", "gdp_growth_lag1", "gdp_growth_lag2"], errors='ignore')
    predictors = sm.add_constant(predictors, has_constant='add')  

    # where mask is the dates to forecast
    mask = quarterly_indicators_forecasted["gdp_growth"].isna()
    predictors_to_forecast = predictors[mask]
    dates_to_forecast = quarterly_indicators_forecasted["date"][mask]

    nowcast_growth = ols_model.predict(predictors_to_forecast)
    #print("OUTCOMES (gdp_growth forecast):", nowcast_growth)

    # Get last known actual GDP level from training data
    last_actual_gdp = train_ols["GDP"].dropna().iloc[-1]

    # Convert nowcasted gdp_growth to GDP level
    nowcasted_gdp_levels = []
    for growth in nowcast_growth:
        next_gdp = last_actual_gdp * np.exp(growth / 400)
        nowcasted_gdp_levels.append(next_gdp)
        last_actual_gdp = next_gdp  # update for next step

    # Final output DataFrame
    nowcast_df = pd.DataFrame({
        "date": pd.to_datetime(dates_to_forecast),
        "Nowcasted_GDP_Growth": nowcast_growth.values,
        "Nowcasted_GDP": nowcasted_gdp_levels
    })

    return nowcast_df


if __name__ == "__main__":
    file_path = "../Data/bridge_df.csv"
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
    

