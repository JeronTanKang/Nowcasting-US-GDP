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
                    predicted_value = final_model.predict(start=len(data_before_forecast), end=len(data_before_forecast))[0]

                    # Update the DataFrame with the predicted value at the missing date
                    df.loc[forecast_date, col] = predicted_value

                    # Optionally, print the predicted value for debugging
                    #print(f"Predicted {col} for {forecast_date}: {predicted_value}")

                except Exception as e:
                    print(f"Could not forecast {col} for {forecast_date} due to: {e}")

    df = df.reset_index()
    df = df.sort_values(by="date").reset_index(drop=True)
    print("Returned by forecast_indicators:", df)
    return df




def fit_ols_model(df, drop_variables):
    """
    Fits an OLS regression model using GDP as the dependent variable
    and the other 10 indicators as independent variables.

    Args:
    - df (pd.DataFrame): DataFrame with 'date' as index, 'GDP' column, and 10 indicators.

    Returns:
    - results: Fitted OLS model
    """
    print("XXXXXX", df)
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

    print(model.summary())
    
    return model


def model_bridge(df):

    """
    indicators in model_bridge will always be monthly frequency. 
    they will only be converted to quarterly frequency when fitting the model to estimate coefficients, and when predicting nowcast (using aggregate_indicators)
    """
    df["date"] = pd.to_datetime(df["date"])
    print("ORIGINAL DF", df)

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
    drop_variables = ["GDP","gdp_growth",#"gdp_growth_lag1",
                    "gdp_growth_lag2", "gdp_growth_lag3", "gdp_growth_lag4",
                    #"junk_bond_spread",
                    #"junk_bond_spread_lag1",
                    "junk_bond_spread_lag2",
                    "junk_bond_spread_lag3", "junk_bond_spread_lag4",  
                    "Industrial_Production_lag3",
                    "Interest_Rate_lag1",
                    #"dummy",
                    "Construction_Spending"
                     ]
    ##########################
    ols_model = fit_ols_model(train_ols, drop_variables)
    
    # step 2: for loop to forecast values for each indicator
    # OK DONE FOR STEP 2
    monthly_indicators_forecasted = forecast_indicators(df)
    print("monthly_indicators_forecasted",monthly_indicators_forecasted)

    # step 3: generate nowcast
    #monthly_indicators_forecasted.index = pd.to_datetime(monthly_indicators_forecasted.index)
    quarterly_indicators_forecasted = aggregate_indicators(monthly_indicators_forecasted) # aggregate to quartlerly


    drop_variables.append("date")
    predictors = quarterly_indicators_forecasted.drop(columns=drop_variables, errors='ignore')
    #predictors = sm.add_constant(predictors, has_constant='add')  

    # where mask is the dates to forecast
    mask = quarterly_indicators_forecasted["GDP"].isna()
    predictors_to_forecast = predictors[mask]
    dates_to_forecast = quarterly_indicators_forecasted["date"][mask]

    nowcast_growth = ols_model.predict(predictors_to_forecast)
    #print("OUTCOMES (gdp_growth forecast):", nowcast_growth)

    # Iterated forecasting for GDP growth and GDP
    nowcast_growth = []
    nowcasted_gdp_levels = []
    
    # Initialize the last known GDP
    last_actual_gdp = train_ols["GDP"].dropna().iloc[-1]  # Last known actual GDP

    # Initialize a dictionary to store predicted growth values (gdp_growth_lag1)
    predicted_growth_dict = {}

    predictors_to_forecast = predictors_to_forecast.reset_index(drop=True)
        
    print("predictors_to_forecast", predictors_to_forecast)

    # Loop over predictors to forecast iteratively
    for idx, row in predictors_to_forecast.iterrows():
        # Step 1: Update the current row's gdp_growth_lag1 with the predicted value from the dictionary
        if idx > 0:  # Skip the first iteration
            row["gdp_growth_lag1"] = predicted_growth_dict[idx - 1]  # Use the previous row's predicted growth

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

        # Step 5: Store the predicted growth (gdp_growth_lag1) in the dictionary for the next row
        predicted_growth_dict[idx] = predicted_growth

    # Final output DataFrame with Nowcasted GDP Growth and GDP levels
    nowcast_df = pd.DataFrame({
        "date": pd.to_datetime(dates_to_forecast),
        "Nowcasted_GDP_Growth": nowcast_growth,
        "Nowcasted_GDP": nowcasted_gdp_levels
    })

    return nowcast_df


if __name__ == "__main__":
    file_path = "../Data/bridge_df.csv"
    #file_path = "../Data/manual_testing.csv"
    df = pd.read_csv(file_path)

    print(model_bridge(df))







