import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg

def record_months_to_forecast(df, predictors):
    """
    Identifies months that need forecasting for each predictor.

    Args:
    - df (pd.DataFrame): DataFrame with dates as index and predictors as columns.
    - predictors (list): List of predictor variable names.

    Returns:
    - dict: A dictionary where keys are predictor names and values are lists of months to forecast.
    """

    months_to_forecast = {}  # Dictionary to store missing months for each predictor

    for col in predictors:
        months_to_forecast[col] = []  # Initialize list for each predictor

        # Find the most recent non-NaN value
        last_known_index = df[col].dropna().index[0] if df[col].dropna().size > 0 else None
        #print(col, last_known_index)

        if last_known_index:
            # Convert to a Timestamp
            last_known_date = pd.to_datetime(last_known_index)

            # Start checking from the next month
            next_date = (last_known_date + pd.DateOffset(months=1)).strftime('%Y-%m')
            #print(next_date)

            # Iterate forward until we reach the last date in df
            while next_date in df.index and pd.isna(df.loc[next_date, col]):
                months_to_forecast[col].append(next_date)
                next_date = (pd.to_datetime(next_date) + pd.DateOffset(months=1)).strftime('%Y-%m')

    return months_to_forecast

def forecast_indicators(df, exclude=["date", "GDP"]):
    df = df.set_index("date")  # Sets "date" as the index
    print(df)
    """ 
    Handles missing values only for predictor variables (not GDP).
    - Starts 3 months before the current month.
    - If a month has missing data, predict using AR(p).
    - If a month has data, use it to predict the next month.
    - Works column by column, excluding GDP (Y variable).
    """

    today = datetime.today().replace(day=1).strftime('%Y-%m')

    # Define start and end dates in YYYY-MM string format
    end_date = today  # The current month is the last month we predict
    start_date = (datetime.today().replace(day=1) - pd.DateOffset(months=3)).strftime('%Y-%m')

    print("PRINT DATES:", start_date, end_date)

    # Exclude GDP (target variable) from the predictor list
    predictors = df.columns.difference(exclude)

    months_to_forecast = record_months_to_forecast(df, predictors)
    print(months_to_forecast)

    for col in predictors:
        if col in months_to_forecast and months_to_forecast[col]:  # Check if there are months to forecast
            end_month = min(months_to_forecast[col])  # Earliest missing month (YYYY-MM string)
            start_month = max(months_to_forecast[col])    # Latest missing month (YYYY-MM string)
            print(start_month, end_month)
            # Convert start and end to integer index positions for AutoReg
            indic_data = df[col].dropna()

            # Find the last known data point's index
            #last_known_index = df.index.get_loc(indic_data.index[-1])

            # Determine forecast start and end indices (relative to training data)
            forecast_start = 0#last_known_index + 1
            forecast_end = forecast_start + len(months_to_forecast[col]) - 1


            print(f"{col}:  forecasting from {forecast_start} to {forecast_end}")

            # Fit AutoReg model
            indic_data = indic_data.sort_index(ascending=True)
            final_model = AutoReg(indic_data, lags=3).fit()

            # Predict missing values using AutoReg
            predicted_values = final_model.predict(start=forecast_start, end=forecast_end)

            # Store predictions in DataFrame format
            predicted_series = pd.Series(predicted_values.values, index=pd.to_datetime(months_to_forecast[col]))
            df.update(predicted_series.to_frame(name=col))

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

    # Ensure 'date' is the index
    if df.index.name != 'date':
        df = df.set_index('date')

    # Drop rows where GDP is missing
    df = df.dropna(subset=['GDP'])

    # Forward-fill missing values in indicators
    df = df.fillna(method='ffill')  # Fills NaNs with the last available value

    print(df)

    # Define dependent (Y) and independent variables (X)
    Y = df['GDP']  # GDP is the dependent variable
    X = df.drop(columns=['GDP'])  # Drop GDP, keep indicators

    # Add constant for the intercept term
    X = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(Y, X).fit()

    # Print model summary
    print(model.summary())
    
    return model


def exp_almon_weighted(series, alpha=0.9):
    """
    Applies an Exponential Almon transformation for weighted aggregation.
    - Recent values get higher weights.

    Args:
    - series (pd.Series): Time series data to transform.
    - alpha (float): Decay factor (0 < alpha < 1, closer to 1 gives more weight to recent values).

    Returns:
    - float: Weighted aggregated value.
    """
    weights = np.array([(alpha ** i) for i in range(len(series))][::-1])
    return np.sum(series * weights) / np.sum(weights)

def aggregate_indicators(df):
    """
    Function that takes in df with monthly frequency indicators and GDP.
    - Converts indicators to quarterly frequency using specified aggregation rules.
    - GDP remains unchanged (takes the only available value per quarter).

    Returns:
    - DataFrame with quarterly frequency.
    """

    # Convert 'date' column to datetime format if not already
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m")
    
    # Set 'date' as index
    df.set_index('date', inplace=True)

    # Define aggregation rules for each column
    aggregation_rule = {
        "CPI": "mean",
        "Crude_Oil": "mean",
        "Interest_Rate": "mean",
        "Unemployment": "mean",
        "Trade_Balance": "sum",
        "PCE": "sum",
        "Retail_Sales": "sum",
        "Housing_Starts": "sum",
        "Capacity_Utilization": "mean",
        "Industrial_Production": "mean",
        "Nonfarm_Payrolls": "sum",
        "PPI": "mean",
        "Core_PCE": "exp_almon"  
    }

    # Separate GDP column from indicators
    gdp_data = df[['GDP']].resample('Q').last()  # Takes the last available GDP value per quarter

    # Initialize an empty DataFrame for indicators
    indicators_data = pd.DataFrame()

    # Apply different aggregation methods for each indicator
    for col, method in aggregation_rule.items():
        if method == "mean":
            indicators_data[col] = df[col].resample('Q').mean()  # Standard mean
        elif method == "sum":
            indicators_data[col] = df[col].resample('Q').sum()  # Summation for flow variables
        elif method == "exp_almon":
            indicators_data[col] = df[col].resample('Q').apply(exp_almon_weighted)  # Apply Almon weighting

    # Merge back GDP and aggregated indicators
    quarterly_df = gdp_data.merge(indicators_data, left_index=True, right_index=True, how='left')

    return quarterly_df


file_path = "../Data/test_macro_data.csv"
print(forecast_indicators(pd.read_csv(file_path)))

def model_bridge(file_path):

    # note: how should gdp and indicators enter this fn? as a single df?

    # indicators in model_bridge will always be monthly frequency. 
    # they will only be converted to quarterly frequency when fitting the model to estimate coefficients, and when predicting nowcast

    df = pd.read_csv(file_path)
    print(df.head())

    # step 1: fit ols on quarterly data
    ols_model = fit_ols_model(aggregate_indicators(df))

    # step 2: for loop to forecast values for each indicator
    monthly_indicators_forecasted = forecast_indicators(df)

    # combine

    # step 3: generate nowcast
    quarterly_indicators_forecasted = aggregate_indicators(monthly_indicators_forecasted) # aggregate to quartlerly
    nowcast_gdp = ols_model.predict(quarterly_indicators_forecasted)

    return nowcast_gdp


if __name__ == "__main__":
    file_path = "../Data/test_macro_data.csv"
    #nowcast_gdp = model_bridge(file_path)
    
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
    

