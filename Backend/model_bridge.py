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
    print("TODAY MONTH:", today_dt_object.month)

    month_offset = (today_dt_object.month - 1) % 3  # Cycles every 3 months
    months_to_add = [today_dt_object + pd.DateOffset(months=i+1) for i in range(2 - month_offset)]
    
    # Convert to YYYY-MM format and filter out months already in the index
    months_to_add = [date.strftime('%Y-%m') for date in months_to_add if date.strftime('%Y-%m') not in df.index]
    
    if months_to_add:
        new_rows = pd.DataFrame(index=months_to_add, columns=df.columns)
        df = pd.concat([df, new_rows]).sort_index(ascending=False)


    print(f" Added rows: {months_to_add}")

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

            indic_data = df[col].dropna()

            # Find the last known data point's index

            # Determine forecast start and end indices (relative to training data)
            forecast_start = 0
            forecast_end = forecast_start + len(months_to_forecast[col]) - 1


            #print(f"{col}:  forecasting from {forecast_start} to {forecast_end}")

            # Fit AutoReg model
            indic_data = indic_data.sort_index(ascending=True)
            final_model = AutoReg(indic_data, lags=5).fit()

            # Predict missing values using AutoReg
            predicted_values = final_model.predict(start=forecast_start, end=forecast_end)
            #print("predicted_values:", predicted_values)

            # Store predictions in DataFrame format
            predicted_series = pd.Series(predicted_values.values, index=pd.to_datetime(months_to_forecast[col]))
            df.update(predicted_series.to_frame(name=col))


    #df_temp = df_temp.reindex(df.index)  
    #df_temp.update(df) 
    df = df.reset_index()  # Moves the index to a column
    df.rename(columns={"index": "date"}, inplace=True)  # Renames the new column to "date"
    df = df.set_index('date')
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
    # Drop rows where GDP is missing
    df = df.dropna(subset=['GDP'])


    # Forward-fill missing values in indicators
    df = df.fillna(method='ffill')  # Fills NaNs with the last available value

    #print(df)

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
    
    #df['date'] = pd.to_datetime(df['date'], format="%Y-%m")
    #df = df.set_index('date')


    # Define aggregation rules for each column
    aggregation_rule = {
        "Industrial_Production": "mean",  # Average industrial output over the period
        "Retail_Sales": "sum",  # Total sales should be summed
        "Nonfarm_Payrolls": "sum",  # Employment-related numbers are usually summed
        "Trade_Balance": "sum",  # Trade surplus/deficit is accumulated over the period
        "Core_PCE": "exp_almon",  # Applying an exponential Almon lag model for smoothing
        "Unemployment": "mean",  # Unemployment rate is an average
        "Interest_Rate": "mean",  # Interest rates are typically averaged
        "Three_Month_Treasury_Yield": "mean",  # Treasury yields are averaged
        "Construction_Spending": "sum",  # Total spending should be summed
        "Housing_Starts": "sum",  # Count data should be summed
        "Capacity_Utilization": "mean",  # Utilization rate is an average
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
    quarterly_df = quarterly_df.reset_index()
    quarterly_df['date'] = quarterly_df['date'].dt.strftime('%Y-%m')

    return quarterly_df


def model_bridge(df):

    # note: how should gdp and indicators enter this fn? as a single df?

    # indicators in model_bridge will always be monthly frequency. 
    # they will only be converted to quarterly frequency when fitting the model to estimate coefficients, and when predicting nowcast

    df_temp = df.copy()

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
    print(predictors.dtypes)
    
    nowcast_gdp = ols_model.predict(predictors)


    return predictors
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
    

