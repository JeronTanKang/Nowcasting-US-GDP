import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg

def predict_missing_values(df, target_variable="GDP"):
    """ 
    Handles missing values only for predictor variables (not GDP).
    - Starts 3 months before the current month.
    - If a month has missing data, predict using AR(p).
    - If a month has data, use it to predict the next month.
    - Works column by column, excluding GDP (Y variable).
    """

    today = datetime.today().replace(day=1)  # Get first day of the current month
    end_date = today  # The current month is the last month we predict
    start_date = today - pd.DateOffset(months=3)  # Start 3 months before the current month

    # Exclude GDP (target variable) from the predictor list
    predictors = df.columns[df.columns != target_variable]

    for col in predictors:
        try:
            current_date = start_date

            # Train AR model on available data (without dropping NaNs)
            best_p = 1
            best_aic = float('inf')

            for p in range(1, min(6, len(df[col].dropna()))):  # Ensure enough data for lags
                try:
                    model = AutoReg(df[col], lags=p, missing="drop").fit()  # Drop missing values **only for fitting**
                    if model.aic < best_aic:
                        best_p = p
                        best_aic = model.aic
                except:
                    continue

            # Fit the final model using the best p
            final_model = AutoReg(df[col], lags=best_p, missing="drop").fit()

            # Start filling missing values from `start_date` to `end_date`
            while current_date <= end_date:
                if pd.isna(df.loc[current_date, col]):  # If the value is missing, predict it
                    predicted_value = final_model.predict(start=len(df[col].dropna()), end=len(df[col].dropna()))[0]
                    df.loc[current_date, col] = predicted_value
                # Move to next month
                current_date += pd.DateOffset(months=1)

        except Exception as e:
            print(f"Warning: AR model failed for column {col}, error: {e}")

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
print(fit_ols_model(aggregate_indicators(pd.read_csv(file_path))))

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

    # Step 3: Forecast Monthly Indicators Using AR(p) Models
    forecast_horizon = 3  # Predict next 3 months

    indicator_forecasts = {}
    for col in indicators_data.columns:
        series = indicators_data[col].dropna()
        
        # Determine optimal lag order (p) using AIC
        best_p = 1
        best_aic = float('inf')
        for p in range(1, 6):  # Test lags from 1 to 5
            try:
                model = AutoReg(series, lags=p).fit()
                if model.aic < best_aic:
                    best_p = p
                    best_aic = model.aic
            except:
                continue
        
        # Fit AR(p) model with optimal lag order
        final_model = AutoReg(series, lags=best_p).fit()
        forecast_values = final_model.predict(start=len(series), end=len(series) + forecast_horizon - 1)
        
        # Store forecasted values
        indicator_forecasts[col] = forecast_values

    # Convert forecasted indicators into a DataFrame
    forecast_df = pd.DataFrame(indicator_forecasts)
    forecast_df.index = pd.date_range(start=indicators_data.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')

    # Step 4: Aggregate Forecasted Indicators to Quarterly
    forecast_quarterly = forecast_df.resample('Q').mean().iloc[0]  # Take first forecasted quarter

    # Step 5: Nowcast GDP Using the Bridge Equation Coefficients
    X_latest = sm.add_constant(forecast_quarterly.values.reshape(1, -1))
    latest_model = sm.OLS(y[-rolling_window:], sm.add_constant(X[-rolling_window:])).fit()
    nowcast_gdp = latest_model.predict(X_latest)
    """
    
    #return nowcast_gdp