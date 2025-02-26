import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd
import statsmodels.api as sm

import pandas as pd
import statsmodels.api as sm

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


def aggregate_indicators(df):
    """
    Function that takes in df with monthly frequency indicators and GDP.
    - Converts indicators to quarterly frequency using mean aggregation.
    - GDP remains unchanged (takes the only available value per quarter).

    Returns:
    - DataFrame with quarterly frequency.
    """

    # Convert 'date' column to datetime format if not already
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m")

    # Set 'date' as index
    df.set_index('date', inplace=True)

    # Separate GDP column from indicators
    gdp_data = df[['GDP']].resample('Q').last()  # Takes the last available GDP value per quarter
    indicators_data = df.drop(columns=['GDP']).resample('Q').mean()  # Aggregate indicators using mean

    # Merge back GDP and aggregated indicators
    quarterly_df = gdp_data.merge(indicators_data, left_index=True, right_index=True, how='left')

    return quarterly_df

# uncomment below to test

#file_path = "../Data/test_macro_data.csv"
#print(fit_ols_model(aggregate_indicators(pd.read_csv(file_path))))

def model_bridge(file_path):

    # note: how should gdp and indicators enter this fn? as a single df?

    # indicators in model_bridge will always be monthly frequency. 
    # they will only be converted to quarterly frequency when fitting the model to estimate coefficients, and when predicting nowcast

    df = pd.read_csv(file_path)
    print(df.head())

    # step 1: fit ols on quarterly data
    ols_model = fit_ols_model(aggregate_indicators(df))

    # step 2: for loop to forecast values for each indicator
    forecasted_indicators = forecast_indicators(df)

    # combine

    # step 3: generate nowcast
    quarterly_indicators_forecasted = aggregate_indicators(forecasted_indicators) # aggregate to quartlerly
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
    gdp_predictions = rolling_regression(y, X, rolling_window)"""

    """

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