import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=Warning)

def model_bridge(file_path):

    # note: how should gdp and indicators enter this fn? as a single df?

    # indicators in model_bridge will always be monthly frequency. 
    # they will only be converted to quarterly frequency when fitting the model to estimate coefficients, and when predicting nowcast

    df = pd.read_csv(file_path)

    gdp_data = pd.read_csv(file_path, parse_dates=['date']) # quarterly
    indicators_data = pd.read_csv('indicators_data.csv', parse_dates=['date']) # monthly

    gdp_data.set_index('date', inplace=True)
    indicators_data.set_index('date', inplace=True)

    # Step 1: Aggregate Monthly Indicators to Quarterly Frequency
    quarterly_indicators = indicators_data.resample('Q').mean()

    # Merge GDP with Quarterly Aggregated Indicators
    data = pd.merge(gdp_data, quarterly_indicators, left_index=True, right_index=True, how='inner')

    # Define dependent and independent variables
    y = data['gdp']
    X = data.drop(columns=['gdp'])

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
    pass
    #return nowcast_gdp

print(f'Nowcasted GDP for the next quarter: {nowcast_gdp[0]:.2f}')

if __name__ == "__main__":
    file_path = "../Data/test_macro_data.csv"
    nowcast_gdp = model_bridge(file_path)
    
    print("\nFinal Nowcasted GDP:", next_gdp)
