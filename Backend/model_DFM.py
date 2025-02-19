import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter(action='ignore', category=Warning)

def dfm_nowcast(file_path: str, target_variable: str = "GDP"):
    """
    Reads macroeconomic data, trains a Dynamic Factor Model (DFM),
    and returns the GDP nowcast for the most recent available month.

    Prints RMSE of forecasts within the function.

    Args:
        file_path (str): Path to the macroeconomic dataset.
        target_variable (str): Column name to nowcast (default: "GDP").

    Returns:
        float: GDP nowcast for the next available period (latest date).
    """
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")

    df = df.sort_values(by="date")
    df.set_index("date", inplace=True)

    # TEMPORARY INTERPOLATION: forward fill
    df.fillna(method="ffill", inplace=True)

    df_gdp = df[[target_variable]]

    indicators = [col for col in df.columns if col != target_variable]
    df_indicators = df[indicators]

    df_gdp = df_gdp.loc[df_indicators.index]

    def make_stationary(df, max_diff=2): # ensure max amount of differencing applied is 2
        df_stationary = df.copy()
        for col in df_stationary.columns:
            diff_count = 0
            while diff_count < max_diff:
                adf_test = adfuller(df_stationary[col].dropna(), autolag="AIC")
                if adf_test[1] > 0.05:  # non stationary
                    df_stationary[col] = df_stationary[col].diff().dropna()
                    diff_count += 1
                else:
                    break
        return df_stationary  # Do NOT dropna() to keep time alignment

    df_indicators = make_stationary(df_indicators)


    # BIC for optimal k_factors
    def select_optimal_factors(df, max_factors=5):
        best_k = 1
        best_bic = np.inf  # Start with a high BIC value

        for k in range(1, max_factors + 1):
            try:
                dfm = sm.tsa.DynamicFactor(df, k_factors=k, factor_order=1, enforce_stationarity=True)
                dfm_result = dfm.fit(maxiter=5000, method="lbfgs")
                bic = dfm_result.bic  # Get Bayesian Information Criterion

                print(f"k_factors={k}, BIC={bic:.2f}")

                if bic < best_bic:  
                    best_k = k
                    best_bic = bic
            except Exception as e:
                print(f"DFM failed for k={k}: {e}")

        print(f"Optimal number of factors: {best_k}")
        return best_k

    # only run the line of code below if u want to find optimal_k using BIC
    # optimal_k = select_optimal_factors(df_indicators, max_factors=5) #run this to find optimal_k
    #print("optimal_k", optimal_k)
    
    optimal_k = 1
    
    # fit dfm 
    dfm = sm.tsa.DynamicFactor(df_indicators, k_factors=optimal_k, factor_order=1, enforce_stationarity=True)
    # to do: check if we want to search what optimal factor_order to use
    dfm_result = dfm.fit(maxiter=50000, method="lbfgs")

    if not dfm_result.mle_retvals.get("converged", False):
        print("WARNING: The model did not fully converge!")

    # extract latent factor from dfm
    df_gdp = df_gdp.iloc[-dfm_result.smoothed_state.shape[1]:] 
    df_gdp["Factor"] = dfm_result.smoothed_state[0]

    # train VAR model
    df_model = df_gdp.dropna()
    df_model["GDP_diff"] = df_model["GDP"].diff()
    df_model["Factor_diff"] = df_model["Factor"].diff()
    df_model = df_model.dropna()

    var_data = df_model[["GDP_diff", "Factor_diff"]]
    model = VAR(var_data)
    lag_order = model.select_order(maxlags=6).aic
    var_result = model.fit(lag_order)

    # nowcast for latest available date
    latest_date = df.index.max()  # Get the latest date in the dataset
    print(f"\nForecasting GDP for latest date: {latest_date.strftime('%Y-%m')}")

    # Use the most recent available macro indicators for forecasting
    latest_data = df_indicators.loc[latest_date].values.reshape(1, -1)

    # nowcast
    forecast = var_result.forecast(var_data.iloc[-lag_order:].values, steps=1)
    next_gdp_nowcast = df_model["GDP"].iloc[-1] + forecast[0, 0]

    print(f"\nNowcasted GDP for {latest_date.strftime('%Y-%m')}: {next_gdp_nowcast}")

    def compute_forecast_rmse(data, window_size=20, forecast_horizon=1):
        """
        Computes the rolling RMSE of a VAR model using a sliding window approach.
        """
        actual_values = []
        predicted_values = []
        forecast_dates = []
        residual = []

        # Ensure we only use rows where GDP is not missing (depends on if we decide to interpolate GDP data)
        data = data.dropna(subset=["GDP"])  

        # Sliding window approach
        for start in range(int((len(data) - window_size - forecast_horizon + 1) * 0.95), len(data) - window_size - forecast_horizon + 1):
            train = data.iloc[start : start + window_size]  # Train on this window
            test = data.iloc[start + window_size : start + window_size + forecast_horizon]  # Next period to predict

            # Ensure test period has valid GDP data
            if test["GDP"].isna().any():
                continue  # Skip iteration if GDP is missing in the test set

            # Fit VAR model
            var_model = VAR(train)
            var_result = var_model.fit(maxlags=2)  # Using lag 2 based on previous selection

            # Get the last known GDP before forecast
            last_gdp = 0#train["GDP"].iloc[-1] 

            # Forecast the next GDP_diff
            lagged_data = train.values[-var_result.k_ar :]
            forecast = var_result.forecast(lagged_data, steps=forecast_horizon)

            # Convert differenced prediction back to original scale
            forecasted_gdp = last_gdp + forecast[0, 0]

            # Store actual and predicted values
            actual_values.append(test["GDP"].values[forecast_horizon - 1])
            predicted_values.append(forecasted_gdp)
            residual.append(forecasted_gdp - test["GDP"].values[forecast_horizon - 1])
            forecast_dates.append(test.index[forecast_horizon - 1])

        results_df = pd.DataFrame({"Forecast Date": forecast_dates, "Actual GDP": actual_values, "Predicted GDP": predicted_values, "Residuals": residual})
        print("\nForecast vs. Actual GDP:")
        print(results_df)

        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        print(f"\nRolling RMSE: {rmse:.4f}")

        return rmse

    compute_forecast_rmse(df_model)

    return next_gdp_nowcast

# Example usage
if __name__ == "__main__":
    file_path = "../Data/test_macro_data.csv"
    next_gdp = dfm_nowcast(file_path)
    
    print("\nFinal Nowcasted GDP:", next_gdp)
