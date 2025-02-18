# dont use this for now, experimenting

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.structural import UnobservedComponents

import warnings
warnings.filterwarnings("ignore")

file_path = "../Data/test_macro_data.csv"
df = pd.read_csv(file_path)


df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

target_variable = "GDP"
df_gdp = df.set_index("date")[[target_variable]]

indicators = [col for col in df.columns if col not in ["date", target_variable]]
print(f"Indicators used for DFM (excluding GDP): {indicators}")

df_filtered = df.set_index("date")[indicators]


# Apply Kalman Smoothing Before Differencing 
def apply_kalman_smoothing(series):
    """
    Applies Kalman smoothing to handle missing values in a time series using
    the UnobservedComponents model. Ensures that all NaNs are interpolated.
    """
    if series.isna().all():  # If the entire series is NaN, WHY? return as is
        return series

    # Fit the UnobservedComponents model with a local level component
    model = UnobservedComponents(series, level='local level')
    result = model.fit(disp=False)

    # extract smoothed values
    smoothed_values = pd.Series(result.smoothed_state[0], index=series.index)

    # makr sure all NaNs in the original series are filled with the smoothed values
    series_interpolated = series.combine_first(smoothed_values)

    return series_interpolated


# ensure stationarity before dfm
def make_stationary(df):
    """
    Ensures all time series in the dataframe are stationary.
    Applies Kalman smoothing FIRST, then checks stationarity, then applies differencing if necessary.
    """
    stationary_df = df.copy()

    for col in df.columns:
        # kalman smoothing first
        stationary_df[col] = apply_kalman_smoothing(stationary_df[col])

        #test for stationarity
        adf_test = adfuller(stationary_df[col].dropna(), autolag="AIC")
        p_value = adf_test[1]

        # apply differencing if stationary
        if p_value > 0.05:
            print(f"⚠ {col} is still non-stationary after Kalman smoothing (p={p_value:.3f}). Applying differencing.")
            print(stationary_df[col].tail())
            stationary_df[col] = stationary_df[col].diff()

            # apply Kalman smoothing again (to correct NaNs caused by differencing)
            stationary_df[col] = apply_kalman_smoothing(stationary_df[col])

    # temporary fix: Drop rows where more than 60% of columns are NaN
    threshold = int(0.4 * df.shape[1])
    stationary_df = stationary_df.dropna(thresh=threshold)

    # temporary fix: Fill remaining NaNs using forward & backward fill (limited to 1 step)
    stationary_df = stationary_df.ffill(limit=1).bfill(limit=1)

    return stationary_df


df_filtered = make_stationary(df_filtered)

# temp fix: drop initial rows that still contain too many NaNs
df_filtered = df_filtered.iloc[5:].dropna()

# JUST A TEMP FIX HERE Ensure at least 2 indicators remain
if df_filtered.shape[1] < 2:
    print("WARNING: Restoring non-stationary indicators to ensure DFM has at least 2 columns.")
    df_filtered = df.set_index("date")[indicators]  


# ---- Handle Missing Data in Indicators Using Dynamic Factor Model ---- #
dfm_missing = sm.tsa.DynamicFactor(df_filtered, k_factors=3, factor_order=1, enforce_stationarity=False)
dfm_missing_result = dfm_missing.fit()
df_filtered[:] = dfm_missing_result.fittedvalues  # Fill missing indicator values

# Normalize data after differencing
df_standardized = (df_filtered - df_filtered.mean()) / df_filtered.std()

# ---- Build the Dynamic Factor Model (DFM) ---- #
dfm = sm.tsa.DynamicFactor(df_standardized, k_factors=3, factor_order=1, enforce_stationarity=False)
dfm_result = dfm.fit(maxiter=10000, method="lbfgs")

if not dfm_result.mle_retvals.get("converged", False):
    print("⚠️ WARNING: The model did not fully converge! Consider increasing maxiter.")


# ---- Apply Kalman Smoothing to GDP Interpolation ---- #
df_gdp["GDP_interpolated"] = apply_kalman_smoothing(df_gdp["GDP"])
df["GDP"] = df_gdp["GDP_interpolated"]


# ---- Ensure GDP and Factors Are Properly Aligned ---- #
df_model = df_gdp.copy()
df_model["Factor"] = dfm_result.smoothed_state[0]

# Drop NaNs in case of misalignment
df_model = df_model.dropna().sort_index()


# ---- Train VAR Model ---- #
var_data = df_model[["GDP", "Factor"]].diff().dropna()

# Fit VAR model with optimal lag selection
model = VAR(var_data)
lag_order = model.select_order(maxlags=6)
print("Optimal Lag Order:", lag_order.selected_orders)

# Fit the VAR model
var_result = model.fit(lag_order.aic)
print(var_result.summary())


# ---- Nowcasting GDP ---- #
lagged_data = var_data.iloc[-lag_order.aic:]
forecast = var_result.forecast(lagged_data.values, steps=1)

# Convert back from differenced values
predicted_gdp = df_model["GDP"].iloc[-1] + forecast[0, 0]
print(f"\nNowcasted GDP: {predicted_gdp}")


# ---- Compute Forecast RMSE ---- #
def compute_forecast_rmse(data, window_size=20, forecast_horizon=1):
    """
    Computes the rolling RMSE of a VAR model using a sliding window approach.
    """
    actual_values, predicted_values = [], []

    # Sliding window approach
    for start in range(len(data) - window_size - forecast_horizon + 1):
        train = data.iloc[start : start + window_size]
        test = data.iloc[start + window_size : start + window_size + forecast_horizon]

        if test["GDP"].isna().any():
            continue  # Skip iteration if GDP is missing in the test set

        var_model = VAR(train)
        var_result = var_model.fit(maxlags=2)
        forecast = var_result.forecast(train.values[-var_result.k_ar:], steps=forecast_horizon)

        forecasted_gdp = test["GDP"].iloc[0] + forecast[0, 0]
        actual_values.append(test["GDP"].values[forecast_horizon - 1])
        predicted_values.append(forecasted_gdp)

    print(f"\nRolling RMSE: {np.sqrt(mean_squared_error(actual_values, predicted_values)):.4f}")


# Compute RMSE
compute_forecast_rmse(df_model, window_size=20, forecast_horizon=1)
