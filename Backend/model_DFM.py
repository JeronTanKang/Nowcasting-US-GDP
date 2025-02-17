import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "../Data/test_macro_data.csv"  
df = pd.read_csv(file_path)

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Store GDP separately for evaluation later
target_variable = "GDP"
df_gdp = df.set_index("date")[[target_variable]]  # Keep GDP separately

# Dynamically select all indicators except GDP
indicators = [col for col in df.columns if col not in ["date", target_variable]]
print(f"Indicators used for DFM (excluding GDP): {indicators}")

# Filter dataset to include only the selected indicators
df_filtered = df.set_index("date")[indicators]

# ---------------------- Function to Make Data Stationary ----------------------
def make_stationary(df, max_diff=2):
    """
    Ensures all columns in df are stationary using Augmented Dickey-Fuller (ADF) test.
    If a column is non-stationary, applies differencing until stationarity is achieved.
    
    Args:
        df (pd.DataFrame): Input DataFrame with time series data.
        max_diff (int): Maximum number of times to difference a column.

    Returns:
        pd.DataFrame: DataFrame with stationary time series.
    """
    df_stationary = df.copy()
    
    for col in df_stationary.columns:
        diff_count = 0  # Track number of differences applied

        while diff_count < max_diff:
            adf_test = adfuller(df_stationary[col].dropna(), autolag="AIC")
            p_value = adf_test[1]

            if p_value > 0.05:  # Non-stationary
                df_stationary[col] = df_stationary[col].diff().dropna()
                diff_count += 1
                print(f"{col} was differenced {diff_count} time(s) to become stationary.")
            else:
                print(f"{col} is already stationary.")
                break  # Stop differencing if already stationary
    
    return df_stationary.dropna()  # Ensure no NaNs remain

# Make all indicators stationary before running DFM
df_filtered = make_stationary(df_filtered)

# ---------------------- Fit Dynamic Factor Model (DFM) ----------------------
# Reduce factor complexity and increase max iterations
dfm = sm.tsa.DynamicFactor(df_filtered, k_factors=2, factor_order=1, enforce_stationarity=True)
dfm_result = dfm.fit(maxiter=50000, method="lbfgs")  # Increase max iterations

# Check if model converged
if not dfm_result.mle_retvals.get("converged", False):
    print("WARNING: The model did not fully converge!")

# ---------------------- Ensure Proper Index Alignment ----------------------
# Ensure GDP index matches the filtered dataset
df_gdp = df_gdp.loc[df_filtered.index]  # Align df_gdp with df_filtered dates

# Ensure extracted factor length matches df_gdp before assignment
if len(df_gdp) == len(dfm_result.smoothed_state[0]):
    df_gdp["Factor"] = dfm_result.smoothed_state[0]
else:
    print(f"WARNING: Mismatch in factor length ({len(dfm_result.smoothed_state[0])}) vs GDP rows ({len(df_gdp)})")

# Ensure the model has converged
convergence_info = dfm_result.mle_retvals
if not convergence_info.get("converged", False):
    print("WARNING: The model did not fully converge! Consider increasing maxiter.")

# Extract the estimated common factor using Kalman smoothed states
df_gdp["Factor"] = dfm_result.smoothed_state[0]  # Use smoothed state instead of filtered_state

# Compute and display correlation with GDP
correlation_matrix = df_gdp.corr()
print("\nCorrelation Matrix (Factor vs. GDP):")
print(correlation_matrix)

print("\nFirst few rows of GDP and Extracted Factor:")
print(df_gdp.head(50))

# Ensure GDP and Factors are properly aligned
df_model = df_gdp.copy()
df_model["Factor"] = dfm_result.smoothed_state[0]  # Add extracted factor

# Drop NaNs in case of misalignment
df_model = df_model.dropna().sort_index()

print("First few rows of VAR input data:")
print(df_model.head())

# ---------------------- Function to Test Stationarity ----------------------
def check_stationarity(series, name):
    adf_test = adfuller(series.dropna(), autolag="AIC")
    print(f"ADF Test for {name}: p-value = {adf_test[1]}")
    if adf_test[1] > 0.05:
        print(f"{name} is NOT stationary. Differencing is needed.\n")
    else:
        print(f"{name} is stationary.\n")

# Check stationarity for GDP and Factor
check_stationarity(df_model["GDP"], "GDP")
check_stationarity(df_model["Factor"], "Factor")

# Differencing if needed
df_model["GDP_diff"] = df_model["GDP"].diff()
df_model["Factor_diff"] = df_model["Factor"].diff()

# Drop NA values introduced by differencing
df_model = df_model.dropna()

# Check stationarity again
check_stationarity(df_model["GDP_diff"], "Differenced GDP")
check_stationarity(df_model["Factor_diff"], "Differenced Factor")

# ---------------------- Train VAR Model ----------------------
var_data = df_model[["GDP_diff", "Factor_diff"]]

# Fit VAR model with optimal lag selection
model = VAR(var_data)
lag_order = model.select_order(maxlags=6)  # Select best lag using AIC/BIC
print("Optimal Lag Order:", lag_order.selected_orders)

# Fit the VAR model
var_result = model.fit(lag_order.aic)  # Using AIC-selected lag
print(var_result.summary())

# Nowcasting GDP using last available data
lagged_data = var_data.iloc[-lag_order.aic:]

# Forecast next period
forecast = var_result.forecast(lagged_data.values, steps=1)

# Convert back from differenced values
predicted_gdp = df_model["GDP"].iloc[-1] + forecast[0, 0]

#print(f"\nNowcasted GDP: {predicted_gdp}")

# ---------------------- Compute Forecast RMSE ----------------------
def compute_forecast_rmse(data, window_size=20, forecast_horizon=1):
    """
    Computes the rolling RMSE of a VAR model using a sliding window approach.
    """
    actual_values = []
    predicted_values = []
    forecast_dates = []
    residual = []

    # Ensure we only use rows where GDP is not missing
    data = data.dropna(subset=["GDP"])  # Remove missing GDP rows

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
        #last_gdp = train["GDP"].iloc[-1]
        last_gdp=0

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

    # Compute RMSE
    results_df = pd.DataFrame({"Forecast Date": forecast_dates, "Actual GDP": actual_values, "Predicted GDP": predicted_values, "Residuals": residual})
    print("\nForecast vs. Actual GDP:")
    print(results_df)

    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    print(f"\nRolling RMSE: {rmse:.4f}")

    return rmse

# Compute RMSE for forecasts
rmse = compute_forecast_rmse(df_model, window_size=20, forecast_horizon=1)