
import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Set figure size for plots
plt.rcParams["figure.figsize"] = [15, 10]

# Load dataset
data = pd.read_csv("../Data/test_macro_data.csv", parse_dates=["date"])

# Sort data and set date as index
data = data.sort_values(by="date")
data.set_index("date", inplace=True)

# Define target variable (GDP)
target_variable = "GDP"

# Filter only rows where GDP is available (NO INTERPOLATION)
gdp_data = data.dropna(subset=[target_variable])

print(gdp_data)

# Train ARMA model using available GDP values
train_series = gdp_data[target_variable]

print("train_series:", train_series)

# Auto-select ARMA order
model = pm.auto_arima(train_series, seasonal=False, stationary=True)
ar_order, ma_order = model.order[0], model.order[2]  # Extract AR and MA orders

# Fit ARMA model
arma_model = ARIMA(train_series, order=(ar_order, 0, ma_order)).fit()

# Generate GDP forecasts for the next 4 quarters
future_steps = 4
forecast = arma_model.forecast(steps=future_steps)

# Print model summary & forecasted values
print(arma_model.summary())
print("Next 4 quarters' predicted GDP values:")
print(forecast)

# Plot actual vs predicted GDP
plt.figure(figsize=(12, 6))
plt.plot(gdp_data.index, gdp_data[target_variable], label="Actual GDP", marker="o")
plt.plot(pd.date_range(gdp_data.index[-1], periods=future_steps+1, freq="Q")[1:], forecast, 
         label="Predicted GDP", marker="x", linestyle="dashed")

plt.xlabel("Year")
plt.ylabel("GDP")
plt.title("ARMA Model - GDP Forecast")
plt.legend()
plt.grid()
plt.show()
