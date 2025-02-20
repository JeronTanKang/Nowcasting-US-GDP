import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "../Data/test_macro_data.csv" 
data = pd.read_csv(file_path, parse_dates=["date"])
data["date"] = pd.to_datetime(data["date"], format="%Y-%m")

#extract quarterly gdp values
gdp_data = data[["date", "GDP"]].dropna() 
gdp_data["quarter"] = gdp_data["date"].dt.to_period("Q")


#change indicators from monthly to quarterly by taking mean
monthly_data = data.drop(columns = ["GDP"])
monthly_data["quarter"] = monthly_data["date"].dt.to_period("Q")
quarterly_indicators = monthly_data.groupby("quarter").mean().reset_index()

#merge with GDP values
merged_data = gdp_data.merge(quarterly_indicators, on="quarter", how="inner")

# Define features (X) and target (y)
X = merged_data.drop(columns=["GDP", "date", "quarter"], errors='ignore')  # Remove unnecessary columns
y = merged_data["GDP"]


# Standardize features (Lasso is sensitive to scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run Lasso with cross-validation
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)


# Extract important indicators (non-zero coefficients)
selected_features = X.columns[lasso.coef_ != 0]

print("Selected indicators:", selected_features.tolist())

# Store results in a dataframe
lasso_results = pd.DataFrame({"Feature": X.columns, "Coefficient": lasso.coef_})
lasso_results = lasso_results[lasso_results["Coefficient"] != 0]  # Keep only important indicators
print(lasso_results)

lasso_results["Importance"] = abs(lasso_results["Coefficient"])  # Compute absolute importance
lasso_results = lasso_results.sort_values(by="Importance", ascending=False)

print(lasso_results)
