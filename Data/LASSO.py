import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler




# Load dataset
file_path = "../Data/fredmd_all_indicators.csv" 
data = pd.read_csv(file_path, parse_dates=["date"])
data["date"] = pd.to_datetime(data["date"], format="%Y-%m")

data = data.dropna(subset=["GDP"])

data.fillna(method='bfill', inplace=True)

# Define features (X) and target (y)
X = data.drop(columns=["GDP", "date", "quarter", 'date_x', 'date_y'], errors='ignore')  # Remove unnecessary columns
y = data["GDP"]


# Standardize
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
