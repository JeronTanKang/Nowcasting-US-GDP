import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import sys
import os
# Add the Backend folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
# Import the functions from make_stationary.py
from data_processing import is_stationary, make_stationary

# Load dataset
file_path = "test_macro_data.csv" 
data = pd.read_csv(file_path, parse_dates=["date"])
data["date"] = pd.to_datetime(data["date"], format="%Y-%m")

data["GDP"] = data["GDP"].shift(1) #shift gdp values down to correct quarter


#drop rows with no gdp values
data = data.dropna(subset=["GDP"]) 

#back fill NANs for indicators
data.fillna(method='bfill', inplace=True)  

data = data.sort_values(by="date", ascending=True) #arrange in descending date order for differencing

#drop date column
data.drop(columns= "date", inplace= True) 


data_stationary, diff_counts = make_stationary(data)
print(data_stationary)
#drop 2 NA rows
data_stationary.dropna(inplace=True) 

# Define features (X) and target (y)
x = data_stationary.drop(columns=["GDP", "date", "quarter", 'date_x', 'date_y'], errors='ignore')  # keep indicator variables only
y = data_stationary["GDP"]

# Standardize
scaler = StandardScaler()
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std  
x_scaled = scaler.fit_transform(x)

# Run Lasso with cross-validation
lasso = LassoCV(cv=5, random_state=42, max_iter = 5000).fit(x_scaled, y_scaled)

selected_features = x.columns[lasso.coef_ != 0]
print("Selected indicators:", selected_features.tolist())

# Store results in a dataframe
lasso_results = pd.DataFrame({"Feature": x.columns, "Coefficient": lasso.coef_})
lasso_results = lasso_results[lasso_results["Coefficient"] != 0]  # Keep only important indicators

lasso_results["Importance"] = abs(lasso_results["Coefficient"])  # Compute absolute importance
lasso_results = lasso_results.sort_values(by="Importance", ascending=False)

print(lasso_results) #just to check out indicators that are non zero

# Get feature importance and sort by absolute values
feature_importance = np.abs(lasso.coef_)
sorted_features = x.columns[np.argsort(-feature_importance)]  # Sort in descending order

### **AIC Feature Selection**
def compute_aic_statsmodels(x_subset, y_subset):
    x_with_const = sm.add_constant(x_subset)  # Add intercept term
    ols_model = sm.OLS(y_subset, x_with_const).fit()
    return ols_model.aic

# Test different feature counts for AIC selection
aic_scores = []
feature_counts = []

for num_features in range(1, len(sorted_features) + 1, 1):  # for loop with increasing number of indicators
    selected_features = sorted_features[:num_features]
    x_subset = x[selected_features]

    # Compute AIC using statsmodels OLS
    aic = compute_aic_statsmodels(x_subset, y)

    aic_scores.append(aic)
    feature_counts.append(num_features)

# Convert results to DataFrame
results = pd.DataFrame({"Feature Count": feature_counts, "AIC": aic_scores})

# Find best feature count based on AIC
best_aic_count = results.loc[results["AIC"].idxmin(), "Feature Count"]

# Select final feature count based on AIC
optimal_feature_count = best_aic_count 
final_selected_features = sorted_features[:optimal_feature_count]

print("Final selected indicators (based on AIC):")
print(final_selected_features.tolist())

