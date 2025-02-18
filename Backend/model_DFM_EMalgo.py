# dont use this for now, experimenting

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from scipy.stats import multivariate_normal

file_path = "../Data/test_macro_data.csv"
df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").asfreq("MS")  #ensure time series has a frequency

df = df.sort_index()

# store gdp separate
target_variable = "GDP"
df_gdp = df[[target_variable]].copy()

indicators = [col for col in df.columns if col not in [target_variable]]

df_filtered = df[indicators]

# ---------------------- EM Algorithm to impute missing Data ----------------------
def em_impute_missing_data(df, max_iter=100, tol=1e-4):
    df_imputed = df.copy()

    #fill initial missing values with column means
    for col in df.columns:
        df_imputed[col] = df_imputed[col].fillna(df[col].mean())

    prev_log_likelihood = -np.inf
    for iteration in range(max_iter):
        mean_vector = df_imputed.mean()
        cov_matrix = df_imputed.cov()

        for idx in range(len(df)):
            if df.iloc[idx].isnull().any():
                observed_idx = df.columns[df.iloc[idx].notnull()]
                missing_idx = df.columns[df.iloc[idx].isnull()]

                observed_values = df_imputed.loc[df.index[idx], observed_idx]
                mean_obs = mean_vector.loc[observed_idx]
                mean_mis = mean_vector.loc[missing_idx]

                cov_oo = cov_matrix.loc[observed_idx, observed_idx]
                cov_om = cov_matrix.loc[observed_idx, missing_idx]
                cov_mo = cov_matrix.loc[missing_idx, observed_idx]
                cov_mm = cov_matrix.loc[missing_idx, missing_idx]

                # Regularization to prevent singular matrix
                cov_oo += np.eye(cov_oo.shape[0]) * 1e-4  

                # Ensure covariance matrix is positive definite
                if np.all(np.linalg.eigvals(cov_oo) > 0):
                    inv_cov_oo = np.linalg.pinv(cov_oo.to_numpy())
                    conditional_mean = mean_mis + cov_mo.to_numpy() @ inv_cov_oo @ (observed_values.to_numpy() - mean_obs.to_numpy())
                    df_imputed.loc[df.index[idx], missing_idx] = conditional_mean
                else:
                    print(f"Warning: Non-positive definite covariance matrix at iteration {iteration}.")
                    continue

        new_mean_vector = df_imputed.mean()
        new_cov_matrix = df_imputed.cov()

        try:
            log_likelihood = np.sum(multivariate_normal.logpdf(df_imputed.dropna(), mean=new_mean_vector, cov=new_cov_matrix))
        except ValueError:
            print("Warning: Invalid covariance matrix, skipping log-likelihood computation.")
            break

        if np.abs(log_likelihood - prev_log_likelihood) < tol:
            print(f"EM Algorithm Converged in {iteration} iterations.")
            break
        prev_log_likelihood = log_likelihood

    return df_imputed

# EM to impute missing data
df_filtered_imputed = em_impute_missing_data(df_filtered)

# Normalize 
df_standardized = (df_filtered_imputed - df_filtered_imputed.mean()) / df_filtered_imputed.std()

# ---------------------- Dynamic Factor Model (DFM) ----------------------
k_factors = min(len(df_standardized.columns) // 2, 3)

dfm = sm.tsa.DynamicFactor(df_standardized, k_factors=k_factors, factor_order=1)
dfm_result = dfm.fit(maxiter=50000, method="nm")  

if not dfm_result.mle_retvals.get("converged", False):
    print("WARNING: The model did not fully converge!")

df_gdp["Factor"] = dfm_result.smoothed_state[0]

# ---------------------- Train VAR Model ----------------------
df_model = df_gdp.copy()
df_model["GDP_diff"] = df_model["GDP"].diff()
df_model["Factor_diff"] = df_model["Factor"].diff()
df_model = df_model.dropna()

var_data = df_model[["GDP_diff", "Factor_diff"]]
max_lags = min(6, len(var_data) // 5)

model = VAR(var_data)
lag_order = model.select_order(maxlags=max_lags)
var_result = model.fit(lag_order.aic)

forecast = var_result.forecast(var_data.iloc[-lag_order.aic:].values, steps=1)
predicted_gdp = df_model["GDP"].iloc[-1] + forecast[0, 0]

print(f"\nNowcasted GDP: {predicted_gdp}")
