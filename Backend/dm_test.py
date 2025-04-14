"""
This file contains functions for conducting statistical tests to compare the forecast performance of different models using the Diebold-Mariano (DM) test. 
The `apply_df_gls_test` function applies the DF-GLS unit root test to the residuals of model predictions to assess their stationarity, a crucial aspect of time-series analysis. 
The `dm_test_hac_regression` function computes the DM test statistic, evaluating whether the forecast performance of two models differs significantly in terms of prediction accuracy using either Mean Squared Error (MSE) or Mean Absolute Deviation (MAD) as criteria. It also accounts for heteroskedasticity by using HAC standard errors.

Functions:
- `apply_df_gls_test`: Applies the DF-GLS unit root test to the residuals and returns the test statistic and p-value.
- `dm_test_hac_regression`: Calculates the Diebold-Mariano test statistic to compare the forecast accuracy between two models, using MSE or MAD and HAC standard errors.
- `run_dm_test`: Runs the DM test for multiple model comparisons and returns the results, including the DM statistic, p-value, and significance of the difference in forecast performance.
"""
import numpy as np
import pandas as pd 
from arch.unitroot import DFGLS
import statsmodels.api as sm
import os
import sys
import scipy.stats as stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))

#DF-GLS test
def apply_df_gls_test(residuals):
    # Apply the DF-GLS test
    df_gls = DFGLS(residuals)
    
    # Access the test statistic and p-value
    df_gls_statistic = df_gls.stat
    p_value = df_gls.pvalue

    if p_value < 0.05:
        print("The series is likely stationary (reject H0).")
    else:
        print("The series is likely non-stationary (fail to reject H0).")
    
    return df_gls_statistic, p_value


def dm_test_hac_regression(actual, pred1, pred2, h, crit="MSE"):
    # compute the residuals (d_t)
    #H0: both models perform the same H1: bridge model is better
    if crit == "MSE":
        d = (actual - pred1) ** 2 - (actual - pred2) ** 2
    elif crit == "MAD":
        d = np.abs(actual - pred1) - np.abs(actual - pred2)

    print(apply_df_gls_test(d))
    T = len(d)
    print(T)
    d_mean = np.mean(d)
    print(d_mean)
    #compute maxlags as T^(1/3)
    maxlags = int(np.round(T**(1/3))) 

    # regress the residuals (d) on a constant term (intercept)
    X = np.ones(T)  # constant (intercept)
    y = d  # dependent variable is the residuals (d_t)
    
    model = sm.OLS(y, X)
    results = model.fit()
    # calculate HAC standard errors
    hac_results = results.get_robustcov_results(cov_type='HAC', maxlags=maxlags)

    #HAC standard errors
    hac_sd = hac_results.bse[0]  # Extract standard error of the constant term (index 0)
    print(f"HAC Standard Error (maxlags={maxlags}): {hac_sd}")
    
    #calculate the DM statistic using formula in notes
    dm_stat = d_mean / hac_sd


    p_value = 1 - stats.norm.cdf(dm_stat)


    return dm_stat, p_value


def run_dm_test(df):  # takes input from RMSFE function (row_error.csv)
    comparisons = [
        ("model_AR_h1", "model_ADL_bridge_m3"),
        ("model_AR_h2", "model_ADL_bridge_m6"),
        ("model_RF_h1", "model_RF_bridge_m3"),
        ("model_RF_h2", "model_RF_bridge_m6")
        #("model_AR_h1", "combined_bridge_forecast_m3"),
        #("model_AR_h2", "combined_bridge_forecast_m6"),
        #("model_RF_h1", "combined_bridge_forecast_m3"),
        #("model_RF_h2", "combined_bridge_forecast_m6"),
    ]
    
    results = []

    for model1, model2 in comparisons:
        h = int(model1[-1])  # determine forecast horizon (steps)

        # drop NAs
        df_clean = df[["date", "actual_gdp_growth", model1, model2]].dropna()
        df_clean = df_clean.sort_values(by="date", ascending=True)

        actual = df_clean["actual_gdp_growth"]
        pred1 = df_clean[model1]  # benchmark model
        pred2 = df_clean[model2]

        dm_stat, p_val = dm_test_hac_regression(actual, pred1, pred2, h)

        results.append({
            "model1 (benchmark)": model1,
            "model2": model2,
            "Steps": h,
            "t_stat": round(dm_stat, 4),
            "p_value": p_val,
            "Significant (<0.05)": p_val < 0.05
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    file_path = "../Data/row_error.csv"
    df = pd.read_csv(file_path)

    print(run_dm_test(df))























