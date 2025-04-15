"""
This file contains functions for generating out-of-sample (OOS) forecasts, calculating forecast errors, and computing the Root Mean Squared Forecast Error (RMSFE) for different forecasting models.

The primary functions in this file are:
1. `generate_oos_forecast`: This function generates out-of-sample forecasts for multiple models (AR, ADL bridge, RF, RF bridge) using a rolling window approach.
2. `calculate_row_error`: This function calculates the row-level forecast error (prediction minus actual) for each model.
3. `calculate_rmsfe`: This function computes the Root Mean Squared Forecast Error (RMSFE) for each forecast model.

The RMSFE window drops dummy column if covid recession is not in the training window.

It also includes helper functions to handle missing months and preprocess data.
"""

import numpy as np
import pandas as pd 
import os
import sys
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import math
import textwrap
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from model_AR import model_AR
from model_ADL_bridge import model_ADL_bridge
from model_RF import model_RF
from model_RF_bridge import model_RF_bridge

from data_processing import get_missing_months, add_missing_months

pd.set_option('display.float_format', '{:.2f}'.format)

# Forecasts will be generated for the following models and forecasts horizons
# h1 and h2 represent a 1 step and 2 step ahead forecast respectively (Quarterly frequency)
# m1 to m6 represent the monthly flash estimate revisions for bridge models as new monthly data is included in the forecast
model_and_horizon =[
        'model_AR_h1', 'model_AR_h2',
        'model_ADL_bridge_m1', 'model_ADL_bridge_m2', 'model_ADL_bridge_m3',
        'model_ADL_bridge_m4', 'model_ADL_bridge_m5', 'model_ADL_bridge_m6',
        'model_RF_h1', 'model_RF_h2',
        'model_RF_bridge_m1', 'model_RF_bridge_m2', 'model_RF_bridge_m3',
        'model_RF_bridge_m4', 'model_RF_bridge_m5', 'model_RF_bridge_m6'
    ]



def generate_oos_forecast(df, df_nonlinear, window_size=int(12*17.5), time_travel_date=None, usage="multi_period_forecast"):
    """
    Generates out-of-sample forecasts using a rolling window approach for multiple models.

    This function forecasts GDP growth and GDP levels for each forecast horizon (from multiple models: AR, ADL bridge, RF, RF bridge) 
    by iterating over a fixed-size rolling window and applying the appropriate models for each window.

    Window size of 17.5 years ensures that there will be 12.5 years of data to calculate RMSFE from. Since currently we are using 30 years of historical data. 
    12.5 years * 4 quarters = 50 data points to calculate RSMFE.

    Release schedule for selected indicators:
    example: 
        if the current window is predicting the monthly flash estimate of 2012-09-01, our methodology assumes we are at the end of the month (2012-09-30). 
        
        based on Fred MD release schedule the following monthly indicators are typically available by the end of the same month
        -Trade_Balance_lag1 
        -Industrial_Production_lag1
        -Housing_Starts_lag2
        -junk_bond_spread
        -junk_bond_spread_lag1
        -Nonfarm_Payrolls_lag1
        -New_Home_Sales_lag1

        the following will only be available for the previous month:
        -Construction_Spending : typically released on the first day, 2 months later
        -Housing_Starts : typically released the second/third week the next month
        -Capacity_Utilization : typically released the second/third week the next month
        -Unemployment : typically released the first week the next month 
        -New_Orders_Durable_Goods : typically released on the first day, 2 months later
        

    Args:
        df (pd.DataFrame): DataFrame containing the raw data with GDP and other economic indicators.
        df_nonlinear (pd.DataFrame): DataFrame containing additional non-linear data for forecasting.
        window_size (tuple): The rolling window size for training (default is 240 months, equivalent to 20 years).
        time_travel_date (date object): specify quarter and year Q3 2020 -> 2020-09-01

    Returns:
        pd.DataFrame: A DataFrame containing forecasted GDP and growth for each model.
    """

    df['date'] = pd.to_datetime(df['date'])  
    df = df.sort_values(by='date', ascending=False)
    df_nonlinear['date'] = pd.to_datetime(df_nonlinear['date']) 
    df_nonlinear = df_nonlinear.sort_values(by='date', ascending=False)
    
    # Find the index of the last non-NA GDP value. This will be the pointer for where to truncate dataframe
    last_valid_gdp_index = df['GDP'].first_valid_index()
    last_valid_gdp_index_tree = df_nonlinear['GDP'].first_valid_index()
    
    # Using pointer, truncate the dataframe after the current quarter we are presently in
    # this heuristic ensures we only do RMSFE calculations on quarters where GDP is announced.
    start_row_index = df.index.get_loc(last_valid_gdp_index) - 2  
    df_trimmed = df.iloc[start_row_index:].sort_values(by='date', ascending=True).reset_index(drop=True)

    start_row_index_tree = df_nonlinear.index.get_loc(last_valid_gdp_index_tree) - 2 
    df_trimmed_tree = df_nonlinear.iloc[start_row_index_tree:].sort_values(by='date', ascending=True).reset_index(drop=True)
    

    results = pd.DataFrame(columns=['date', 'actual_gdp_growth'] + model_and_horizon)

    if usage == "multi_period_forecast":
        range_for_start_indices = len(df_trimmed) - window_size #- remove_covid
    elif usage == "single_period_forecast": # for time travelling
        range_for_start_indices = 6 # only 6 loops for 6 monthly flash estimates  


    # Sliding window begins
    for start_index in range(range_for_start_indices):
        # if generating RMSFE: use maximum number of sliding windows available
        if usage == "multi_period_forecast":
            end_index = start_index + window_size
            print("Window:", df_trimmed.iloc[start_index]['date'].date(), "to", df_trimmed.iloc[start_index + window_size - 1]['date'].date())

        # if time travelling: only need 6 sliding windows
        elif usage == "single_period_forecast":
            try:
                end_index = df_trimmed.index[df_trimmed['date'] == time_travel_date][0] - 1 + start_index
            except IndexError:
                print(f"Date {time_travel_date} not found in df_trimmed.")
                end_index = None
            start_index = 0
            print("Window:", df_trimmed.iloc[start_index]['date'].date(), "to", df_trimmed.iloc[end_index-1]['date'].date())
        
        historical_data = df_trimmed.iloc[start_index:end_index]
        historical_data_tree = df_trimmed_tree.iloc[start_index:end_index]

        # Drop dummy variable if the window does not contain covid recession period
        # prevents model from fitting on a column of 0s only
        if 1 not in historical_data['dummy'].values:
            historical_data = historical_data.drop(columns=['dummy'])

        if 1 not in historical_data_tree['dummy'].values:
            historical_data_tree = historical_data_tree.drop(columns=['dummy'])

        # Rest index and sort in descending order to prepare dataframe for adding rows
        historical_data = historical_data.sort_values(by='date', ascending=False).reset_index(drop=True)
        historical_data_tree = historical_data_tree.sort_values(by='date', ascending=False).reset_index(drop=True)

        # Add 3 months of empty data above the current dataframe to add an additional quarter to forecast
        historical_data = add_missing_months(historical_data, date_column="date")
        historical_data_tree = add_missing_months(historical_data_tree, date_column="date")

        # Recompute gdp_growth_lag1 from gdp_growth since we added rows of NAs
        if 'gdp_growth' in historical_data.columns:
            historical_data['gdp_growth_lag1'] = historical_data['gdp_growth'].shift(3) 

            last_valid_idx_lag1 = historical_data['gdp_growth_lag1'].last_valid_index()

            # Additional safety check to ensure only one value for gdp_growth_lag1 is shifted up to prevent data leakage.
            if last_valid_idx_lag1 is not None:
                historical_data.loc[last_valid_idx_lag1, 'gdp_growth_lag1'] = pd.NA

        
        forecast_date = df_trimmed.iloc[end_index-1]['date']
        print("forecasting this date revision", forecast_date)


        gdp_valid_indices = historical_data['GDP'].dropna().index
        last_gdp = second_last_gdp = None
        if len(gdp_valid_indices) >= 1:
            # Remove the GDP value for quarter being forecasted to trigger heuristic that tells model to forecast this quarter (simulates GDP being unreleased)
            last_gdp = historical_data.at[gdp_valid_indices[-1], 'GDP']
            historical_data.at[gdp_valid_indices[-1], 'GDP'] = float('nan')
            # Ensure we are not leaking data by labelling prediction rows with dummy = 1 
            # if dummy is 1, set to 0
            if 'dummy' in historical_data_tree.columns:
                if historical_data.at[gdp_valid_indices[-1], 'dummy'] == 1:
                    historical_data.at[gdp_valid_indices[-1], 'dummy'] = 0

        #repeat for tree
        gdp_valid_indices_tree = historical_data_tree['GDP'].dropna().index
        if len(gdp_valid_indices_tree) >= 1:
            # Remove the GDP value for quarter being forecasted to trigger heuristic that tells model to forecast this quarter (simulates GDP being unreleased)
            historical_data_tree.at[gdp_valid_indices_tree[-1], 'GDP'] = float('nan')
            # Ensure we are not leaking data by labelling prediction rows with dummy = 1 
            # if dummy is 1, set to 0
            if 'dummy' in historical_data_tree.columns:
                if historical_data_tree.at[gdp_valid_indices_tree[-1], 'dummy'] == 1:
                    historical_data_tree.at[gdp_valid_indices_tree[-1], 'dummy'] = 0

        gdp_growth_valid_indices = historical_data['gdp_growth'].dropna().index
        last_gdp_growth = None
        if len(gdp_growth_valid_indices) >= 1:
            # Ensure we are not leaking data by removing the gdp_growth value for quarter being forecasted
            last_gdp_growth = historical_data.at[gdp_growth_valid_indices[-1], 'gdp_growth']
            historical_data.at[gdp_growth_valid_indices[-1], 'gdp_growth'] = float('nan')

        #repeat above for tree
        gdp_growth_valid_indices_tree = historical_data_tree['gdp_growth'].dropna().index
        if len(gdp_growth_valid_indices_tree) >= 1:
            # Ensure we are not leaking data by removing the gdp_growth value for quarter being forecasted
            historical_data_tree.at[gdp_growth_valid_indices_tree[-1], 'gdp_growth'] = float('nan')

        actual_gdp_growth = last_gdp_growth # store the value for evaluation of prediction results

        # These indicators are typically only released one month later
        # drop these value from the monthly indicators to simulate them being unreleased
        indicators_released_late = ["Construction_Spending", "Housing_Starts", "Capacity_Utilization", "Unemployment", "New_Orders_Durable_Goods"]

        for indic in indicators_released_late:
            if indic in historical_data.columns:
                valid_indices = historical_data[indic].dropna().index
                if len(valid_indices) >= 1:
                    historical_data.at[valid_indices[-1], indic] = float('nan') 
            if indic in historical_data_tree.columns:
                valid_indices = historical_data_tree[indic].dropna().index
                if len(valid_indices) >= 1:
                    historical_data_tree.at[valid_indices[-1], indic] = float('nan') 

        #### FORECAST FROM  ADL BRIDGE ####
        model_adl_output = model_ADL_bridge(historical_data)
        #print("model_adl_output", model_adl_output)

        #### FORECAST FROM RF BRIDGE ####
        model_rf_bridge_output = model_RF_bridge(historical_data_tree)  
        #print("model_adl_output", model_adl_output)
        
        # Heuristic to sort the results into the correct column in results dataframe is determined by forecast month
        month_of_forecast = forecast_date.month 

        if month_of_forecast in [1, 4, 7, 10]:  # January, April, July, October -> m1, m4
            model_adl_m1 = model_adl_output.iloc[0]['Nowcasted_GDP_Growth']  # m1 month 1 flash estimate
            model_adl_m4 = model_adl_output.iloc[1]['Nowcasted_GDP_Growth']  # m4 month 4 flash estimate

            model_rf_bridge_m1 = model_rf_bridge_output.iloc[0]['Nowcasted_GDP_Growth']  # m1 month 1 flash estimate
            model_rf_bridge_m4 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP_Growth']  # m4 month 4 flash estimate 

            print("m1 revision")
        elif month_of_forecast in [2, 5, 8, 11]:  # February, May, August, November -> m2, m5
            model_adl_m2 = model_adl_output.iloc[0]['Nowcasted_GDP_Growth']  # m2 month 2 flash estimate
            model_adl_m5 = model_adl_output.iloc[1]['Nowcasted_GDP_Growth']  # m5 month 5 flash estimate

            model_rf_bridge_m2 = model_rf_bridge_output.iloc[0]['Nowcasted_GDP_Growth']  # m2 month 2 flash estimate
            model_rf_bridge_m5 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP_Growth']  # m5 month 5 flash estimate

            print("m2 revision")
        else:  # March, June, September, December -> m3, m6
            model_adl_m3 = model_adl_output.iloc[0]['Nowcasted_GDP_Growth']  # m3 month 3 flash estimate
            model_adl_m6 = model_adl_output.iloc[1]['Nowcasted_GDP_Growth']  # m6 month 6 flash estimate
        
            model_rf_bridge_m3 = model_rf_bridge_output.iloc[0]['Nowcasted_GDP_Growth']  # m3 month 3 flash estimate
            model_rf_bridge_m6 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP_Growth']  # m6 month 6 flash estimate
        
            print("m3 revision")
        

        # AR Forecast — only run model_AR on the third? month of each quarter
        if month_of_forecast in [3, 6, 9, 12]:
            #### FORECAST FROM  AR BENCHMARK ####
            model_ar_output = model_AR(historical_data)
            model_ar_h1 = model_ar_output.iloc[0]['Nowcasted_GDP_Growth'] # h1: nowcast current quarter
            model_ar_h2 = model_ar_output.iloc[1]['Nowcasted_GDP_Growth'] # h2: forecast next quarter

            #### FORECAST FROM  RF BENCHMARK ####
            model_RF_output = model_RF(historical_data_tree)
            model_RF_h1 = model_RF_output.iloc[0]['Nowcasted_GDP_Growth'] # h1: nowcast current quarter
            model_RF_h2 = model_RF_output.iloc[1]['Nowcasted_GDP_Growth'] # h2: forecast next quarter

        else:
            model_ar_h1 = None
            model_ar_h2 = None
            model_RF_h1 = None
            model_RF_h2 = None

        
        results = pd.concat([results, pd.DataFrame({
            'date': [forecast_date],  
            'actual_gdp_growth': [actual_gdp_growth],
            'model_AR_h1': [model_ar_h1],
            'model_AR_h2': [model_ar_h2],
            'model_ADL_bridge_m1': [model_adl_m1 if month_of_forecast in [1, 4, 7, 10] else None],
            'model_ADL_bridge_m2': [model_adl_m2 if month_of_forecast in [2, 5, 8, 11] else None],
            'model_ADL_bridge_m3': [model_adl_m3 if month_of_forecast in [3, 6, 9, 12] else None],
            'model_ADL_bridge_m4': [model_adl_m4 if month_of_forecast in [1, 4, 7, 10] else None],
            'model_ADL_bridge_m5': [model_adl_m5 if month_of_forecast in [2, 5, 8, 11] else None],
            'model_ADL_bridge_m6': [model_adl_m6 if month_of_forecast in [3, 6, 9, 12] else None],
            'model_RF_h1': [model_RF_h1],
            'model_RF_h2': [model_RF_h2],
            'model_RF_bridge_m1': [model_rf_bridge_m1 if month_of_forecast in [1, 4, 7, 10] else None],
            'model_RF_bridge_m2': [model_rf_bridge_m2 if month_of_forecast in [2, 5, 8, 11] else None],
            'model_RF_bridge_m3': [model_rf_bridge_m3 if month_of_forecast in [3, 6, 9, 12] else None],
            'model_RF_bridge_m4': [model_rf_bridge_m4 if month_of_forecast in [1, 4, 7, 10] else None],
            'model_RF_bridge_m5': [model_rf_bridge_m5 if month_of_forecast in [2, 5, 8, 11] else None],
            'model_RF_bridge_m6': [model_rf_bridge_m6 if month_of_forecast in [3, 6, 9, 12] else None],
        })], ignore_index=True)

    results['model_ADL_bridge_m4'] = results['model_ADL_bridge_m4'].shift(3)
    results['model_ADL_bridge_m5'] = results['model_ADL_bridge_m5'].shift(3)
    results['model_ADL_bridge_m6'] = results['model_ADL_bridge_m6'].shift(3)

    results['model_AR_h2'] = results['model_AR_h2'].shift(3)

    results['model_RF_h2'] = results['model_RF_h2'].shift(3)

    results['model_RF_bridge_m4'] = results['model_RF_bridge_m4'].shift(3)
    results['model_RF_bridge_m5'] = results['model_RF_bridge_m5'].shift(3)
    results['model_RF_bridge_m6'] = results['model_RF_bridge_m6'].shift(3)
    
    return results

def calculate_row_error(df):
    """
    Calculates row-level forecast errors (prediction - actual) for multiple forecast models.

    This function groups the data by the first month of each quarter and computes the forecast errors for each model. 
    The forecast errors are calculated as the difference between the forecasted values and the actual GDP values.

    Args:
        df (pd.DataFrame): DataFrame containing the actual GDP values and the forecasted values from various models.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast errors for each model, along with the actual GDP values for each row.
    """

    df['date'] = pd.to_datetime(df['date'])

    def get_quarter_start_date(d):
        if d.month < 4:
            return pd.Timestamp(f"{d.year}-01-01")
        elif d.month < 7:
            return pd.Timestamp(f"{d.year}-04-01")
        elif d.month < 10:
            return pd.Timestamp(f"{d.year}-07-01")
        else:
            return pd.Timestamp(f"{d.year}-10-01")

    df['quarter_start'] = df['date'].apply(get_quarter_start_date)

    agg_dict = {
        col: 'first' for col in df.columns
        if col not in ['date', 'quarter_start']
    }

    result = df.groupby('quarter_start').agg(agg_dict).reset_index()
    result = result.rename(columns={'quarter_start': 'date'})

    forecast_cols = model_and_horizon

    # Calculate row-by-row forecast errors (prediction - actual)
    for col in forecast_cols:
        result[f'Error_{col}'] = result[col] - result['actual_gdp_growth']

    return result


def calculate_rmsfe_and_mae(df):
    """
    Calculates the Root Mean Squared Forecast Error (RMSFE) and Mean Absolute Error (MAE)
    for each model's forecast, based on quarterly actual and predicted GDP growth rates.

    Args:
        df (pd.DataFrame): DataFrame containing 'date', 'actual_gdp_growth', and forecast columns.

    Returns:
        tuple:
            pd.DataFrame: A one-row DataFrame with RMSFE values for each forecast column.
            pd.DataFrame: A one-row DataFrame with MAE values for each forecast column.
    """

    df['date'] = pd.to_datetime(df['date'])

    # Create a column to identify the first month of each quarter
    def get_quarter_start_date(d):
        if d.month < 4:
            return pd.Timestamp(f"{d.year}-01-01")
        elif d.month < 7:
            return pd.Timestamp(f"{d.year}-04-01")
        elif d.month < 10:
            return pd.Timestamp(f"{d.year}-07-01")
        else:
            return pd.Timestamp(f"{d.year}-10-01")

    df['quarter_start'] = df['date'].apply(get_quarter_start_date)

    # Group by quarter_start, take the first non null value for each column
    agg_dict = {
        col: 'first' for col in df.columns
        if col not in ['date', 'quarter_start']
    }

    result = df.groupby('quarter_start').agg(agg_dict).reset_index()
    result = result.rename(columns={'quarter_start': 'date'})

    # Identify the forecast columns 
    forecast_cols = [col for col in result.columns if col not in ['date', 'actual_gdp_growth']]

    rmsfe_dict = {}
    mae_dict = {}

    for col in forecast_cols:
        valid_rows = result[[col, 'actual_gdp_growth']].dropna()
        error = valid_rows[col] - valid_rows['actual_gdp_growth']
        rmsfe = np.sqrt((error ** 2).mean())
        mae = np.abs(error).mean()
        rmsfe_dict[f'RMSFE_{col}'] = rmsfe
        mae_dict[f'MAE_{col}'] = mae

    return pd.DataFrame([rmsfe_dict]), pd.DataFrame([mae_dict])

def add_combined_bridge_forecasts(df):
    """
    Adds combined forecast columns to the DataFrame, averaging ADL bridge and RF bridge forecasts for m1 to m6.

    Args:
        df (pd.DataFrame): The forecast results DataFrame with ADL and RF bridge model forecasts.

    Returns:
        pd.DataFrame: The original DataFrame with new combined forecast columns added.
    """
    for m in range(1, 7):
        adl_col = f'model_ADL_bridge_m{m}'
        rf_col = f'model_RF_bridge_m{m}'
        combined_col = f'combined_bridge_forecast_m{m}'
        df[combined_col] = df[[adl_col, rf_col]].mean(axis=1)
    
    return df

def plot_residuals(error_df, parts=3, cols_per_row=2):
    """
    Splits the residual plots into batches and displays each batch in a clean, readable grid layout.

    Args:
        error_df (pd.DataFrame): Output from calculate_row_error function with 'Error_' columns.
        parts (int): Number of plot batches.
        cols_per_row (int): Number of plots per row.
    """
    error_cols = [col for col in error_df.columns if col.startswith("Error_")]
    chunk_size = math.ceil(len(error_cols) / parts)

    for part in range(parts):
        chunk = error_cols[part * chunk_size : (part + 1) * chunk_size]
        num_rows = math.ceil(len(chunk) / cols_per_row)

        fig, axs = plt.subplots(num_rows, cols_per_row,
                                figsize=(cols_per_row * 6, num_rows * 4),
                                constrained_layout=True)

        axs = axs.flatten()  
        for ax in axs[len(chunk):]:
            ax.axis('off') 

        for i, col in enumerate(chunk):
            sns.histplot(error_df[col].dropna(), kde=True, bins=30, ax=axs[i])

            title = col.replace("Error_", "")
            wrapped_title = "\n".join(textwrap.wrap(title, width=25))
            axs[i].set_title(wrapped_title, fontsize=11)

            axs[i].set_xlabel("Error", fontsize=10)
            axs[i].set_ylabel("Frequency", fontsize=10)

        fig.suptitle(f"Residual Distributions (Part {part + 1})", fontsize=16)
        plt.show()

def drop_covid(df):
    """
    Drops rows where the year is 2020 or the date is in January 2021.
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Ensure datetime format

    return df[(df['date'].dt.year != 2020) & (df['date'] != pd.Timestamp("2021-01-01"))] 


def calculate_skew_kurtosis(error_df):
    """
    Calculates skewness and kurtosis for each column in the DataFrame that starts with 'Error_'.

    Args:
        error_df (pd.DataFrame): DataFrame containing residual error columns (prefixed with 'Error_').

    Returns:
        pd.DataFrame: A DataFrame with columns ['Model', 'Skewness', 'Kurtosis'].
    """
    error_cols = [col for col in error_df.columns if col.startswith("Error_")]

    results = []
    for col in error_cols:
        residuals = error_df[col].dropna()
        col_skew = skew(residuals)
        col_kurt = kurtosis(residuals, fisher=True)  # Fisher=True gives excess kurtosis

        results.append({
            "Model": col.replace("Error_", ""),
            "Skewness": col_skew,
            "Kurtosis": col_kurt
        })

    return pd.DataFrame(results)



if __name__ == "__main__":
    file_path1 = "../Data/bridge_df.csv"
    file_path2 = "../Data/tree_df.csv"
    df = pd.read_csv(file_path1)
    df_nonlinear = pd.read_csv(file_path2)


    # Test RMSFE generation 
    

    res = generate_oos_forecast(df, df_nonlinear)
    res = add_combined_bridge_forecasts(res)
    model_and_horizon += [
        'combined_bridge_forecast_m1', 'combined_bridge_forecast_m2', 'combined_bridge_forecast_m3',
        'combined_bridge_forecast_m4', 'combined_bridge_forecast_m5', 'combined_bridge_forecast_m6'
        ]

    row_error_df = calculate_row_error(res)

    row_error_df_dropped_covid = drop_covid(row_error_df)

    res_dropped_covid = drop_covid(res)

    rmsfe_df, mae_df = calculate_rmsfe_and_mae(res)
    rmsfe_df_dropped_covid, mae_df_dropped_covid= calculate_rmsfe_and_mae(res_dropped_covid)

    #plot_residuals(row_error_df)
    #print(res_drop_covid)
    print(row_error_df); print(rmsfe_df)

    row_error_df.to_csv("../Data/row_error.csv", index=False);
    row_error_df_dropped_covid.to_csv("../Data/row_error_dropped_covid.csv", index=False);
    rmsfe_df.to_csv("../Data/rmsfe.csv", index=False)
    mae_df.to_csv("../Data/mae_df.csv", index=False)
    rmsfe_df_dropped_covid.to_csv("../Data/rmsfe_dropped_covid.csv", index=False)
    mae_df_dropped_covid.to_csv("../Data/mae_df_dropped_covid.csv", index=False)

    distribution_no_covid = calculate_skew_kurtosis(row_error_df_dropped_covid)
    distribution = calculate_skew_kurtosis(row_error_df)

    distribution_no_covid.to_csv("../Data/distribution_no_covid.csv", index=False)
    distribution.to_csv("../Data/distribution.csv", index=False)


    # Test single window forecast
    """
    res_time_travel = generate_oos_forecast(df, df_nonlinear, time_travel_date="2018-03-01", usage="single_period_forecast")
    print(res_time_travel)
    res_time_travel.to_csv("../Data/res_time_travel.csv", index=False)
    """



# sample
