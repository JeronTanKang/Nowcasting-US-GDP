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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from model_AR import model_AR
from model_ADL_bridge import model_ADL_bridge
from model_RF import model_RF
from model_RF_bridge import model_RF_bridge

from data_processing import get_missing_months, add_missing_months

pd.set_option('display.float_format', '{:.2f}'.format)

model_and_horizon =[
        'model_AR_h1', 'model_AR_h2',
        'model_ADL_bridge_m1', 'model_ADL_bridge_m2', 'model_ADL_bridge_m3',
        'model_ADL_bridge_m4', 'model_ADL_bridge_m5', 'model_ADL_bridge_m6',
        'model_RF_h1', 'model_RF_h2',
        'model_RF_bridge_m1', 'model_RF_bridge_m2', 'model_RF_bridge_m3',
        'model_RF_bridge_m4', 'model_RF_bridge_m5', 'model_RF_bridge_m6'
    ]



def generate_oos_forecast(df, df_nonlinear, window_size=(12*20), time_travel_date=None, usage="multi_period_forecast"):
    """
    Generates out-of-sample forecasts using a rolling window approach for multiple models.

    This function forecasts GDP growth and GDP levels for each forecast horizon (from multiple models: AR, ADL bridge, RF, RF bridge) 
    by iterating over a fixed-size rolling window and applying the appropriate models for each window.

    Args:
        df (pd.DataFrame): DataFrame containing the raw data with GDP and other economic indicators.
        df_nonlinear (pd.DataFrame): DataFrame containing additional non-linear data for forecasting.
        window_size (tuple): The rolling window size for training (default is 240 months, equivalent to 20 years).
        time_travel_date (date object): specify quarter and year Q3 2020 -> 2020-09-01

    Returns:
        pd.DataFrame: A DataFrame containing forecasted GDP and growth for each model.
    """


    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
    df = df.sort_values(by='date', ascending=False)
    df_nonlinear['date'] = pd.to_datetime(df_nonlinear['date'])  # Ensure 'date' is in datetime format
    df_nonlinear = df_nonlinear.sort_values(by='date', ascending=False)
    
    # Find the index of the last non-NA GDP value
    last_valid_gdp_index = df['GDP'].first_valid_index()
    last_valid_gdp_index_tree = df_nonlinear['GDP'].first_valid_index()
    
    ### Trims the dataframe such that the complete quarter is taken
    ### this ensures that we only do RMSFE calculations on quarters where GDP is announced.
    # Get the date two rows above the last available GDP
    start_row_index = df.index.get_loc(last_valid_gdp_index) - 2  # Two rows above the last valid GDP to complete the quarter
    # Trim the DataFrame to include rows starting from this point onward
    df_trimmed = df.iloc[start_row_index:].sort_values(by='date', ascending=True).reset_index(drop=True)

    start_row_index_tree = df_nonlinear.index.get_loc(last_valid_gdp_index_tree) - 2 
    df_trimmed_tree = df_nonlinear.iloc[start_row_index_tree:].sort_values(by='date', ascending=True).reset_index(drop=True)

    #print("df_trimmed",df_trimmed)
    #print("df_trimmed_tree",df_trimmed_tree)
    
    # Initialize results dataframe

    results = pd.DataFrame(columns=['date', 'actual_gdp'] + model_and_horizon)

    #df_sorted = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    
    # Create window: Loop through the trimmed dataframe using a fixed rolling window size
    remove_covid = 12*5

    #if usage = "multi_period_forecast": for start_index in range(len(df_trimmed) - window_size):
    #else: for end_index in [end_index, end_index+1]
    # start index = 0
    # end index = end of forecast,
    if usage == "multi_period_forecast":
        range_for_start_indices = len(df_trimmed) - window_size #- remove_covid
    elif usage == "single_period_forecast":
        range_for_start_indices = 6 # only 6 loops for 6 months forecast 

    for start_index in range(range_for_start_indices):
        print("Window:", df_trimmed.iloc[start_index]['date'].date(), "to", df_trimmed.iloc[start_index + window_size - 1]['date'].date())
        

        if usage == "multi_period_forecast":
            end_index = start_index + window_size
        elif usage == "single_period_forecast":
            try:
                end_index = df_trimmed.index[df_trimmed['date'] == time_travel_date][0] - 1 + start_index
            except IndexError:
                print(f"Date {time_travel_date} not found in df_trimmed.")
                end_index = None
        
        historical_data = df_trimmed.iloc[start_index:end_index]
        historical_data_tree = df_trimmed_tree.iloc[start_index:end_index]

        # drop dummy variable if the window does not contain covid recession period
        if 1 not in historical_data['dummy'].values:
            historical_data = historical_data.drop(columns=['dummy'])
            #print('dropped dummy')

        if 1 not in historical_data_tree['dummy'].values:
            historical_data_tree = historical_data_tree.drop(columns=['dummy'])
            #print('dropped dummy for tree')

        historical_data = historical_data.sort_values(by='date', ascending=False).reset_index(drop=True)
        historical_data_tree = historical_data_tree.sort_values(by='date', ascending=False).reset_index(drop=True)

        # adds 3 months above to add an additional quarter to forecast
        historical_data = add_missing_months(historical_data, date_column="date")
        historical_data_tree = add_missing_months(historical_data_tree, date_column="date")

        # Recompute gdp_growth_lag2 from gdp_growth. will need to do this for lag 1 too
        if 'gdp_growth' in historical_data.columns:
            historical_data['gdp_growth_lag2'] = historical_data['gdp_growth'].shift(6)

            last_valid_idx = historical_data['gdp_growth_lag2'].last_valid_index()

            # Replace it with NaN
            #The logic is so that we only add one gdp_growth_lag2 because we assume we do not have gdp release for the latest quarter in the window
            if last_valid_idx is not None:
                historical_data.loc[last_valid_idx, 'gdp_growth_lag2'] = pd.NA

        
        forecast_date = df_trimmed.iloc[end_index-1]['date']

        print("forecasting this date revision", forecast_date)


        # remove last available GDP THIS STEP IS VERY IMPORTANT AND HAS TO BE DONE BEFORE DATA IS FED INTO MODEL TO PREVENT LEAKAGE 
        # in that case I need to remove gdp_growth too becos this is what im predicting

        # in adition if any of the forecast periods are labelled dummy = 1, drop dummy col
        # intuition: dummy variable only exists to dampen the covid quarters when model fitting

        ## Store and remove last and second last non-NaN GDP values
        gdp_valid_indices = historical_data['GDP'].dropna().index
        last_gdp = second_last_gdp = None
        if len(gdp_valid_indices) >= 1:
            last_gdp = historical_data.at[gdp_valid_indices[-1], 'GDP']
            historical_data.at[gdp_valid_indices[-1], 'GDP'] = float('nan')
            # if dummy is 1, set to 0
            if 'dummy' in historical_data_tree.columns:
                if historical_data.at[gdp_valid_indices[-1], 'dummy'] == 1:
                    historical_data.at[gdp_valid_indices[-1], 'dummy'] = 0

        #repeat for tree
        gdp_valid_indices_tree = historical_data_tree['GDP'].dropna().index
        last_gdp_tree = second_last_gdp_tree = None
        if len(gdp_valid_indices_tree) >= 1:
            #last_gdp_tree = historical_data_tree.at[gdp_valid_indices_tree[-1], 'GDP']
            historical_data_tree.at[gdp_valid_indices_tree[-1], 'GDP'] = float('nan')
            # if dummy is 1, set to 0
            if 'dummy' in historical_data_tree.columns:
                if historical_data_tree.at[gdp_valid_indices_tree[-1], 'dummy'] == 1:
                    historical_data_tree.at[gdp_valid_indices_tree[-1], 'dummy'] = 0


        actual_gdp = last_gdp

        # commenting out below becos i think i can remove it. i dont think i need to store this value i dont need to use it in the future
        ## Store and remove last and second last non-NaN GDP_Growth values
        gdp_growth_valid_indices = historical_data['gdp_growth'].dropna().index
        last_gdp_growth = second_last_gdp_growth = None
        if len(gdp_growth_valid_indices) >= 1:
            #last_gdp_growth = historical_data.at[gdp_growth_valid_indices[-1], 'gdp_growth']
            historical_data.at[gdp_growth_valid_indices[-1], 'gdp_growth'] = float('nan')

        #repeat above for tree
        gdp_growth_valid_indices_tree = historical_data_tree['gdp_growth'].dropna().index
        last_gdp_growth_tree = second_last_gdp_growth_tree = None
        if len(gdp_growth_valid_indices_tree) >= 1:
            #last_gdp_growth = historical_data_tree.at[gdp_growth_valid_indices_tree[-1], 'gdp_growth']
            historical_data_tree.at[gdp_growth_valid_indices_tree[-1], 'gdp_growth'] = float('nan')

        #print("prediction df", historical_data.tail(13))

        #### FORECAST FROM  ADL BRIDGE ####
        model_adl_output = model_ADL_bridge(historical_data)  # Get the model output DataFrame
        #print("model_adl_output", model_adl_output)

        #### FORECAST FROM RF BRIDGE ####
        model_rf_bridge_output = model_RF_bridge(historical_data_tree)  # Get the model output DataFrame
        #print("model_adl_output", model_adl_output)
        
        # Determine the forecast horizon (using a heuristic based on the month)
        month_of_forecast = forecast_date.month  # This will help decide which forecast to use

        if month_of_forecast in [1, 4, 7, 10]:  # January, April, July, October -> m1, m4
            model_adl_m1 = model_adl_output.iloc[0]['Nowcasted_GDP']  # m1 forecast
            model_adl_m4 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m4 forecast

            model_rf_bridge_m1 = model_rf_bridge_output.iloc[0]['Nowcasted_GDP']  # m1 forecast
            model_rf_bridge_m4 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP']  # m4 forecast 

            print("m1 revision")
        elif month_of_forecast in [2, 5, 8, 11]:  # February, May, August, November -> m2, m5
            model_adl_m2 = model_adl_output.iloc[0]['Nowcasted_GDP']  # m2 forecast
            model_adl_m5 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m5 forecast

            model_rf_bridge_m2 = model_rf_bridge_output.iloc[0]['Nowcasted_GDP']  # m2 forecast
            model_rf_bridge_m5 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP']  # m5 forecast

            print("m2 revision")
        else:  # March, June, September, December -> m3, m6
            model_adl_m3 = model_adl_output.iloc[0]['Nowcasted_GDP']  # m3 forecast
            model_adl_m6 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m6 forecast
        
            model_rf_bridge_m3 = model_rf_bridge_output.iloc[0]['Nowcasted_GDP']  # m3 forecast
            model_rf_bridge_m6 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP']  # m6 forecast
        
            print("m3 revision")
        
        #### FORECAST FROM  AR BENCHMARK ####
        # AR Forecast — only run model_AR on the first month of each quarter
        if month_of_forecast in [1, 4, 7, 10]:
            model_ar_output = model_AR(historical_data)
            model_ar_h1 = model_ar_output.iloc[0]['Nowcasted_GDP']
            model_ar_h2 = model_ar_output.iloc[1]['Nowcasted_GDP']

            # insert forecast from RF BENCHMARK
            #print("df fed into RF model", historical_data_tree.tail(15))
            model_RF_output = model_RF(historical_data_tree)
            model_RF_h1 = model_RF_output.iloc[0]['Nowcasted_GDP']
            model_RF_h2 = model_RF_output.iloc[1]['Nowcasted_GDP']


        else:
            model_ar_h1 = None
            model_ar_h2 = None
            model_RF_h1 = None
            model_RF_h2 = None

        
        results = pd.concat([results, pd.DataFrame({
            'date': [forecast_date],  
            'actual_gdp': [actual_gdp],
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
        result[f'Error_{col}'] = result[col] - result['actual_gdp']

    return result


def calculate_rmsfe(df):
    """
    Calculates the Root Mean Squared Forecast Error (RMSFE) for each model's forecast.

    Args:
        df (pd.DataFrame): DataFrame containing 'date', 'actual_gdp', and forecast columns.

    Returns:
        pd.DataFrame: A one-row DataFrame with RMSFE values for each forecast column.
    """

    # Ensure 'date' column is in datetime format
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

    # Group by quarter_start, taking the first non-null value for each column
    agg_dict = {
        col: 'first' for col in df.columns
        if col not in ['date', 'quarter_start']
    }

    result = df.groupby('quarter_start').agg(agg_dict).reset_index()
    result = result.rename(columns={'quarter_start': 'date'})

    # Identify forecast columns (excluding 'date' and 'actual_gdp')
    forecast_cols = [col for col in result.columns if col not in ['date', 'actual_gdp']]

    # Compute RMSFE for each forecast column
    rmsfe_dict = {}
    for col in forecast_cols:
        valid_rows = result[[col, 'actual_gdp']].dropna()
        error = valid_rows[col] - valid_rows['actual_gdp']
        rmsfe = np.sqrt((error ** 2).mean())
        rmsfe_dict[f'RMSFE_{col}'] = rmsfe

    # Convert to single-row DataFrame
    return pd.DataFrame([rmsfe_dict])

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

import matplotlib.pyplot as plt
import seaborn as sns
import math
import textwrap

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

        axs = axs.flatten()  # Make it easy to index
        for ax in axs[len(chunk):]:
            ax.axis('off')  # Hide unused subplots

        for i, col in enumerate(chunk):
            sns.histplot(error_df[col].dropna(), kde=True, bins=30, ax=axs[i])

            # Wrap long model names
            title = col.replace("Error_", "")
            wrapped_title = "\n".join(textwrap.wrap(title, width=25))
            axs[i].set_title(wrapped_title, fontsize=11)

            axs[i].set_xlabel("Error", fontsize=10)
            axs[i].set_ylabel("Frequency", fontsize=10)

        fig.suptitle(f"Residual Distributions (Part {part + 1})", fontsize=16)
        plt.show()

def drop_covid(df):
    """
    Drops rows where the year in the 'date' column is 2020,
    and also removes the specific date 2021-01-01.

    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: Filtered DataFrame with 2020 and 2021-01-01 removed.
    """
    df['date'] = pd.to_datetime(df['date'])  

    return df[(df['date'].dt.year != 2020) & (df['date'] != pd.Timestamp("2021-01-01"))] # fix this its not dropping the whole row

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
res_drop_covid = drop_covid(res)
row_error_df = calculate_row_error(res_drop_covid); rmsfe_df = calculate_rmsfe(res_drop_covid)
#plot_residuals(row_error_df)
print(res_drop_covid)
print(row_error_df); print(rmsfe_df)

row_error_df.to_csv("../Data/row_error.csv", index=False); rmsfe_df.to_csv("../Data/rmsfe.csv", index=False)


# Test single window forecast
"""
res_time_travel = generate_oos_forecast(df, df_nonlinear, time_travel_date="2016-03-01", usage="single_period_forecast")
print(res_time_travel)
res_time_travel.to_csv("../Data/res_time_travel.csv", index=False)
"""