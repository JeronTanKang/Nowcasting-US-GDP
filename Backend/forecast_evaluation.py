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
        'model_AR h1', 'model_AR h2',
        'model_ADL_bridge m1', 'model_ADL_bridge m2', 'model_ADL_bridge m3',
        'model_ADL_bridge m4', 'model_ADL_bridge m5', 'model_ADL_bridge m6',
        'model_RF h1', 'model_RF h2',
        'model_RF_bridge m1', 'model_RF_bridge m2', 'model_RF_bridge m3',
        'model_RF_bridge m4', 'model_RF_bridge m5', 'model_RF_bridge m6'
    ]

def generate_oos_forecast(df, df_nonlinear, window_size=(12*20)):
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
    results = pd.DataFrame(columns=[
        'date', 'actual gdp', 
        'model_AR h1', 'model_AR h2',  # Placeholder for AR model forecasts (to be filled later)
        'model_ADL_bridge m1', 'model_ADL_bridge m2', 'model_ADL_bridge m3',
        'model_ADL_bridge m4', 'model_ADL_bridge m5', 'model_ADL_bridge m6',
        'model_RF h1', 'model_RF h2',
        'model_RF_bridge m1', 'model_RF_bridge m2', 'model_RF_bridge m3',
        'model_RF_bridge m4', 'model_RF_bridge m5', 'model_RF_bridge m6'
    ])

    #df_sorted = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    
    # Loop through the trimmed dataframe using a fixed rolling window size
    remove_covid = 12*5
    for start_index in range(len(df_trimmed) - window_size - remove_covid):
        print("Window:", df_trimmed.iloc[start_index]['date'].date(), "to", df_trimmed.iloc[start_index + window_size - 1]['date'].date())
        
        # Get the historical data window of size `window_size`
        end_index = start_index + window_size
        historical_data = df_trimmed.iloc[start_index:end_index]
        historical_data_tree = df_trimmed_tree.iloc[start_index:end_index]

        historical_data = historical_data.sort_values(by='date', ascending=False).reset_index(drop=True)
        historical_data_tree = historical_data_tree.sort_values(by='date', ascending=False).reset_index(drop=True)

        #print("before bfill", historical_data.head(10))
        #print("before shift up", historical_data.head(10))

        # bfill gdp here
        #historical_data['GDP'] = historical_data['GDP'].fillna(method='bfill')
        #historical_data['GDP'] = historical_data['GDP'].shift(-1)

        #print("after shift up", historical_data.head(10))

        historical_data = add_missing_months(historical_data, date_column="date")
        historical_data_tree = add_missing_months(historical_data_tree, date_column="date")

        # Recompute gdp_growth_lag2 from gdp_growth
        if 'gdp_growth' in historical_data.columns:
            historical_data['gdp_growth_lag2'] = historical_data['gdp_growth'].shift(6)

            last_valid_idx = historical_data['gdp_growth_lag2'].last_valid_index()

            # Replace it with NaN
            #The logic is so that we only add one gdp_growth_lag2 because we assume we do not have gdp release for the latest quarter in the window
            if last_valid_idx is not None:
                historical_data.loc[last_valid_idx, 'gdp_growth_lag2'] = pd.NA



        #print("historical_data", historical_data.tail(15))

        
        # Get actual GDP value for the current period (the next row outside the window)
        #actual_gdp = df_trimmed.iloc[end_index]['GDP']  # The next row after the window  # this will no longer be the actual gdp. actual gdp will now be for the quarter that the last month in the window belongs to

        date = df_trimmed.iloc[end_index-1]['date']

        print("forecasting this date revision", date)
        
        # Adjust the date to the first month of the quarter
        if date.month < 4:  # January, February, March
            adjusted_date = pd.to_datetime(f'{date.year}-01-01')
        elif date.month < 7:  # April, May, June
            adjusted_date = pd.to_datetime(f'{date.year}-04-01')
        elif date.month < 10:  # July, August, September
            adjusted_date = pd.to_datetime(f'{date.year}-07-01')
        else:  # October, November, December
            adjusted_date = pd.to_datetime(f'{date.year}-10-01')

        # Forecast using model_ADL_bridge (for the rolling window)

        # remove last available GDP THIS STEP IS VERY IMPORTANT AND HAS TO BE DONE BEFORE DATA IS FED INTO MODEL TO PREVENT LEAKAGE 
        # in that case I need to remove gdp_growth too becos this is what im predicting

        ## Store and remove last and second last non-NaN GDP values
        gdp_valid_indices = historical_data['GDP'].dropna().index
        last_gdp = second_last_gdp = None
        if len(gdp_valid_indices) >= 1:
            last_gdp = historical_data.at[gdp_valid_indices[-1], 'GDP']
            historical_data.at[gdp_valid_indices[-1], 'GDP'] = float('nan')
        if len(gdp_valid_indices) >= 2:
            second_last_gdp = historical_data.at[gdp_valid_indices[-2], 'GDP']
            historical_data.at[gdp_valid_indices[-2], 'GDP'] = float('nan')

        #repeat above for tree
        gdp_valid_indices_tree = historical_data_tree['GDP'].dropna().index
        last_gdp_tree = second_last_gdp_tree = None
        if len(gdp_valid_indices_tree) >= 1:
            last_gdp_tree = historical_data_tree.at[gdp_valid_indices_tree[-1], 'GDP']
            historical_data_tree.at[gdp_valid_indices_tree[-1], 'GDP'] = float('nan')
        if len(gdp_valid_indices_tree) >= 2:
            second_last_gdp_tree = historical_data_tree.at[gdp_valid_indices_tree[-2], 'GDP']
            historical_data_tree.at[gdp_valid_indices_tree[-2], 'GDP'] = float('nan')

        actual_gdp = last_gdp

        ## Store and remove last and second last non-NaN GDP_Growth values
        gdp_growth_valid_indices = historical_data['gdp_growth'].dropna().index
        last_gdp_growth = second_last_gdp_growth = None
        if len(gdp_growth_valid_indices) >= 1:
            last_gdp_growth = historical_data.at[gdp_growth_valid_indices[-1], 'gdp_growth']
            historical_data.at[gdp_growth_valid_indices[-1], 'gdp_growth'] = float('nan')
        if len(gdp_growth_valid_indices) >= 2:
            second_last_gdp_growth = historical_data.at[gdp_growth_valid_indices[-2], 'gdp_growth']
            historical_data.at[gdp_growth_valid_indices[-2], 'gdp_growth'] = float('nan')

        #repeat above for tree
        gdp_growth_valid_indices_tree = historical_data_tree['gdp_growth'].dropna().index
        last_gdp_growth_tree = second_last_gdp_growth_tree = None
        if len(gdp_growth_valid_indices_tree) >= 1:
            last_gdp_growth = historical_data_tree.at[gdp_growth_valid_indices_tree[-1], 'gdp_growth']
            historical_data_tree.at[gdp_growth_valid_indices_tree[-1], 'gdp_growth'] = float('nan')
        if len(gdp_growth_valid_indices_tree) >= 2:
            second_last_gdp_growth = historical_data_tree.at[gdp_growth_valid_indices_tree[-2], 'gdp_growth']
            historical_data_tree.at[gdp_growth_valid_indices_tree[-2], 'gdp_growth'] = float('nan')

        #print("prediction df", historical_data.tail(13))

        #### FORECAST FROM  ADL BRIDGE ####

        model_adl_output = model_ADL_bridge(historical_data)  # Get the model output DataFrame
        #print("model_adl_output", model_adl_output)

        #### FORECAST FROM RF BRIDGE ####
        model_rf_bridge_output = train_and_nowcast_rf(historical_data_tree)  # Get the model output DataFrame
        #print("model_adl_output", model_adl_output)


        
        # Determine the forecast horizon (using a heuristic based on the month)
        month_of_forecast = date.month  # This will help decide which forecast to use

        if month_of_forecast in [1, 4, 7, 10]:  # January, April, July, October -> m1, m4
            model_adl_m1 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m1 forecast
            model_adl_m4 = model_adl_output.iloc[2]['Nowcasted_GDP']  # m4 forecast

            model_rf_bridge_m1 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP']  # m1 forecast
            model_rf_bridge_m4 = model_rf_bridge_output.iloc[2]['Nowcasted_GDP']  # m4 forecast 

            print("m1 revision")
        elif month_of_forecast in [2, 5, 8, 11]:  # February, May, August, November -> m2, m5
            model_adl_m2 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m2 forecast
            model_adl_m5 = model_adl_output.iloc[2]['Nowcasted_GDP']  # m5 forecast

            model_rf_bridge_m2 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP']  # m2 forecast
            model_rf_bridge_m5 = model_rf_bridge_output.iloc[2]['Nowcasted_GDP']  # m5 forecast

            print("m2 revision")
        else:  # March, June, September, December -> m3, m6
            model_adl_m3 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m3 forecast
            model_adl_m6 = model_adl_output.iloc[2]['Nowcasted_GDP']  # m6 forecast
        
            model_rf_bridge_m3 = model_rf_bridge_output.iloc[1]['Nowcasted_GDP']  # m3 forecast
            model_rf_bridge_m6 = model_rf_bridge_output.iloc[2]['Nowcasted_GDP']  # m6 forecast
        
            print("m3 revision")
        
        #### FORECAST FROM  AR BENCHMARK ####
        # AR Forecast — only run model_AR on the first month of each quarter
        if month_of_forecast in [1, 4, 7, 10]:
            model_ar_output = model_AR(historical_data)
            model_ar_h1 = model_ar_output.iloc[1]['Nowcasted_GDP']
            model_ar_h2 = model_ar_output.iloc[2]['Nowcasted_GDP']

            # insert forecast from RF BENCHMARK
            #print("df fed into RF model", historical_data_tree.tail(15))
            model_RF_output = model_rf(historical_data_tree)
            model_RF_h1 = model_RF_output.iloc[1]['Nowcasted_GDP']
            model_RF_h2 = model_RF_output.iloc[2]['Nowcasted_GDP']


        else:
            model_ar_h1 = None
            model_ar_h2 = None
            model_RF_h1 = None
            model_RF_h2 = None

        
        results = pd.concat([results, pd.DataFrame({
            'date': [date],  
            'actual gdp': [actual_gdp],
            'model_AR h1': [model_ar_h1],
            'model_AR h2': [model_ar_h2],
            'model_ADL_bridge m1': [model_adl_m1 if month_of_forecast in [1, 4, 7, 10] else None],
            'model_ADL_bridge m2': [model_adl_m2 if month_of_forecast in [2, 5, 8, 11] else None],
            'model_ADL_bridge m3': [model_adl_m3 if month_of_forecast in [3, 6, 9, 12] else None],
            'model_ADL_bridge m4': [model_adl_m4 if month_of_forecast in [1, 4, 7, 10] else None],
            'model_ADL_bridge m5': [model_adl_m5 if month_of_forecast in [2, 5, 8, 11] else None],
            'model_ADL_bridge m6': [model_adl_m6 if month_of_forecast in [3, 6, 9, 12] else None],
            'model_RF h1': [model_RF_h1],
            'model_RF h2': [model_RF_h2],
            'model_RF_bridge m1': [model_rf_bridge_m1 if month_of_forecast in [1, 4, 7, 10] else None],
            'model_RF_bridge m2': [model_rf_bridge_m2 if month_of_forecast in [2, 5, 8, 11] else None],
            'model_RF_bridge m3': [model_rf_bridge_m3 if month_of_forecast in [3, 6, 9, 12] else None],
            'model_RF_bridge m4': [model_rf_bridge_m4 if month_of_forecast in [1, 4, 7, 10] else None],
            'model_RF_bridge m5': [model_rf_bridge_m5 if month_of_forecast in [2, 5, 8, 11] else None],
            'model_RF_bridge m6': [model_rf_bridge_m6 if month_of_forecast in [3, 6, 9, 12] else None],
        })], ignore_index=True)

    results['model_ADL_bridge m4'] = results['model_ADL_bridge m4'].shift(3)
    results['model_ADL_bridge m5'] = results['model_ADL_bridge m5'].shift(3)
    results['model_ADL_bridge m6'] = results['model_ADL_bridge m6'].shift(3)

    results['model_AR h2'] = results['model_AR h2'].shift(3)

    results['model_RF h2'] = results['model_RF h2'].shift(3)

    results['model_RF_bridge m4'] = results['model_RF_bridge m4'].shift(3)
    results['model_RF_bridge m5'] = results['model_RF_bridge m5'].shift(3)
    results['model_RF_bridge m6'] = results['model_RF_bridge m6'].shift(3)
    
    return results

def calculate_row_error(df):
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
        result[f'Error_{col}'] = result[col] - result['actual gdp']

    return result

def calculate_rmsfe(df):
    # Ensure date column is in datetime format
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

    # Columns to calculate RMSFE for
    forecast_cols = model_and_horizon

    # Compute RMSFE for each forecast column
    for col in forecast_cols:
        # Drop rows where either forecast or actual is NaN
        valid_rows = result[[col, 'actual gdp']].dropna()
        error = valid_rows[col] - valid_rows['actual gdp']
        rmsfe = np.sqrt((error ** 2).mean())
        result[f'RMSFE_{col}'] = rmsfe

    return result


file_path1 = "../Data/bridge_df.csv"
file_path2 = "../Data/tree_df.csv"
df = pd.read_csv(file_path1)
df_nonlinear = pd.read_csv(file_path2)
res = generate_oos_forecast(df, df_nonlinear)
row_error_df = calculate_row_error(res)
rmsfe_df = calculate_rmsfe(res)
print(res)
print(row_error_df)
print(rmsfe_df)

row_error_df.to_csv("../Data/row_error.csv", index=False)
rmsfe_df.to_csv("../Data/rmsfe.csv", index=False)
