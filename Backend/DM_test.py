import numpy as np
import pandas as pd 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from model_AR import model_AR
from model_ADL_bridge import model_ADL_bridge
#from diebold_mariano_test import diebold_mariano_test 
pd.set_option('display.float_format', '{:.2f}'.format)

def get_missing_months(df, date_column="date"):
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Get the latest date in the dataset
    latest_date = df[date_column].max()
    
    # Get the month number (1 to 12)
    latest_month = latest_date.month

    #print("latest_month", latest_month)
    
    # Calculate how many months remain to complete the current quarter
    months_to_complete_quarter = (3 - ((latest_month ) % 3)) % 3

    # Total months to add: remaining months of current quarter + 2 quarters (6 months)
    total_months_to_add = months_to_complete_quarter + 3

    return total_months_to_add

def add_missing_months(df, date_column="date"):
    # Calculate how many months are missing
    num_extra_rows = get_missing_months(df, date_column)

    if num_extra_rows > 0:
        # Get the latest date in the dataset
        latest_date = pd.to_datetime(df[date_column].max())

        # Generate new dates starting from the next month after the latest date
        new_dates = pd.date_range(
            start=latest_date + pd.DateOffset(months=1),
            periods=num_extra_rows,
            freq='MS'
        )

        # Create new rows with NaN values except the date column
        new_rows = pd.DataFrame({date_column: new_dates})

        # Append to the original dataframe
        df = pd.concat([df, new_rows], ignore_index=True)

    # Sort and reset index
    df = df.sort_values(by=date_column).reset_index(drop=True)
    return df



def rolling_window_benchmark_evaluation(df, window_size=(12*20)):
    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
    df = df.sort_values(by='date', ascending=False)
    
    # Find the index of the last non-NA GDP value
    last_valid_gdp_index = df['GDP'].first_valid_index()
    
    # Get the date two rows above the last available GDP
    start_row_index = df.index.get_loc(last_valid_gdp_index) - 2  # Two rows above the last valid GDP to complete the quarter

    
    # Trim the DataFrame to include rows starting from this point onward
    df_trimmed = df.iloc[start_row_index:].sort_values(by='date', ascending=True).reset_index(drop=True)

    #print("df_trimmed",df_trimmed)
    
    # Initialize results dataframe
    results = pd.DataFrame(columns=[
        'date', 'actual gdp', 
        'model_AR h1', 'model_AR h2',  # Placeholder for AR model forecasts (to be filled later)
        'model_ADL_bridge m1', 'model_ADL_bridge m2', 'model_ADL_bridge m3',
        'model_ADL_bridge m4', 'model_ADL_bridge m5', 'model_ADL_bridge m6'
    ])

    #df_sorted = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    
    # Loop through the trimmed dataframe using a fixed rolling window size
    remove_covid = 12*5
    for start_index in range(len(df_trimmed) - window_size - remove_covid):
        print("Window:", df_trimmed.iloc[start_index]['date'].date(), "to", df_trimmed.iloc[start_index + window_size - 1]['date'].date())
        
        # Get the historical data window of size `window_size`
        end_index = start_index + window_size
        historical_data = df_trimmed.iloc[start_index:end_index]

        historical_data = historical_data.sort_values(by='date', ascending=False).reset_index(drop=True)

        #print("before bfill", historical_data.head(10))
        #print("before shift up", historical_data.head(10))

        # bfill gdp here
        #historical_data['GDP'] = historical_data['GDP'].fillna(method='bfill')
        #historical_data['GDP'] = historical_data['GDP'].shift(-1)

        #print("after shift up", historical_data.head(10))

        historical_data = add_missing_months(historical_data, date_column="date")

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

        # Store and remove last and second last non-NaN GDP values
        gdp_valid_indices = historical_data['GDP'].dropna().index
        last_gdp = second_last_gdp = None
        if len(gdp_valid_indices) >= 1:
            last_gdp = historical_data.at[gdp_valid_indices[-1], 'GDP']
            historical_data.at[gdp_valid_indices[-1], 'GDP'] = float('nan')
        if len(gdp_valid_indices) >= 2:
            second_last_gdp = historical_data.at[gdp_valid_indices[-2], 'GDP']
            historical_data.at[gdp_valid_indices[-2], 'GDP'] = float('nan')

        actual_gdp = last_gdp

        # Store and remove last and second last non-NaN GDP_Growth values
        gdp_growth_valid_indices = historical_data['gdp_growth'].dropna().index
        last_gdp_growth = second_last_gdp_growth = None
        if len(gdp_growth_valid_indices) >= 1:
            last_gdp_growth = historical_data.at[gdp_growth_valid_indices[-1], 'gdp_growth']
            historical_data.at[gdp_growth_valid_indices[-1], 'gdp_growth'] = float('nan')
        if len(gdp_growth_valid_indices) >= 2:
            second_last_gdp_growth = historical_data.at[gdp_growth_valid_indices[-2], 'gdp_growth']
            historical_data.at[gdp_growth_valid_indices[-2], 'gdp_growth'] = float('nan')

        #print("prediction df", historical_data.tail(13))

        model_adl_output = model_ADL_bridge(historical_data)  # Get the model output DataFrame
        print("model_adl_output", model_adl_output)

        # insert forecast from RF BRIDGE
        
        # Determine the forecast horizon (using a heuristic based on the month)
        month_of_forecast = date.month  # This will help decide which forecast to use

        if month_of_forecast in [1, 4, 7, 10]:  # January, April, July, October -> m1, m4
            model_adl_m1 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m1 forecast
            model_adl_m4 = model_adl_output.iloc[2]['Nowcasted_GDP']  # m4 forecast
            print("m1 revision")
        elif month_of_forecast in [2, 5, 8, 11]:  # February, May, August, November -> m2, m5
            model_adl_m2 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m2 forecast
            model_adl_m5 = model_adl_output.iloc[2]['Nowcasted_GDP']  # m5 forecast
            print("m2 revision")
        else:  # March, June, September, December -> m3, m6
            model_adl_m3 = model_adl_output.iloc[1]['Nowcasted_GDP']  # m3 forecast
            model_adl_m6 = model_adl_output.iloc[2]['Nowcasted_GDP']  # m6 forecast
            print("m3 revision")
        
        # AR Forecast — only run model_AR on the first month of each quarter
        if month_of_forecast in [1, 4, 7, 10]:
            model_ar_output = model_AR(historical_data)
            model_ar_h1 = model_ar_output.iloc[1]['Nowcasted_GDP']
            model_ar_h2 = model_ar_output.iloc[2]['Nowcasted_GDP']

            # insert forecast from RF BENCHMARK
            # model_RF_benchmark = model_RF_benchmark(historical_data)
            #
            #

        else:
            model_ar_h1 = None
            model_ar_h2 = None
        
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
        })], ignore_index=True)

    results['model_ADL_bridge m4'] = results['model_ADL_bridge m4'].shift(3)
    results['model_ADL_bridge m5'] = results['model_ADL_bridge m5'].shift(3)
    results['model_ADL_bridge m6'] = results['model_ADL_bridge m6'].shift(3)

    results['model_AR h2'] = results['model_AR h2'].shift(3)
    
    return results

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
    forecast_cols = [
        'model_AR h1', 'model_AR h2',
        'model_ADL_bridge m1', 'model_ADL_bridge m2', 'model_ADL_bridge m3',
        'model_ADL_bridge m4', 'model_ADL_bridge m5', 'model_ADL_bridge m6'
    ]

    # Compute RMSFE for each forecast column
    for col in forecast_cols:
        # Drop rows where either forecast or actual is NaN
        valid_rows = result[[col, 'actual gdp']].dropna()
        error = valid_rows[col] - valid_rows['actual gdp']
        rmsfe = np.sqrt((error ** 2).mean())
        result[f'RMSFE_{col}'] = rmsfe

    return result

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

    forecast_cols = [
        'model_AR h1', 'model_AR h2',
        'model_ADL_bridge m1', 'model_ADL_bridge m2', 'model_ADL_bridge m3',
        'model_ADL_bridge m4', 'model_ADL_bridge m5', 'model_ADL_bridge m6'
    ]

    # Calculate row-by-row forecast errors (prediction - actual)
    for col in forecast_cols:
        result[f'Error_{col}'] = result[col] - result['actual gdp']

    return result


file_path = "../Data/bridge_df.csv"
df = pd.read_csv(file_path)
res = rolling_window_benchmark_evaluation(df)
print(res)
print(calculate_row_error(res))
print(calculate_rmsfe(res))


"""if __name__ == "__main__":
    file_path = "../Data/lasso_indicators.csv"
    df = pd.read_csv(file_path)

    # Run rolling evaluation
    forecast_results = rolling_window_benchmark_evaluation(df)

    # Perform Diebold-Mariano test
    dm_results = diebold_mariano_test(
        actual_values=forecast_results["Actual_GDP"],
        bridge_forecast=forecast_results["Bridge_Forecast"],
        ar_forecast=forecast_results["AR_Forecast"]
    )

    print("\n Forecast Results:")
    print(forecast_results)

    print("\n Diebold-Mariano Test Results:")
    print(dm_results)"""
