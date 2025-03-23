import numpy as np
import pandas as pd 
from model_AR import model_AR
from model_bridge import model_bridge

#from diebold_mariano_test import diebold_mariano_test 

def rolling_window_benchmark_evaluation(df, target_variable="GDP", 
                                        window_size=350, start_date="1996-01", end_date="2024-09"):
    """
    Methodology:

    Takes in df: GDP | GDP Diff | NFP L1 | Retail Sales L1 | Housing Starts L1 | Junk Bond Spreads L1

    Total number of quarters available: 120

    Define a rolling window size: 80


    First iteration: 
    1. Create curr_window (1995Q1 to 2014 Q4)
    2. Generate 2015Q1 forecast using model_AR(curr_window) and model_bridge(curr_window) benchmark vs bridge forecast 
    3. Update in forecast_results_df: date | GDP | Benchmark Forecast | Bridge Forecast
    4. Move window until last available GDP (2024Q3) 

    5. forecast_results_df can be used to calculate OOS RMSFE and DM test

    # to clarify with Prof: methodology above does a forecast of 1 step from the horizon so it can be used to calculate OOS RMSFE of 1 step from horizon right?

    # maybe after this is done, make another function to calculate OOS RMSFE 2 steps from horizon.

    # to make fan chart: forecast interval which RMSFE (what step from horizon should i be using?) 
    # eg if latest gdp avail is 2024Q3 and the model forecasts 2024Q4 to 2025Q2 (1 quarter ahead from present day:2025Q1)
    # then the forecast will be 3 steps ahead of the horizon. so i will need to create a function to calculate also the 2 step and 3 step ahead forecast? 

    # small detail im confused about: while the above talks about 1/2/3 step ahead forecast relative to latest gdp release, 
    # do we just ignore the forecast horizon of the AR(P) for each indicators? since there will be some error when making those predictions too 

    

    Uses a fixed-length rolling window approach to evaluate the forecasting performance 
    of model_AR and model_bridge.

    Args:
        df (pd.DataFrame): DataFrame containing macroeconomic indicators.
        target_variable (str): Column name of the GDP variable.
        window_size (int): Number of past quarters used for training in each rolling window.
        start_date (str): The earliest quarter for making predictions (YYYY-MM format).
        end_date (str): The latest quarter for making predictions (YYYY-MM format).

    Returns:
        pd.DataFrame: A DataFrame storing actual GDP values and model forecasts for each quarter.
    """

    # date in desc order
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date", ascending=False).reset_index(drop=True)

    forecast_results = []

    # Loop through each quarter where we have enough data for training (moving backwards)
    for forecast_idx in range(len(df) - window_size - 2, -1, -3):  
        # Define the training window range (moving backwards)
        training_end_idx = forecast_idx + window_size  # The most recent data point in window
        training_start_idx = forecast_idx  # The oldest data point in window
        training_data = df.loc[df.index[training_start_idx : training_end_idx]].copy()

        # Identify the quarter we are predicting (ensure correct format)
        target_quarter = df.iloc[forecast_idx]["date"].strftime("%Y-%m")  # Fix formatting
        print(f"Training from {training_start_idx} to {training_end_idx} → Predicting: {target_quarter}")

        # Run both models using the training data
        bridge_forecast_df = model_bridge(training_data)
        ar_forecast_df = model_AR(training_data)

        # Ensure forecasted DataFrames have their date as strings for matching
        bridge_forecast_df["date"] = bridge_forecast_df["date"].astype(str)
        ar_forecast_df["date"] = ar_forecast_df["date"].astype(str)

        # Extract forecasts for the specific quarter
        bridge_forecast = bridge_forecast_df.loc[
            bridge_forecast_df["date"] == target_quarter, "Nowcasted_GDP"
        ].values

        ar_forecast = ar_forecast_df.loc[
            ar_forecast_df["date"] == target_quarter, "Nowcasted_GDP"
        ].values

        # 🔹 Ensure `df["date"]` is in correct format for matching
        df["date_str"] = df["date"].dt.strftime("%Y-%m")  # Convert to YYYY-MM format

        # Retrieve the actual GDP value for the target quarter
        actual_gdp = df.loc[df["date_str"] == target_quarter, target_variable].values

        # Debugging print statements
        print(f"Target Quarter: {target_quarter}")
        print(f"Bridge Forecast Found: {bridge_forecast}")
        print(f"AR Forecast Found: {ar_forecast}")
        print(f"Actual GDP Found: {actual_gdp}")

        # Store the results if forecasts and actual values exist
        if bridge_forecast.size > 0 and ar_forecast.size > 0 and actual_gdp.size > 0:
            forecast_results.append({
                "date": target_quarter,
                "Actual_GDP": actual_gdp[0],
                "Bridge_Forecast": bridge_forecast[0],
                "AR_Forecast": ar_forecast[0]
            })

    # Convert results to DataFrame
    forecast_results_df = pd.DataFrame(forecast_results)
    
    return forecast_results_df


file_path = "../Data/lasso_indicators.csv"
df = pd.read_csv(file_path)
print(rolling_window_benchmark_evaluation(df))


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
