"""
This file contains the `model_RF_bridge` function, which implements a Random Forest model pipeline for forecasting GDP growth and generating nowcasts. 
The process involves:
1. Forecasting macroeconomic indicators using an AR(p) model.
2. Aggregating and lagging the features.
3. Training a Random Forest model using selected features.
4. Generating nowcasts for missing GDP values based on the trained model.

The pipeline follows these steps:
- Feature selection via RFECV.
- Training the Random Forest model on preprocessed data.
- Forecasting GDP growth and converting it into GDP levels.
- Optionally saving the trained model.

Functions:
- `get_selected_features`: Selects important features using RFECV with time series split.
- `model_RF_bridge`: Full Random Forest pipeline for forecasting, aggregation, lagging, training, and nowcasting GDP.
"""


import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFECV
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from data_processing import aggregate_indicators, create_lag_features
from forecast_bridge_indicators import forecast_indicators

def get_selected_features(X, Y, n_splits=5):
    """
    Runs RFECV using time series split on preprocessed and lagged X, Y data.
    Only the training set is used for feature selection.

    # Kelli pls specify here that this function is used only when u wna do feature selection. Once features are chosen, this part isnt run again to save computation time.

    Args:
        X (pd.DataFrame): Feature matrix containing the preprocessed and lagged macroeconomic data.
        Y (pd.Series): Target vector containing GDP growth values.
        n_splits (int): The number of splits for the time series cross-validation. Default is 5.

    Returns:
        list: A list of selected feature names after running RFECV.
    """
    #Fit RFECV on training data 
    rf_model = RandomForestRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rfecv = RFECV(
        estimator=rf_model,
        step=1,
        cv=tscv,
        scoring='neg_mean_squared_error',
        min_features_to_select= 7,
        n_jobs=-1
    )

    #Fitting RFECV on training set 
    rfecv.fit(X, Y)

    selected_features = list(X.columns[rfecv.support_])

    # Desired simplified output
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"These are your selected features: {selected_features}")


    return selected_features

def tune_random_forest(X, Y, selected_features, model_path='../Backend/tuned_RF_bridge_model.joblib'):
    """
    Tunes a RandomForestRegressor using time series-aware GridSearchCV and saves the model.

    Args:
        X (pd.DataFrame): Feature matrix.
        Y (pd.Series): Target values (GDP growth).
        model_path (str): Path to save the tuned model.

    Returns:
        best_model (RandomForestRegressor): The tuned best model.
    """

    X = X[selected_features]

    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading tuned model from {model_path}")
        best_model = joblib.load(model_path)
        return best_model

    # Grid search tuning
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': [400],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [4, 6, 10],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, Y)

    best_model = grid_search.best_estimator_

    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    return best_model

def model_RF_bridge(df, model_path='../Backend/tuned_RF_bridge_model.joblib'):
    """
    Loads a pre-tuned RF model and runs nowcasting without re-tuning.

    Args:
        df (pd.DataFrame): Original raw DataFrame.
        model_path (str): Path to the saved RF model.

    Returns:
        pd.DataFrame: DataFrame with nowcasted GDP growth and GDP values.
    """

    selected_features = ['Unemployment', 'Capacity_Utilization', 'Nonfarm_Payrolls', 
                         'New_Orders_Durable_Goods', 'Housing_Starts_lag2', 
                         'Nonfarm_Payrolls_lag1', 'New_Home_Sales_lag1']

    # Your data preprocessing steps
    df = df.sort_values(by='date', ascending=True)
    df_model = forecast_indicators(df)
    df_model = aggregate_indicators(df_model)
    df_model = create_lag_features(df_model, exclude_columns=["date", "GDP"], max_lag=4)
    df_model = df_model.sort_values(by='date', ascending=True)

    # Clean data
    df_model = df_model[~((df_model["date"].dt.year == 1995) & (df_model.isna().any(axis=1)))]
    df_model = df_model[~((df_model["date"] == "1996-03-31") & (df_model.isna().any(axis=1)))]

    df_model = df_model[['date', 'GDP', 'gdp_growth'] + selected_features]

    # Load pre-tuned RF model
    if os.path.exists(model_path):
        final_rf_model = joblib.load(model_path)
        #print(f"Loaded tuned model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}. Run tuning first.")

    # Nowcasting steps
    recent_quarters_mask = df_model.index >= df_model.index[-4]
    nowcast_mask = (df_model["GDP"].isna()) | ((df_model["gdp_growth"].isna()) & recent_quarters_mask)
    forecast_df = df_model[nowcast_mask].copy()

    if not forecast_df.empty:
        last_known_idx = df_model[df_model["GDP"].notna()].index[-1]
        last_actual_gdp = df_model.loc[last_known_idx, "GDP"]

        X_forecast = forecast_df[selected_features]
        predicted_growth = final_rf_model.predict(X_forecast)

        predicted_gdp = []
        for growth in predicted_growth:
            next_gdp = last_actual_gdp * np.exp(growth / 400)
            predicted_gdp.append(next_gdp)
            last_actual_gdp = next_gdp

        nowcast_results = forecast_df[["date"]].copy()
        nowcast_results["Nowcasted_GDP_Growth"] = predicted_growth
        nowcast_results["Nowcasted_GDP"] = predicted_gdp
    else:
        #print("No rows to nowcast.")
        nowcast_results = pd.DataFrame()

    return nowcast_results



if __name__ == "__main__":
    # #Actual data
    file_path = "../Data/tree_df.csv"
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # #Sort date in ascending order
    df = df.sort_values(by='date', ascending= True)

    # #Aggregate Data
    df1 = aggregate_indicators(df)

    # #Create lagged features
    exclude_columns = ["date", "GDP", "dummy"]
    df1 = create_lag_features(df1, exclude_columns, 4) 

    # #Sort date again 
    df1 = df1.sort_values(by='date', ascending= True)

    # #Drop NaN values created by lagging 
    df1.dropna(inplace = True)

    # #Drop 'Date' before defining X 
    
    X = df1.drop(columns=["gdp_growth", "GDP", "date"])  
    Y = df1['gdp_growth']

    selected_features = [
        'Unemployment',
        'Capacity_Utilization',
        'Nonfarm_Payrolls',
        'New_Orders_Durable_Goods',
        'Housing_Starts_lag2',
        'Nonfarm_Payrolls_lag1',
        'New_Home_Sales_lag1'
    ]

    model_path = '../Backend/tuned_RF_bridge_model.joblib'

    
    if not os.path.exists(model_path):
        _ = tune_random_forest(X, Y, selected_features, model_path=model_path)

    results = model_RF_bridge(df)
    print(results)
