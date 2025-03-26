import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
#pd.set_option("display.max_columns", None)
pd.reset_option("display.max_columns")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from data_processing import aggregate_indicators, create_lag_features
from model_ADL_bridge import forecast_indicators, record_months_to_forecast #AR(p) model 

file_path = "../Data/tree_df.csv"
df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

#Sort date in ascending order
df = df.sort_values(by='date', ascending= True)
#df.drop(columns = ["dummy"], inplace= True)

#Aggregate Data
df1 = aggregate_indicators(df)

#Create lagged features
exclude_columns = ["date", "GDP", "dummy"]
df1 = create_lag_features(df1, exclude_columns, 4) 
print("column names", df1.columns)
#Sort date again 
df1 = df1.sort_values(by='date', ascending= True)

#drop gdp_growth_lag1
df1 = df1.drop(columns = ['gdp_growth_lag1'])

#Drop NaN values created by lagging 
df1.dropna(inplace = True)

#Drop 'Date' before defining X 
X = df1.drop(columns=["gdp_growth", "GDP", "date"])
Y = df1['gdp_growth']

# Train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Train Random Forest model for feature selection
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selector.fit(X_train, Y_train)

# # ✅ Step 6: Extract feature importance scores
feature_importances = pd.Series(rf_selector.feature_importances_, index=X.columns)

# # ✅ Step 7: Select the top 10 most important features
top_features = feature_importances.nlargest(10).index
print(top_features)


#  #Building the RF model 
df = df.sort_values(by='date', ascending= True)

# #forecast with AR(p)
df_model = forecast_indicators(df)

#Aggregate to quarterly 
df_model = aggregate_indicators(df_model)
print("check the df hehehehe", df_model)

#lag the df 
exclude_columns = ["date", "GDP"]
max_lag = 4
df_model = create_lag_features(df_model, exclude_columns, max_lag) 

#Sort Lags Again 
df_model = df_model.sort_values(by='date', ascending=True)
print("i just want to check my lags", df_model)

#Updated df_model with selected features 
df_model = df_model[['date', 'GDP', 'gdp_growth'] + list(top_features)]

# #Drop only rows where date is in 1995 & Contains NaN 
df_model = df_model[~((df_model["date"].dt.year == 1995) & (df_model.isna().any(axis=1)))]

# Remove the row for 1996-03-31 if it contains any missing values
df_model = df_model[~((df_model["date"] == "1996-03-31") & (df_model.isna().any(axis=1)))]

#Subset the data to include all rows up to last known GDP 
#print("here is your df_model",df_model)
                                                       
# Train Random Forest Model using only the selected features 
X_final = df_model.drop(columns= ["GDP", "gdp_growth", "date"]) #Features
Y_final = df_model["gdp_growth"] #Target 

# #Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.2, shuffle=False)

# #Predict GDP and Evaluate model performance
final_rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
final_rf_model.fit(X_train, Y_train)

# # Saving the model
model_eval_path = "../Model/eval_rf_model.pkl"
os.makedirs("../Model", exist_ok=True)
joblib.dump(final_rf_model, model_eval_path)
print(f"Saved evaluation model (80% train) to: {model_eval_path}")

#check if NaN or 0 

def nowcast_recursive_rf(df_model, final_rf_model):
    df_model = df_model.copy()
    
    # Split the data into known and unknown GDP parts
    last_known_idx = df_model[df_model["GDP"].notna()].index[-1]
    known_df = df_model.loc[:last_known_idx].copy()
    forecast_df = df_model.loc[last_known_idx + 1:].copy().reset_index(drop=True)

    # Initialize containers
    last_actual_gdp = known_df.iloc[-1]["GDP"]
    predicted_growth_list = []
    predicted_gdp_list = []
    
    for i in range(len(forecast_df)):
        row = forecast_df.iloc[i].copy()

        # If we're beyond the first step, use the previously predicted gdp_growth as lagged input
        if i > 0:
            for lag in range(1, 5):  # Assuming 4 lags exist
                col_name = f"gdp_growth_lag{lag}"
                if col_name in forecast_df.columns:
                    if i - lag >= 0:
                        row[col_name] = predicted_growth_list[i - lag]
        
        # Drop target columns and predict
        input_features = row.drop(labels=["date", "GDP", "gdp_growth"]).values.reshape(1, -1)
        predicted_growth = final_rf_model.predict(input_features)[0]
        predicted_growth_list.append(predicted_growth)
        
        # Convert predicted growth to GDP level
        next_gdp = last_actual_gdp * (1 + predicted_growth / 100)
        predicted_gdp_list.append(next_gdp)
    
        # Update last known GDP
        last_actual_gdp = next_gdp

    # Output results
    results_df = forecast_df[["date"]].copy()
    results_df["Nowcasted_GDP_Growth"] = predicted_growth_list
    results_df["Nowcasted_GDP"] = predicted_gdp_list

    return results_df

# Nowcast the GDP using the trained Random Forest model
nowcast_results = nowcast_recursive_rf(df_model, final_rf_model)

# View the nowcasted GDP and growth
print(nowcast_results)
