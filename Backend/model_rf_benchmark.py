import pandas as pd
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
pd.set_option("display.max_columns", None)
#pd.reset_option("display.max_columns")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from data_processing import aggregate_indicators, create_lag_features
from model_ADL_bridge import forecast_indicators, record_months_to_forecast #AR(p) model 
file_path = "../Data/tree_df_test.csv"
df = pd.read_csv(file_path)


def model_rf_benchmark(df):

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.drop(columns=["dummy"], inplace=True) #drop dummy col

    #extract col names for later use
    cols_to_keep = ['date', 'GDP', "gdp_growth"]
    contemp_indicators = [col for col in df.columns if col not in cols_to_keep]

    #Sort date in ascending order
    df = df.sort_values(by='date', ascending=True)

    #Aggregate Data
    df_aggregated = aggregate_indicators(df) #gdp_growth is excluded
    exclude_columns = ["date", "GDP"] #df to exclude later when laggin indicators

    #create a df to store nowcasted values
    df_to_pred = df_aggregated[df_aggregated["GDP"].isnull() & (df_aggregated.index >= df_aggregated.index[-4])]
    df_to_pred = df_to_pred[["date", "gdp_growth", "GDP"]]

    #lag variables
    df_lagged = create_lag_features(df_aggregated, exclude_columns, 6) # 6 lags of each indicator 

    #drop contemporary indicators and gdp_growth_lag1
    df_lagged = df_lagged.drop(columns=contemp_indicators + ["gdp_growth_lag1"])
    df_lagged = df_lagged.dropna(subset=["GDP"])

    #drop lags of each indicator depending on how many steps model is predicting
    def prepare_model_data(df, drop_lags, drop_growth_lag):
        df_model = df.drop(columns=[col for col in df.columns if any(col.endswith(f'_lag{i}') for i in drop_lags)])
        if drop_growth_lag:
            df_model = df_model.drop(columns=drop_growth_lag)
        #Sort date again 
        df_model = df_model.sort_values(by='date', ascending=True)
        #Drop NaN values created by lagging 
        df_model = df_model.dropna()
        #Drop 'Date' before defining X 
        X = df_model.drop(columns=["gdp_growth", "GDP", "date"])
        y = df_model['gdp_growth']
        return train_test_split(X, y, test_size=0.2, shuffle=False)

    #Model 1 (1 step forecast, so take lag1 to lag 4)
    X_train_1, X_test_1, Y_train_1, Y_test_1 = prepare_model_data(df_lagged, [5, 6], None)
    rf_model_1 = RandomForestRegressor(random_state=42).fit(X_train_1, Y_train_1)

    ##Model 2 (2 step forecast, so take lag2 to lag5)
    X_train_2, X_test_2, Y_train_2, Y_test_2 = prepare_model_data(df_lagged, [1, 6], ["gdp_growth_lag2"])
    rf_model_2 = RandomForestRegressor(random_state=42).fit(X_train_2, Y_train_2)

    ##Model 3 (3 step forecast, so take lag3 to lag6)
    X_train_3, X_test_3, Y_train_3, Y_test_3 = prepare_model_data(df_lagged, [1, 2], ["gdp_growth_lag3"])
    rf_model_3 = RandomForestRegressor(random_state=42).fit(X_train_3, Y_train_3)

    #filter out latest row with available data
    df_latest_date = df_lagged.loc[df_lagged['date'].idxmax():]
    df_latest_indicators = df_latest_date.drop(columns=["date", "GDP", "gdp_growth"])

    #ensure that df_pred is sorted in ascending order
    df_to_pred = df_to_pred.reset_index(drop=True).sort_values(by="date", ascending=True)

    #for loop to fill in gdp_growth
    for index, row in df_to_pred.iterrows():
        # Access values in each row by column name
        date = row['date']
        gdp_growth = row['gdp_growth']
        GDP = row['GDP']
        if index == 0:
            #use model 1
            df_indicators = df_latest_indicators.drop(columns=[col for col in df_latest_indicators.columns if any(col.endswith(f'_lag{i}') for i in [5, 6])])
            gdp_growth = rf_model_1.predict(df_indicators)

        elif index == 1:
            #use model 2
            df_indicators = df_latest_indicators.drop(columns=[col for col in df_latest_indicators.columns if any(col.endswith(f'_lag{i}') for i in [1, 6])])
            df_indicators = df_indicators.drop(columns=["gdp_growth_lag2"])
            gdp_growth = rf_model_2.predict(df_indicators)

        elif index == 2:
            #use model 3
            df_indicators = df_latest_indicators.drop(columns=[col for col in df_latest_indicators.columns if any(col.endswith(f'_lag{i}') for i in [1, 2])])
            df_indicators = df_indicators.drop(columns=["gdp_growth_lag3"])
            gdp_growth = rf_model_3.predict(df_indicators)

        df_to_pred.at[index, 'gdp_growth'] = gdp_growth

    #fill in first nowcasted gdp
    df_to_pred["Nowcasted_GDP"] = np.nan
    df_to_pred.at[0, "Nowcasted_GDP"] = df_latest_date["GDP"].values[0] * np.exp(df_to_pred.at[0, "gdp_growth"] / 400)

    #for loop to fill in rest of nowcasted GDP
    for i in range(1, len(df_to_pred)):
        # Calculate 'Nowcasted_GDP' for the current row
        df_to_pred.at[i, "Nowcasted_GDP"] = df_to_pred.at[i - 1, "Nowcasted_GDP"] * np.exp(df_to_pred.at[i, "gdp_growth"] / 400)

    df_to_pred.drop(columns=["GDP"], inplace=True)
    df_to_pred.rename(columns={"gdp_growth": "gdp_growth_forecast"}, inplace=True)
    df_to_pred = df_to_pred.head(3)

    return df_to_pred



"""import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
pd.set_option("display.max_columns", None)
#pd.reset_option("display.max_columns")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from data_processing import aggregate_indicators, create_lag_features
from model_ADL_bridge import forecast_indicators, record_months_to_forecast #AR(p) model 


#Load data 
file_path = "../Data/tree_df.csv"
df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df.drop(columns= ["dummy"], inplace=True) #drop dummy col
#extract col names for later use
cols_to_keep = ['date', 'GDP', "gdp_growth"]
contemp_indicators = [col for col in df.columns if col not in cols_to_keep]

#Sort date in ascending order
df = df.sort_values(by='date', ascending= True)



#Aggregate Data
df_aggregated = aggregate_indicators(df) #gdp_growth is excluded
df_final = aggregate_indicators(df) #create a copy of df for later
exclude_columns = ["date", "GDP"] #df to exclude later when laggin indicators

#create a df to store nowcasted values
df_to_pred = df_aggregated[df_aggregated["GDP"].isnull() & (df_aggregated.index >= df_aggregated.index[-4])]
df_to_pred = df_to_pred[["date", "gdp_growth", "GDP"]]


#lag variables
df_lagged = create_lag_features(df_aggregated, exclude_columns, 6) # 6 lags of each indicator 

#drop contemporary indicators and gdp_growth_lag1
df_lagged = df_lagged.drop(columns=contemp_indicators)
df_lagged = df_lagged.drop(columns=["gdp_growth_lag1"])
df_lagged = df_lagged.dropna(subset = ["GDP"])

#drop lags of each indicator depending on how many steps model is predicting
#Model 1 (1 step forecast, so take lag1 to lag 4)
df_model1 = df_lagged.drop(columns=[col for col in df_lagged.columns if any(col.endswith(f'_lag{i}') for i in [5,6])]) #drop cols of lag5,lag6

#Sort date again 
df_model1 = df_model1.sort_values(by='date', ascending= True)


#Drop NaN values created by lagging 
df_tree_model1 = df_model1.dropna()

#Drop 'Date' before defining X 
X_model1 = df_tree_model1.drop(columns=["gdp_growth", "GDP", "date"])
Y_model1 = df_tree_model1['gdp_growth']

# Train test split 
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_model1, Y_model1, test_size=0.2, shuffle=False)

# Train Random Forest model for feature selection
rf_model_1 = RandomForestRegressor(random_state=42)
rf_model_1.fit(X_train_1, Y_train_1)


##Model 2 (2 step forecast, so take lag2 to lag5)
df_model2 = df_lagged.drop(columns=[col for col in df_lagged.columns if any(col.endswith(f'_lag{i}') for i in [1, 6])]) #drop lag1, lag6
df_model2 = df_model2.drop(columns = ["gdp_growth_lag2"])

#Sort date again 
df_model2 = df_model2.sort_values(by='date', ascending= True)


#Drop NaN values created by lagging 
df_tree_model2 = df_model2.dropna()

#Drop 'Date' before defining X 
X_model2 = df_tree_model2.drop(columns=["gdp_growth", "GDP", "date"])
Y_model2 = df_tree_model2['gdp_growth']

# Train test split 
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X_model2, Y_model2, test_size=0.2, shuffle=False)

# Train Random Forest model for feature selection
rf_model_2 = RandomForestRegressor(random_state=42)
rf_model_2.fit(X_train_2, Y_train_2)

##Model 3 (3 step forecast, so take lag3 to lag6)
df_model3 = df_lagged.drop(columns=[col for col in df_lagged.columns if any(col.endswith(f'_lag{i}') for i in [1,2])]) #drop lag1, lag6
df_model3 = df_model3.drop(columns = "gdp_growth_lag3")

#Sort date again 
df_model3 = df_model3.sort_values(by='date', ascending= True)


#Drop NaN values created by lagging 
df_tree_model3 = df_model3.dropna()

#Drop 'Date' before defining X 
X_model3 = df_tree_model3.drop(columns=["gdp_growth", "GDP", "date"])
Y_model3 = df_tree_model3['gdp_growth']

# Train test split 
X_train_3, X_test_3, Y_train_3, Y_test_3 = train_test_split(X_model3, Y_model3, test_size=0.2, shuffle=False)

# Train Random Forest model for feature selection
rf_model_3 = RandomForestRegressor(random_state=42)
rf_model_3.fit(X_train_3, Y_train_3)

#filter out latest row with available data
df_latest_date = df_lagged.loc[df_lagged['date'].idxmax():]
df_latest_indicators = df_latest_date.drop(columns=["date", "GDP", "gdp_growth"])

#ensure that df_pred is sorted in ascending order
df_to_pred = df_to_pred.reset_index(drop=True)
df_to_pred = df_to_pred.sort_values(by = "date", ascending = True)

#for loop to fill in gdp_growth
for index, row in df_to_pred.iterrows():
    # Access values in each row by column name
    date = row['date']
    gdp_growth = row['gdp_growth']
    GDP = row['GDP']
    if index == 0:
        #use model 1
        df_indicators = df_latest_indicators.drop(columns=[col for col in df_latest_indicators.columns if any(col.endswith(f'_lag{i}') for i in [5,6])]) #drop cols of lag5,lag6
        gdp_growth = rf_model_1.predict(df_indicators)
        
    elif index == 1: 
        df_indictors = df_latest_indicators.drop(columns=[col for col in df_latest_indicators.columns if any(col.endswith(f'_lag{i}') for i in [1, 6])]) #drop lag1, lag6
        df_indicators = df_indictors.drop(columns = ["gdp_growth_lag2"])
        gdp_growth = rf_model_2.predict(df_indicators)
    
    elif index == 2:
        df_indicators = df_latest_indicators.drop(columns=[col for col in df_latest_indicators.columns if any(col.endswith(f'_lag{i}') for i in [1,2])]) #drop lag1, lag6
        df_indicators = df_indicators.drop(columns = "gdp_growth_lag3")
        gdp_growth = rf_model_3.predict(df_indicators)

    df_to_pred.at[index, 'gdp_growth'] = gdp_growth


#fill in first nowcasted gdp
df_to_pred["Nowcasted_GDP"] = np.nan
df_to_pred.at[0, "Nowcasted_GDP"] = df_latest_date["GDP"].values[0] * np.exp(df_to_pred.at[0, "gdp_growth"]/400)

#for loop to fill in rest of nowcasted GDP
for i in range(1, len(df_to_pred)):
    # Calculate 'Nowcasted_GDP' for the current row
    df_to_pred.at[i, "Nowcasted_GDP"] = df_to_pred.at[i-1, "Nowcasted_GDP"] * np.exp(df_to_pred.at[i, "gdp_growth"]/400)

df_to_pred.drop(columns = ["GDP"], inplace = True)
df_to_pred.rename(columns = {"gdp_growth":"gdp_growth_forecast"}, inplace = True)
df_to_pred = df_to_pred.head(3)
print(df_to_pred)"""
