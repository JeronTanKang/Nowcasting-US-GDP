library(fredr)
library(dplyr)
library(tidyr)
library(purrr)
library(zoo) 
library(lubridate)
library(data.table)
library(hdm)
library(glmnet)
library(forecast)


# Set FRED API key
source("api_keys.R")
fredr_set_key(api_key)

#30 years of data
start_date <- as.Date("2025-03-21") - 365 * 30
end_date <- Sys.Date()

# List of variables a
variables <- list(
  "GDP" = "GDPC1",
  "CPI" = "CPIAUCSL",
  "Crude_Oil" = "DCOILWTICO",
  "Interest_Rate" = "FEDFUNDS",
  "Unemployment" = "UNRATE",
  "Trade_Balance" = "BOPGSTB",
  "Retail_Sales" = "RSAFS",
  "Housing_Starts" = "HOUST",
  "Capacity_Utilization" = "TCU",
  "Industrial_Production" = "INDPRO",
  "Nonfarm_Payrolls" = "PAYEMS",
  "PPI" = "PPIACO",
  "Core_PCE" = "PCEPILFE",
  "New_Orders_Durable_Goods" = "DGORDER",
  "Three_Month_Treasury_Yield" = "DTB3",
  "Consumer_Confidence_Index" = "UMCSENT",
  "New_Home_Sales" = "HSN1F",
  "Business_Inventories" = "BUSINV",
  "Construction_Spending" = "TTLCONS",
  "Wholesale_Inventories" = "WHLSLRIMSA",
  "Personal_Income" = "DSPIC96",
  "AAA" = "AAA",
  "BAA" = "BAA",
  "yield_spread" = "T10Y3MM"
  
)

# Function to retrieve and clean data
fetch_fred_data <- function(series_id, new_name) {
  fredr(series_id = series_id, observation_start = start_date, observation_end = end_date) %>%
    mutate(date = format(date, "%Y-%m")) %>%  # Convert to Year-Month format
    rename(!!new_name := value) %>%
    distinct(date, .keep_all = TRUE)  # Remove duplicates if any
}

# Fetch data for all variables
data_list <- lapply(names(variables), function(name) fetch_fred_data(variables[[name]], name))

# Merge all datasets on 'date'
final_data <- reduce(data_list, full_join, by = "date")
final_data <- final_data %>% mutate(date = as.Date(paste0(date, "-01"))) %>%
  select(date, GDP, CPI, Crude_Oil, Interest_Rate, Unemployment, Trade_Balance, 
         Retail_Sales,Housing_Starts, Capacity_Utilization,
         Industrial_Production, Nonfarm_Payrolls, PPI, Core_PCE,
         New_Orders_Durable_Goods, Three_Month_Treasury_Yield,
         Consumer_Confidence_Index, New_Home_Sales, Business_Inventories,
         Construction_Spending,
         Wholesale_Inventories, Personal_Income, AAA, BAA, yield_spread
  ) %>% mutate(junk_bond_spread = BAA - AAA) %>% select(-AAA) %>% select(-BAA) %>%
  arrange(desc(date))

#add a year_quarter col
final_data <- final_data %>%
  mutate(year_quarter = paste0(year(date), "Q", quarter(date)) 
  )  %>% select(date, year_quarter, everything())




# Function to aggregate indicators from monthly to quarterly
aggregate_indicators <- function(df) {
  
  aggregation_rule <- list(
    "CPI" = "mean",
    "Crude_Oil" = "mean",
    "Interest_Rate" = "mean",
    "Unemployment" = "mean",
    "Trade_Balance" = "sum",
    "Retail_Sales" = "sum",
    "Housing_Starts" = "sum",
    "Capacity_Utilization" = "mean",
    "Industrial_Production" = "mean",
    "Nonfarm_Payrolls" = "sum",
    "PPI" = "mean",
    "Core_PCE" = "mean",
    "New_Orders_Durable_Goods" = "sum",
    "Three_Month_Treasury_Yield" = "mean",
    "Consumer_Confidence_Index" = "mean",
    "New_Home_Sales" = "sum",
    "Business_Inventories" = "sum",
    "Construction_Spending" = "sum",
    "Wholesale_Inventories" = "sum",
    "Personal_Income" = "mean",
    "yield_spread" = "mean",
    "junk_bond_spread" = "mean"
  )
  
  
  gdp_data <- df %>%
    filter(!is.na(GDP)) %>%  # Remove NA values first
    group_by(year_quarter) %>%
    summarise(GDP = last(GDP)) %>%
    ungroup()
  
  # Aggregating indicators 
  indicators_data <- df %>%
    group_by(year_quarter) %>%
    summarise(across(
      names(aggregation_rule),
      ~ if (aggregation_rule[[cur_column()]] == "mean") mean(.x, na.rm = TRUE) else
        if (aggregation_rule[[cur_column()]] == "sum") sum(.x, na.rm = TRUE),
      .names = "{.col}"
    )) %>% ungroup()
  
  # Merging GDP and indicators
  quarterly_df <- full_join(gdp_data, indicators_data, by = "year_quarter")
  
  
  return(quarterly_df)
}


temp_data <- final_data


final_data <- aggregate_indicators(final_data) 
final_data <- final_data %>% arrange((year_quarter)) #arrange in asc date order






#temp_df to store D1_Unemployment and junk_bond_spread, yield_spread (remove obs after covid)
temp_df <- final_data %>% 
  select(year_quarter, Unemployment, junk_bond_spread, yield_spread) %>% arrange(year_quarter) %>%
  mutate(Unemployment = c(NA, diff(Unemployment))) %>% arrange(desc(year_quarter))%>% filter(row_number() > 21) %>% arrange(year_quarter) 


#rest of indicators 
df_not_stationary <- final_data %>% select(-c(Unemployment, GDP, junk_bond_spread, yield_spread)) %>% 
  arrange(desc(year_quarter)) %>% filter(row_number() > 21) %>% arrange(year_quarter)



#gdf_df to store create annualised gdp growth rate.
gdp_df <- final_data %>% select(year_quarter, GDP) %>% mutate(gdp_lag = lag(GDP)) %>% 
  mutate(gdp_growth = 400 * (log(GDP) - log(gdp_lag))) %>% select(-gdp_lag) %>% arrange(desc(year_quarter)) %>%
  filter(row_number() > 21)


#create an empty list to store differencing orders
diff_orders <- c()

#making data stationary (ndiffs uses KPSS test by default)
# Loop through each column exluding date
for (col in colnames(df_not_stationary)) {
  if (col != "year_quarter") {  
    
    # Find the number of differences needed for each indicator
    order <- ndiffs(df_not_stationary[[col]], test = "adf")  #use adf test
    
    # Store the differencing order
    diff_orders[col] <- order
  }
}

# Create a copy of the dataset
data_stationary <- df_not_stationary

# Apply differencing column by column
for (col in colnames(df_not_stationary)) {
  if (col != "year_quarter" && col %in% names(diff_orders)) {  # Skip date column
    
    # Get the differencing order for this column
    order <- diff_orders[col]
    
    # Apply differencing only if needed
    if (order > 0) {
      data_stationary[[col]] <- c(rep(NA, order), diff(df_not_stationary[[col]], differences = order))
    }
  }
}

#join the 3 dfs together and scale indicators except gdp_growth
df_stationary <- gdp_df %>% left_join(data_stationary, by = "year_quarter") %>% left_join(temp_df, by = "year_quarter") %>% arrange(year_quarter) %>% 
  mutate(across(-c(year_quarter, GDP, gdp_growth), ~ as.numeric(scale(.)))) #arrange in desc date order





# Function to create lagged variables
lag_features <- function(data, lags = 4) {
  data %>%
    mutate(across(
      .cols = -year_quarter,  # Exclude the date column
      .fns = list(
        lag1 = ~ lag(., 1),
        lag2 = ~ lag(., 2),
        lag3 = ~ lag(., 3),
        lag4 = ~ lag(., 4)
      ),
      .names = "{.col}_{.fn}"  # Naming format: "GDP_lag1", "CPI_lag2", etc.
    ))
}



# Apply the function to your dataset
df_lagged <- lag_features(df_stationary, lags = 4) %>% arrange(desc(year_quarter)) %>% 
  select(-c(GDP_lag1, GDP_lag2, GDP_lag3, GDP_lag4))


#running LASSO to identify important indicators
df_lasso <- df_lagged %>% drop_na()  %>% arrange(year_quarter)# 8 obs dropped - first 4 and last 2



#y variable
y <- df_lasso$gdp_growth

#indicators in matrix form
x <- df_lasso %>% select(-c(year_quarter, GDP, gdp_growth))

lasso_model <- rlasso(x, y, post = FALSE)



lasso_coeffs <- coef(lasso_model)

# Convert to a data frame
coeff_df <- data.frame(variable = names(lasso_coeffs), coefficient = as.numeric(lasso_coeffs))

# Remove intercept and filter non-zero coefficients
selected_vars <- coeff_df %>%
  filter(coefficient != 0, variable != "(Intercept)") %>% mutate(Importance = abs(coefficient)) %>%
  arrange(desc(Importance))  # Sorting in descending order

# Print sorted coefficients
print(selected_vars)
#chosen indicators (in order of importance): 
#Nonfarm_Payrolls, Construction_Spending, Trade_Balance_lag1 , Industrial_Production_lag3, Housing_Starts,
#Capacity_Utilization, New_Orders_Durable_Goods, Interest_Rate_lag1, junk_bond_spread_lag1, Unemployment 
