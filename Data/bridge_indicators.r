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
fredr_set_key("ae58a77f9383ad8ed12a84122eaa71e6") 

#30 years of data
start_date <- Sys.Date() - 365 * 30
end_date <- Sys.Date()

# List of variables 
variables <- list(
  "GDP" = "GDPC1",
  "Nonfarm_Payrolls" = "PAYEMS",
  "Construction_Spending" = "TTLCONS",
  "Trade_Balance" = "BOPGSTB",
  "Industrial_Production" = "INDPRO",
  "Housing_Starts" = "HOUST",
  "Capacity_Utilization" = "TCU",
  "New_Orders_Durable_Goods" = "DGORDER",
  "Interest_Rate" = "FEDFUNDS",
  "Unemployment" = "UNRATE",
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
  select(date, GDP,Nonfarm_Payrolls, Construction_Spending, Trade_Balance, Industrial_Production, Housing_Starts,
         Capacity_Utilization,New_Orders_Durable_Goods,Interest_Rate, Unemployment, AAA, BAA, yield_spread) %>% 
  mutate(junk_bond_spread = BAA - AAA) %>% select(-AAA) %>% select(-BAA) %>% arrange(desc(date))

#function to determine how many extra rows
get_missing_months <- function(df, date_column = "date") {
  # Ensure the date column is in Date format
  df <- df %>%
    mutate(!!sym(date_column) := as.Date(!!sym(date_column)))
  
  # Get the latest date in the dataset
  latest_date <- max(df[[date_column]], na.rm = TRUE)
  
  # Get the current date and find the end of the quarter 2 quarters from now
  current_date <- floor_date(latest_date, unit = "month")  # First day of current month
  target_date <- ceiling_date(current_date %m+% months(6), unit = "quarter") - days(1)  # End of 2 quarters from now
  
  # Calculate number of months missing
  num_missing_months <- interval(latest_date, target_date) %/% months(1)
  
  return(max(0, num_missing_months))
}

#no.of rows to add:
num_extra_rows <- get_missing_months(final_data, "date")


#get the latest date in the dataset
latest_date <- max(final_data$date, na.rm = TRUE)

# Create new rows with incremented dates
new_dates <- seq.Date(from = latest_date + months(1), by = "1 month", length.out = num_extra_rows)
new_rows <- data.frame(date = new_dates) %>% as_tibble()

# Add missing columns with NA values
new_rows <- new_rows %>% mutate(date = as.Date(date)) # Assuming numerical columns, adjust if needed

# Bind new rows to the existing dataframe
final_data <- bind_rows(new_rows, final_data) %>% arrange(date) 




#gdf_df to store GDP and gdp growth rate.
gdp_growth <- final_data %>% select(date, GDP) %>% drop_na() %>% mutate(gdp_lag = lag(GDP)) %>% 
  mutate(gdp_growth = 400 * (log(GDP) - log(gdp_lag))) %>% select(-c(GDP, gdp_lag)) %>% arrange(desc(date))

gdp_df <- final_data %>% select(date, GDP) %>% left_join(gdp_growth, by = "date")


#temp_df to store indicators that are already stationary(junk_bond_spread, yield_spread) and difference unemployment once
temp_df <- final_data %>% 
  select(date, Unemployment, junk_bond_spread, yield_spread) %>% arrange(date) %>%
  mutate(Unemployment = c(NA, diff(Unemployment)))

#df_not_stationary to store rest of indicators that needs to be made stationary
df_not_stationary <- final_data %>% select(-c(GDP,Unemployment, junk_bond_spread, yield_spread)) %>% arrange(date)

# differencing orders from lasso
diff_orders <- c(
  CPI = 1,
  Crude_Oil = 1,
  Interest_Rate = 1,
  Trade_Balance = 1,
  Retail_Sales = 1,
  Housing_Starts = 1,
  Capacity_Utilization = 0,
  Industrial_Production = 0,
  Nonfarm_Payrolls = 2,
  PPI = 1,
  Core_PCE = 1,
  New_Orders_Durable_Goods = 1,
  Three_Month_Treasury_Yield = 1,
  Consumer_Confidence_Index = 1,
  New_Home_Sales = 1,
  Business_Inventories = 1,
  Construction_Spending = 1,
  Wholesale_Inventories = 1,
  Personal_Income = 1
)

# Create a copy of the dataset
data_stationary <- df_not_stationary

# Apply differencing column by column
for (col in colnames(df_not_stationary)) {
  if (col != "date" && col %in% names(diff_orders)) {  # Skip date column
    
    # Get the differencing order for this column
    order <- diff_orders[col]
    
    # Apply differencing only if needed
    if (order > 0) {
      data_stationary[[col]] <- c(rep(NA, order), diff(df_not_stationary[[col]], differences = order))
    }
  }
}

#join 3 sets together
data_stationary <- gdp_df %>% left_join(temp_df, by = "date") %>% left_join(data_stationary, by = "date") %>% 
  arrange(date) 


lag_features <- function(data, lags = 4) {
  data %>%
    mutate(across(
      .cols = -c(date, GDP, gdp_growth),  # Exclude the date column and raw_GDP
      .fns = list(
        lag1 = ~ lag(., 1),
        lag2 = ~ lag(., 2),
        lag3 = ~ lag(., 3),
        lag4 = ~ lag(., 4)
      ),
      .names = "{.col}_{.fn}"  # Naming format: "GDP_lag1", "CPI_lag2", etc.
    ))
}

#to lag gdp_growth
lag_gdp_growth <- function(data, lags = 4) {
  data %>%
    mutate(across(
      .cols = gdp_growth,  # Exclude the date column and raw_GDP
      .fns = list(
        lag1 = ~ lag(., 3),
        lag2 = ~ lag(., 6),
        lag3 = ~ lag(., 9),
        lag4 = ~ lag(., 12)
      ),
      .names = "{.col}_{.fn}"  # Naming format: "GDP_lag1", "CPI_lag2", etc.
    ))
}




# Apply the function to your dataset
df_lagged <- lag_features(data_stationary, lags = 4)
df_lagged <- lag_gdp_growth(df_lagged, lags = 4)
df_lagged <- df_lagged %>% arrange(desc(date))

#only select indicators that are needed
final_df <- df_lagged %>% select(date, GDP, gdp_growth, gdp_growth_lag1, gdp_growth_lag2, gdp_growth_lag3, gdp_growth_lag4,
                                 Nonfarm_Payrolls, Construction_Spending, Trade_Balance_lag1, Industrial_Production_lag1,
                                 Industrial_Production_lag3, Housing_Starts, Capacity_Utilization, New_Orders_Durable_Goods, 
                                 Interest_Rate_lag1, Unemployment, junk_bond_spread, junk_bond_spread_lag1,
                                 junk_bond_spread_lag2, junk_bond_spread_lag3, junk_bond_spread_lag4)

#add dummy variable to indicate recession
recession_dates <- as.Date(c(
  "2020-06-01", "2020-05-01", "2020-04-01", "2020-03-01", "2020-02-01", "2020-01-01",
  "2009-06-01", "2009-05-01", "2009-04-01", "2009-03-01", "2009-02-01", "2009-01-01",
  "2008-12-01", "2008-11-01", "2008-10-01", "2008-09-01", "2008-08-01", "2008-07-01",
  "2008-06-01", "2008-05-01", "2008-04-01", "2008-03-01", "2008-02-01", "2008-01-01",
  "2001-12-01", "2001-11-01", "2001-10-01", "2001-09-01", "2001-08-01", "2001-07-01"
))

final_df <- final_df %>% mutate(dummy = if_else(date %in% recession_dates, 1, 0))



write.csv(final_df, "bridge_df.csv", row.names = FALSE)



