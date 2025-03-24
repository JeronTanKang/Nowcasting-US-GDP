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

write.csv(final_df, "tree_df.csv", row.names = FALSE)
