library(fredr)
library(dplyr)
library(tidyr)
library(purrr)
library(zoo) 

# Set FRED API key
fredr_set_key("ae58a77f9383ad8ed12a84122eaa71e6") 

# 30 years of data
start_date <- Sys.Date() - 365 * 30
end_date <- Sys.Date()

# List of variables selected using LASSO
variables <- list(
  "GDP" = "GDPC1",
  "Industrial_Production" = "INDPRO",
  "Retail_Sales" = "RSAFS",
  "Nonfarm_Payrolls" = "PAYEMS",
  "Trade_Balance" = "BOPGSTB",
  "Core_PCE" = "PCEPILFE",
  "Unemployment" = "UNRATE",
  "Interest_Rate" = "FEDFUNDS",
  "Three_Month_Treasury_Yield" = "DTB3",
  "Construction_Spending" = "TTLCONS",
  "Housing_Starts" = "HOUST",
  "Capacity_Utilization" = "TCU"
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
final_data <- final_data %>%
  select(date, GDP, Industrial_Production, Retail_Sales, Nonfarm_Payrolls,
         Trade_Balance, Core_PCE, Unemployment, Interest_Rate, Three_Month_Treasury_Yield,
         Construction_Spending, Housing_Starts, Capacity_Utilization
  ) %>%
  arrange(desc(date))


write.csv(final_data, "lasso_indicators.csv", row.names = FALSE)
