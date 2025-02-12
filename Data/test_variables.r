library(fredr)
library(dplyr)
library(tidyr)
library(purrr)
library(zoo)  # For handling missing values

# Set your FRED API key
fredr_set_key("ae58a77f9383ad8ed12a84122eaa71e6")

# Define date range (past 3 years)
start_date <- Sys.Date() - 365 * 20
end_date <- Sys.Date()

# Function to retrieve data and keep only Year-Month
fetch_fred_data <- function(series_id, new_name) {
  fredr(series_id = series_id, observation_start = start_date, observation_end = end_date) %>%
    mutate(date = format(date, "%Y-%m")) %>%  # Convert to Year-Month format
    rename(!!new_name := value) %>%
    distinct(date, .keep_all = TRUE)  # Remove duplicates if any
}

# Retrieve data from FRED & rename value column appropriately
gdp_data <- fetch_fred_data("GDP", "GDP")
cpi_data <- fetch_fred_data("CPIAUCSL", "CPI")
oil_data <- fetch_fred_data("DCOILWTICO", "Crude_Oil")
ir_data <- fetch_fred_data("FEDFUNDS", "Interest_Rate")
unemployment_data <- fetch_fred_data("UNRATE", "Unemployment") 
trade_bal_data <- fetch_fred_data("BOPGSTB", "Trade_bal") 
consumption_data <- fetch_fred_data("PCEC96", "Consumption")
investment_data <- fetch_fred_data("GPDIC1", "Investment")

# Create a full sequence of Year-Month dates for merging
date_seq <- tibble(date = seq.Date(from = as.Date(start_date), to = as.Date(end_date), by = "month")) %>%
  mutate(date = format(date, "%Y-%m"))

# Merge all datasets by Year-Month
data_list <- list(gdp_data, cpi_data, oil_data, ir_data, unemployment_data, trade_bal_data, consumption_data, investment_data)
combined_data <- reduce(data_list, full_join, by = "date")

# Merge with full date sequence to ensure all Year-Months are included
combined_data <- full_join(date_seq, combined_data, by = "date")
combined_data <- combined_data %>% select(date, GDP, CPI, Crude_Oil, Interest_Rate, Unemployment, Trade_bal, Consumption, Investment )

write.csv(combined_data, "../test_macro_data.csv", row.names = FALSE)
