library(fredr)
library(dplyr)
library(tidyr)
library(purrr)
library(zoo) 

# Set FRED API key
fredr_set_key("ae58a77f9383ad8ed12a84122eaa71e6") 

# 40 years of data
start_date <- Sys.Date() - 365 * 40
end_date <- Sys.Date()

# List of variables 
variables <- list(
  "GDP" = "GDPC1",
  "CPI" = "CPIAUCSL",
  "Crude_Oil" = "DCOILWTICO",
  "Interest_Rate" = "FEDFUNDS",
  "Unemployment" = "UNRATE",
  "Trade_Balance" = "BOPGSTB",
  "PCE" = "PCEC96",
  "Retail_Sales" = "RSAFS",
  "Housing_Starts" = "HOUST",
  "Capacity_Utilization" = "TCU",
  "SP500" = "SP500",
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
  "Personal_Income" = "DSPIC96"
  
)

#retrieve and clean data
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
  select(date, GDP, CPI, Crude_Oil, Interest_Rate, Unemployment, Trade_Balance, 
         PCE, Retail_Sales,Housing_Starts, Capacity_Utilization,
         SP500, Industrial_Production, Nonfarm_Payrolls, PPI, Core_PCE,
         New_Orders_Durable_Goods, Three_Month_Treasury_Yield,
         Consumer_Confidence_Index, New_Home_Sales, Business_Inventories,
         Construction_Spending,
         Wholesale_Inventories, Personal_Income
  ) %>%
  arrange(desc(date))


write.csv(final_data, "../test_macro_data.csv", row.names = FALSE)
