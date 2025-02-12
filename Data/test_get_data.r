library(fredr)
library(dplyr)

fredr_set_key("ae58a77f9383ad8ed12a84122eaa71e6")

start_date <- Sys.Date() - 365*3  # Approx. 3 years back
end_date <- Sys.Date()  # Today's date

# Retrieve GDP data
gdp_data <- fredr(
  series_id = "GDP",  # U.S. Gross Domestic Product
  observation_start = start_date,
  observation_end = end_date
)

# Retrieve CPI data
cpi_data <- fredr(
  series_id = "CPIAUCSL",  # Consumer Price Index for All Urban Consumers
  observation_start = start_date,
  observation_end = end_date
)

# Combine GDP and CPI into a single dataframe by date
combined_data <- full_join(gdp_data, cpi_data, by = "date", suffix = c("_GDP", "_CPI"))


# Ensure data is sorted by date before creating lags
combined_data <- combined_data %>%
  arrange(desc(date)) %>%
  mutate(
    cpi_lag1 = lead(value_CPI, 1),
    cpi_lag2 = lead(value_CPI, 2)
  ) %>%
  select(date, value_GDP, value_CPI)

# Print combined data
print(combined_data)

write.csv(combined_data, "../test_macro_data.csv", row.names = FALSE)
