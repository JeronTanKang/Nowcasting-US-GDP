#This code is to extract all columns in FRED MD current.csv
library(fredr)
library(dplyr)
library(tidyr)
library(purrr)
library(zoo) 
library(lubridate)

# Read the CSV file
df <- read.csv("../data/current.csv")
df <- df[-1, ]

# Convert 'sasdate' to Date format
df <- df %>%
  mutate(sasdate = as.Date(sasdate, format="%m/%d/%Y")) %>% rename(date = sasdate) %>% arrange(desc(date))

#filter dates
start_date <- Sys.Date() - 365 * 40
end_date <- Sys.Date()
df <- df %>% filter(date >= start_date) 

#change columns to numeric
df <- df %>% mutate(across(where(is.character), as.numeric)) %>% mutate(date = format(date, "%Y-%m"))



