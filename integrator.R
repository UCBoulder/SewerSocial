# Integrate all relevant data files

# load libraries
library(readxl)
library(tidyverse)

# load data
meps_2014 = read.csv("data/meps_clean_2014.csv")
meps_2016 = read.csv("data/meps_clean_2016.csv")
meps_2018 = read.csv("data/meps_clean_2018.csv")
meps_2020 = read.csv("data/meps_clean_2020.csv")
meps_2021 = read.csv("data/meps_clean_2021.csv")
meps_2022 = read.csv("data/meps_clean_2022.csv")

# change colnames and prepare to merge the dataframes
meps_2014$Drug <- tolower(meps_2014$Drug)
meps_2016$Drug <- tolower(meps_2016$Drug)
meps_2018$Drug <- tolower(meps_2018$Drug)
meps_2020$Drug <- tolower(meps_2020$Drug)
meps_2021$Drug <- tolower(meps_2021$Drug)
meps_2022$Drug <- tolower(meps_2022$Drug)

# change PMED access column to have same name
meps_2018 <- meps_2018 %>% rename(Can.t_get_PMED = Can.t_afford_PMED)
meps_2020 <- meps_2020 %>% rename(Can.t_get_PMED = Can.t_afford_PMED)
meps_2021 <- meps_2021 %>% rename(Can.t_get_PMED = Can.t_afford_PMED)
meps_2022 <- meps_2022 %>% rename(Can.t_get_PMED = Can.t_afford_PMED)

# drop extraneous columns in each MEPS data
meps_2014 = meps_2014 %>%
  select(-c(X,Amount.paid.self.or.family,Amount.paid.Medicare,Amount.paid.Medicaid,
            Amount.paid.private.insurance,Amount.paid.veterans.CHAMPVA,Amount.paid.Tricare,
            Amount.paid.other.federal,Amount.paid.state.or.local,Amount.paid.worker.s.comp,
            Amount.paid.other.insurance,Amount.paid.other.private,Amount.paid.other.public))
meps_2016 = meps_2016 %>%
  select(-c(X,Amount.paid.self.or.family,Amount.paid.Medicare,Amount.paid.Medicaid,
            Amount.paid.private.insurance,Amount.paid.veterans.CHAMPVA,Amount.paid.Tricare,
            Amount.paid.other.federal,Amount.paid.state.or.local,Amount.paid.worker.s.comp,
            Amount.paid.other.insurance,Amount.paid.other.private,Amount.paid.other.public))
meps_2018 = meps_2018 %>%
  select(-c(X,Amount.paid.self.or.family,Amount.paid.Medicare,Amount.paid.Medicaid,
            Amount.paid.private.insurance,Amount.paid.veterans.CHAMPVA,Amount.paid.Tricare,
            Amount.paid.other.federal,Amount.paid.state.or.local,Amount.paid.worker.s.comp,
            Amount.paid.other.insurance,Amount.paid.other.private,Amount.paid.other.public))
meps_2020 = meps_2020 %>%
  select(-c(X,Amount.paid.self.or.family,Amount.paid.Medicare,Amount.paid.Medicaid,
            Amount.paid.private.insurance,Amount.paid.veterans.CHAMPVA,Amount.paid.Tricare,
            Amount.paid.other.federal,Amount.paid.state.or.local,Amount.paid.worker.s.comp,
            Amount.paid.other.insurance))
meps_2021 = meps_2021 %>%
  select(-c(X,Amount.paid.self.or.family,Amount.paid.Medicare,Amount.paid.Medicaid,
            Amount.paid.private.insurance,Amount.paid.veterans.CHAMPVA,Amount.paid.Tricare,
            Amount.paid.other.federal,Amount.paid.state.or.local,Amount.paid.worker.s.comp,
            Amount.paid.other.insurance))
meps_2022 = meps_2022 %>%
  select(-c(X,Amount.paid.self.or.family,Amount.paid.Medicare,Amount.paid.Medicaid,
            Amount.paid.private.insurance,Amount.paid.veterans.CHAMPVA,Amount.paid.Tricare,
            Amount.paid.other.federal,Amount.paid.state.or.local,Amount.paid.worker.s.comp,
            Amount.paid.other.insurance))

# bind MEPS dataframes
integrated_data = rbind(meps_2014,meps_2016,meps_2018,meps_2020,meps_2021,meps_2022)

# retain drugs repeated in all 6 years, drop others
integrated_data <- integrated_data %>%
  group_by(Drug) %>%
  filter(n_distinct(Year) == 6) %>%
  ungroup()

# remove irrelevant columns for predicting drug from demographics
integrated_data = integrated_data %>%
  select(-c(Administration_route,Form.Units,NDC,Sum.of.all.payments,Form))

# print to csv
write.csv(integrated_data, "data/integrated_data.csv")