# Integrate all relevant data files

# load libraries
library(readxl)
library(tidyverse)
library(janitor)

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

# check column dimensions
meps_2014 = meps_2014 %>%
  select(-c(X, Form.Units, Therapeutic_class, Administration_route))
meps_2016 = meps_2016 %>%
  select(-c(X, Form.Units, Therapeutic_class, Administration_route, route))
meps_2018 = meps_2018 %>%
  select(-c(Form.Units, Therapeutic_class, Administration_route, route))
meps_2020 = meps_2020 %>%
  select(-c(X, Form.Units, Therapeutic_class, Administration_route))
meps_2022 = meps_2022 %>%
  select(-c(Form.Units, Therapeutic_class, Administration_route, route))

# bind MEPS dataframes
integrated_data = rbind(meps_2014,meps_2016,meps_2018,meps_2020,meps_2021,meps_2022)

# retain drugs repeated in all 6 years, drop others
integrated_data <- integrated_data %>%
  group_by(Drug) %>%
  filter(n_distinct(Year) == 6) %>%
  ungroup()

# convert categorical variables back to original values (to integrate better
# with all models, which require different categorical variable handling)
integrated_data = integrated_data %>%
  mutate(Sex = as.character(Sex)) %>%
  mutate(Insurance_coverage = as.character(Insurance_coverage)) %>%
  mutate(Race_ethnicity = as.character(Race_ethnicity)) %>%
  mutate(Sex = case_when(Sex==1 ~ "Male",
                         Sex==2 ~ "Female",
                         TRUE ~ Sex)) %>%
  mutate(Insurance_coverage = case_when(Insurance_coverage==1 ~ "Private",
                                        Insurance_coverage==2 ~ "Public",
                                        Insurance_coverage==3 ~ "Uninsured",
                                        TRUE ~ Insurance_coverage)) %>%
  mutate(Race_ethnicity = case_when(Race_ethnicity==1 ~ "Hispanic",
                                    Race_ethnicity==2 ~ "Non-Hispanic White",
                                    Race_ethnicity==3 ~ "Non-Hispanic Black",
                                    Race_ethnicity==4 ~ "Non-Hispanic Asian",
                                    Race_ethnicity==5 ~ "Non-Hispanic Other",
                                    TRUE ~ Race_ethnicity))

# checked for exact duplicates with this (there were none):
# duplicates <- integrated_data[duplicated(integrated_data), ]

# create unique Observation_ID for each row to link metadata with model output
integrated_data = integrated_data %>%
  rowid_to_column(var = "Observation_ID")

# remove last year of data (2022) for internal validation
data_2022 = integrated_data %>%
  filter(Year=="2022")
integrated_data = integrated_data %>%
  filter(Year!="2022")

# create metadata dataframe
metadata = integrated_data %>%
  select(c(Observation_ID, Person_ID, NDC, Household_ID))
metadata_2022 = data_2022 %>%
  select(c(Observation_ID, Person_ID, NDC, Household_ID))

# create dataframe with drug prescription info
prescription = integrated_data %>%
  select(c(Observation_ID, Drug, Quantity, Form, Strength, Day_Supply,
           Daily_Frequency, Daily_Dosage, Year))
prescription_2022 = data_2022 %>%
  select(c(Observation_ID, Drug, Quantity, Form, Strength, Day_Supply,
           Daily_Frequency, Daily_Dosage, Year))

# update integrated_data for modeling (keep Person ID here as well for proper
# train/test data splitting)
integrated_data = integrated_data %>%
  select(c(Observation_ID, Person_ID, Household_ID, Drug, Age, Sex, Family_income,
           Insurance_coverage, Race_ethnicity, Year))
model_data_2022 = data_2022 %>% 
  select(c(Observation_ID, Person_ID, Household_ID, Drug, Age, Sex, Family_income,
           Insurance_coverage, Race_ethnicity, Year))

# print to csv
write.csv(integrated_data, "data/integrated_data.csv")
write.csv(model_data_2022, "data/data_2022.csv")
write.csv(metadata, "data/metadata.csv")
write.csv(metadata_2022, "data/metadata_2022.csv")
write.csv(prescription, "data/prescription.csv")
write.csv(prescription_2022, "data/prescription_2022.csv")