# check dataset features for correlations

library(tidyverse)
library(corrplot)
library(polycor)
library(rcompanion)
library(lsr)

# load data
data = read.csv("data/super_integrated_data.csv")

# check for correlations between numeric variables (ignored NA to get general idea)
numeric_df <- data[, c("Age", "Family_income", "Quantity", "Strength", "Day_Supply")] 
cor_matrix <- cor(numeric_df, method = "spearman", use = "pairwise.complete.obs")

# check for correlations between categorical variables
formSex = cramerV(table(data$Form, data$Sex))
formInsurance = cramerV(table(data$Form, data$Insurance_coverage))
formRaceEth = cramerV(table(data$Form, data$Race_ethnicity))
sexInsurance = cramerV(table(data$Sex, data$Insurance_coverage))
sexRaceEth = cramerV(table(data$Sex, data$Race_ethnicity))
insuranceRaceEth = cramerV(table(data$Insurance_coverage, data$Race_ethnicity))

print(paste0("Form Sex Correlation: ", formSex))
print(paste0("Form Insurance Correlation: ", formInsurance))
print(paste0("Form Race Ethnicity Correlation: ", formRaceEth))
print(paste0("Sex Insurance Correlation: ", sexInsurance))
print(paste0("Sex Race Ethnicity Correlation: ", sexRaceEth))
print(paste0("Insurance Race Ethnicity Correlation: ", insuranceRaceEth))

# check for correlations between categorical and numerics
eta_squared_age_form <- etaSquared(lm(Age ~ Form, data = data))
eta_squared_age_sex <- etaSquared(lm(Age ~ Sex, data = data))
eta_squared_age_inscov <- etaSquared(lm(Age ~ Insurance_coverage, data = data))
eta_squared_age_raceeth <- etaSquared(lm(Age ~ Race_ethnicity, data = data))

eta_squared_income_form <- etaSquared(lm(Family_income ~ Form, data = data))
eta_squared_income_sex <- etaSquared(lm(Family_income ~ Sex, data = data))
eta_squared_income_inscov <- etaSquared(lm(Family_income ~ Insurance_coverage, data = data))
eta_squared_income_raceeth <- etaSquared(lm(Family_income ~ Race_ethnicity, data = data))

eta_squared_quantity_form <- etaSquared(lm(Quantity ~ Form, data = data))
eta_squared_quantity_sex <- etaSquared(lm(Quantity ~ Sex, data = data))
eta_squared_quantity_inscov <- etaSquared(lm(Quantity ~ Insurance_coverage, data = data))
eta_squared_quantity_raceeth <- etaSquared(lm(Quantity ~ Race_ethnicity, data = data))

eta_squared_strength_form <- etaSquared(lm(Strength ~ Form, data = data))
eta_squared_strength_sex <- etaSquared(lm(Strength ~ Sex, data = data))
eta_squared_strength_inscov <- etaSquared(lm(Strength ~ Insurance_coverage, data = data))
eta_squared_strength_raceeth <- etaSquared(lm(Strength ~ Race_ethnicity, data = data))

eta_squared_day_supply_form <- etaSquared(lm(Day_Supply ~ Form, data = data))
eta_squared_day_supply_sex <- etaSquared(lm(Day_Supply ~ Sex, data = data))
eta_squared_day_supply_inscov <- etaSquared(lm(Day_Supply ~ Insurance_coverage, data = data))
eta_squared_day_supply_raceeth <- etaSquared(lm(Day_Supply ~ Race_ethnicity, data = data))

print(paste0("Age Form Correlation: ", eta_squared_age_form))
print(paste0("Age Sex Correlation: ", eta_squared_age_sex))
print(paste0("Age Insurance Correlation: ", eta_squared_age_inscov))
print(paste0("Age Race Ethnicity Correlation: ", eta_squared_age_raceeth))

print(paste0("income Form Correlation: ", eta_squared_income_form))
print(paste0("income Sex Correlation: ", eta_squared_income_sex))
print(paste0("income Insurance Correlation: ", eta_squared_income_inscov))
print(paste0("income Race Ethnicity Correlation: ", eta_squared_income_raceeth))

print(paste0("quantity Form Correlation: ", eta_squared_quantity_form))
print(paste0("quantity Sex Correlation: ", eta_squared_quantity_sex))
print(paste0("quantity Insurance Correlation: ", eta_squared_quantity_inscov))
print(paste0("quantity Race Ethnicity Correlation: ", eta_squared_quantity_raceeth))

print(paste0("strength Form Correlation: ", eta_squared_strength_form))
print(paste0("strength Sex Correlation: ", eta_squared_strength_sex))
print(paste0("strength Insurance Correlation: ", eta_squared_strength_inscov))
print(paste0("strength Race Ethnicity Correlation: ", eta_squared_strength_raceeth))

print(paste0("day_supply Form Correlation: ", eta_squared_day_supply_form))
print(paste0("day_supply Sex Correlation: ", eta_squared_day_supply_sex))
print(paste0("day_supply Insurance Correlation: ", eta_squared_day_supply_inscov))
print(paste0("day_supply Race Ethnicity Correlation: ", eta_squared_day_supply_raceeth))