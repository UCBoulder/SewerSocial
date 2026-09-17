# PharmFlush mdoel, converted to a function, specific return format for
# compatible user interface in Shiny app.

################################################################################
# pharmflush_model function:
#   * Description: Predicts the influent concentration and mass load of each of
#                  313 pharmaceuticals provided the PharmUse database, the flow
#                  capacity of each water resource recovery facility (WRRF), and
#                  the sewershed population size.
#   * Input: 
#       * flow_capacity: class = numeric, typical daily flow or design capacity
#                        (whichever is available) for the given WRRF in L/d
#       * population_served: class = numeric, size of the population in the 
#                            sewershed of the given WRRF
#   * Output:
#       * average_profile: class = list, list that includes the
#                          average predicted concentration and mass load for
#                          each drug across the 100 sampled ensembles. The 
#                          average number of prescriptions, mass of the dose,
#                          and standard error of the mass load are also included
#                          for each pharmaceutical.
################################################################################

library("stats")
library("tidyverse")
library("svglite")
library("patchwork")
library("cowplot")
library("readxl")
library("scales")

pharmflush_model <- function(flow_capacity, population_served) {
  
  pharmuse = read.csv("pharmuse.csv")
  
  # Define model parameters
  n = 26847 # size of representative sample (obtained from MEPS 2020
  # documentation, Table 3.2)
  ww_volume_per_capita = flow_capacity / population_served
  x = 0:population_served # range of number of prescriptions detected in
  # sewershed any given day for given drug
  p = 0 # initialize probability of detecting certain pharmaceutical on any given day
  avg_mass = 0 # initialize avg mass per day variable for each pharmaceutical
  
  # mutate PharmUse to get excreted_dose column from metabolism data
  pharmuse = pharmuse %>%
    mutate(Excretion_fraction = Excretion_percentage/100) %>%
    mutate(Excreted_dose = Average_Daily_Dosage*Excretion_fraction)
  
  # change column order
  pharmuse = pharmuse %>%
    relocate(c(Excretion_fraction, Excreted_dose), .after = Average_Daily_Dosage)
  
  # update PharmUse to be total excreted dose for all administration routes
  # instead of the excreted dose for each route
  # (average of average excreted doses per route, sum of number of prescriptions
  # per route, average of average duration of prescription per route)
  pharmuse2 = pharmuse %>%
    group_by(Pharmaceutical) %>%
    summarise(
      Excreted_dose = mean(Excreted_dose, na.rm = TRUE),
      Number_of_Prescriptions = sum(Number_of_Prescriptions, na.rm = TRUE),
      Average_Duration_per_Prescription = mean(Average_Duration_per_Prescription,
                                               na.rm = TRUE),
      .groups = "drop"
    )
  
  # pull relevant variables from updated PharmUse
  drugs = pharmuse2$Pharmaceutical
  excreted_dose = pharmuse2$Excreted_dose
  prescrips = pharmuse2$Number_of_Prescriptions
  durations = pharmuse2$Average_Duration_per_Prescription
  
  # calculate probability of detecting each pharmaceutical on any given day and 
  # create list of average daily mass for each pharmaceutical
  for (i in 1:length(drugs)){
    p[i] = (prescrips[i]*durations[i])/(n*365) 
    avg_mass[i] = excreted_dose[i]
  }
  
  # Calculate probability P of detecting exactly x prescriptions for given drug in
  # sewershed of size population_served with detection probability p
  P = list()
  for (i in 1:length(drugs)){ 
    new_P = dbinom(x,population_served,p[i]) 
    P = append(P, list(new_P))
  }
  
  # convert P to data frame and label each column by drug name
  P = data.frame(P)
  names(P) = drugs
  
  # add number of prescriptions detected (x variable) to P
  P = P %>%
    mutate(Number_prescrips_detected = x)
  
  # name common column and all drug columns to allow splitting into separate df's
  Number_prescrips_detected = "Number_prescrips_detected"
  drug_col_names = colnames(P)[!colnames(P) %in% Number_prescrips_detected]
  
  # Split the data frame into separate data frames for each drug with number of
  # prescriptions detected and probability of each number of prescriptions being
  # detected 
  drug_dataframes = lapply(drug_col_names, function(drug_col) {
    split_df = P[c(Number_prescrips_detected, drug_col)]
    colnames(split_df)[2] = "Probability"  
    return(split_df)
  })
  names(drug_dataframes) = drugs
  
  # randomly pull (# prescriptions, Probability) pairs for each drug to randomly
  # construct wastewater flows by bootstrapping (generating 100 possible drug
  # concentration/mass load profiles or ensembles)
  set.seed(123) 
  sampled_dataframes = list()
  sample_size = 100 
  
  for (i in 1:length(drugs)) {
    drug_df = drug_dataframes[[i]] 
    prob_distribution = drug_df[2]
    sampled_rows = sample(nrow(drug_df), size = sample_size, replace = TRUE,
                          prob=unlist(prob_distribution)) 
    sampled_df = drug_df[sampled_rows, ]
    sampled_df$Drug = drugs[i]
    sampled_dataframes[[i]] = sampled_df
  }
  
  final_sampled_data = do.call(rbind, sampled_dataframes)
  
  # randomly combine results to get sample_size number of drug profiles
  ensemble_drug_profiles = list()
  for (i in 1:sample_size){
    selected_rows = final_sampled_data[seq(i, nrow(final_sampled_data),
                                           by = sample_size), ]
    ensemble = data.frame(
      Drug = selected_rows$Drug,
      Number_prescrips_detected = selected_rows$Number_prescrips_detected,
      Probability = selected_rows$Probability
    )
    ensemble = ensemble %>%
      mutate(Average_daily_mass = avg_mass) %>%
      mutate(Predicted_mass = Average_daily_mass*Number_prescrips_detected) %>%
      mutate(Predicted_concentration = Predicted_mass/flow_capacity) %>%
      # divide by 4 to correct for medication non-compliance of 50% and employees'
      # contributions to wastewater flow (assume equal number of non-resident
      #  employees to residents)
      mutate(Predicted_concentration = Predicted_concentration/4) %>%
      mutate(Predicted_mass_load = Predicted_concentration*ww_volume_per_capita) %>%
      arrange(desc(Predicted_concentration))
    ensemble_drug_profiles[[i]] = ensemble
  }
  
  # get separate data frame for each drug's ensemble prediction (100 per drug)
  
  # Initialize an empty list to store the 290 new dataframes
  indiv_drug_profiles = list()
  
  # Loop over each pharmaceutical name
  for (drug in drugs) {
    
    # Initialize an empty list to collect data for this pharmaceutical across all
    # dataframes
    pharma_data = list()
    
    # Loop over the list of dataframes
    for (ensemble_drug_profile in ensemble_drug_profiles) {
      
      # Extract rows corresponding to the current pharmaceutical
      pharma_row = ensemble_drug_profile[ensemble_drug_profile$Drug == drug, ]
      
      # Add the row to the pharma_data list
      pharma_data = rbind(pharma_data, pharma_row)
    }
    
    # Combine all the data for this pharmaceutical into a single dataframe
    indiv_drug_profiles[[drug]] = pharma_data
  }
  
  # average predicted concentration/mass load for each drug to form final drug
  # concentration/mass load profile that averages across all 100 ensembles for
  # each drug
  average_concs = list()
  average_loads = list()
  average_prescrips = list()
  average_masses = list()
  sem_loads = list() # SEM = standard error of mean
  
  for (i in 1:length(indiv_drug_profiles)) {
    avg_conc = mean(unlist(indiv_drug_profiles[[i]][6]))
    avg_conc = matrix(avg_conc, ncol = 1)
    average_concs[i] = avg_conc
    avg_load = mean(unlist(indiv_drug_profiles[[i]][7]))
    avg_load = matrix(avg_load, ncol = 1)
    average_loads[i] = avg_load
    avg_mass = mean(unlist(indiv_drug_profiles[[i]][5]))
    avg_mass = matrix(avg_mass, ncol = 1)
    average_masses[i] = avg_mass
    stan_dev_load = sd(unlist(indiv_drug_profiles[[i]][7]))
    sem_load = stan_dev_load / sqrt(sample_size)
    sem_load = matrix(sem_load, ncol = 1)
    sem_loads[i] = sem_load
    
    avg_prescrip = mean(unlist(indiv_drug_profiles[[i]][2]))
    avg_prescrip = matrix(avg_prescrip, ncol = 1)
    average_prescrips[i] = avg_prescrip
  }
  
  # put in form that is compatible with data frame
  average_concs = unlist(average_concs)
  average_loads = unlist(average_loads)
  average_prescrips = unlist(average_prescrips)
  average_masses = unlist(average_masses)
  sem_loads = unlist(sem_loads)
  
  # make data frame for average profile for each drug
  average_profile = data.frame(Drug=drugs,
                               Average_Number_Prescriptions=average_prescrips,
                               Average_Masses=average_masses,
                               Average_Predicted_Concentration=average_concs,
                               Average_Predicted_Mass_Load=average_loads,
                               Standard_Error_Predicted_Mass_Load=sem_loads)
  average_profile = average_profile %>%
    arrange(desc(Average_Predicted_Concentration))

  # Final return
  return(
    list(
      average_profile = average_profile
    )
  )
}
