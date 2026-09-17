# This script uses the distributions of demographic features in the MEPS data to
# assign prescription features to demographic data from specific communities.

library(dplyr)
library(tidyr)
library(stringr)
library(purrr)
library(readxl)
library(data.table)

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING (Original Pipeline)
# ==============================================================================

# Load all MEPS data
meps_set <- read.csv("data/super_integrated_data.csv")

# set Person ID col
person_id_col <- "Person_ID" 

# Define demographic vs. prescription columns
demo_cols <- c("Age", "Sex", "Race_ethnicity", "Family_income",
               "Insurance_coverage")
rx_cols   <- c("Strength", "Day_Supply", "Quantity", "Form") 

# Make sure all missing values are NA
meps_set <- meps_set %>%
  mutate(across(all_of(rx_cols), ~ {
    x <- str_trim(as.character(.)) 
    x <- na_if(x, "")              
    x <- na_if(x, "NA")           
    return(x)
  }))

# Drop NA from MEPS (except if all rx_cols are NA because this is for the
# "no prescriptions" class)
meps_set <- meps_set %>%
  drop_na(all_of(demo_cols))

na_counts <- rowSums(is.na(meps_set[, rx_cols]))
meps_set  <- meps_set[na_counts == 0 | na_counts == length(rx_cols), ]
final_na_counts <- rowSums(is.na(meps_set[, rx_cols]))

meps_set <- meps_set %>%
  mutate(across(all_of(rx_cols), ~ as.character(.))) %>% 
  mutate(across(all_of(rx_cols), ~ ifelse(final_na_counts == length(rx_cols),
                                          "NONE", .)))

# Load synthetic population data (CHANGE BASED ON YOUR COMMUNITY)
#synth_pop <- fread("synthetic_population_clark_county_nv.csv")
synth_pop <- fread("synthetic_population_sandwich_ma.csv")
#synth_pop <- fread("synthetic_population_urbana_champaign_il.csv")

# Update column names in synth_pop to perfectly align with demo_cols
colnames(synth_pop) <- c("Sex", "Race_ethnicity", "Age", "Family_income",
                         "Insurance_coverage")

# ==============================================================================
# STRATEGY 2 IMPLEMENTATION: temporarily bin age and income for better matching
# ==============================================================================
message("Applying Strategy 2: Pre-bracketing continuous variables for high-match rates...")

# 1. Back up the original exact numbers that your ML model requires
synth_pop[, Original_Age := Age]
synth_pop[, Original_Income := Family_income]

# 2. Convert synth_pop numeric values into broader demographic buckets
synth_pop[, Age := cut(as.numeric(Original_Age), 
                       breaks = c(-1, 2, 12, 17, 29, 44, 64, 79, Inf), 
                       labels = c("Infant", "Child", "Teen", "Young_Adult",
                                  "Adult", "Middle_Aged", "Senior", "Elderly"))]

synth_pop[, Family_income := cut(as.numeric(Original_Income), 
                                 breaks = c(-Inf, 15000, 35000, 75000, 150000,
                                            Inf), 
                                 labels = c("Poor", "Low", "Middle", "High", 
                                            "Very_High"))]

# 3. Convert meps_set numeric values into matching buckets
meps_set <- as.data.table(meps_set)
meps_set[, Age := cut(as.numeric(Age), 
                      breaks = c(-1, 2, 12, 17, 29, 44, 64, 79, Inf), 
                      labels = c("Infant", "Child", "Teen", "Young_Adult",
                                 "Adult", "Middle_Aged", "Senior", "Elderly"))]

meps_set[, Family_income := cut(as.numeric(Family_income), 
                                breaks = c(-Inf, 15000, 35000, 75000, 150000,
                                           Inf), 
                                labels = c("Poor", "Low", "Middle", "High",
                                           "Very_High"))]

# ==============================================================================
# 2. POLYPHARMACY ALIGNMENT FUNCTION (Upgraded Architecture)
# ==============================================================================
assign_polypharmacy_rx <- function(df_ref, df_target, demo_cols, rx_cols, id_col) {
  
  # Force inputs to pure data.tables
  df_ref <- as.data.table(df_ref)
  df_target <- as.data.table(df_target)
  
  df_target[, Synth_Person_ID := .I]
  
  for (col in demo_cols) {
    df_ref[, (col) := as.character(get(col))]
    df_target[, (col) := as.character(get(col))]
  }
  
  # Clean and parse Day_Supply into a clean numeric column
  df_ref[, Clean_Days := as.numeric(gsub("[^0-9]", "", Day_Supply))]
  df_ref[is.na(Clean_Days) | Clean_Days <= 0, Clean_Days := 30] 
  
  # ==============================================================================
  # STEP A: POINT-IN-TIME TIMELINE CONCURRENCY ENGINE
  # ==============================================================================
  message("Building point-in-time overlap matrices from longitudinal data...")
  
  set.seed(42) 
  
  df_ref[, Start_Day := sample(1:335, .N, replace = TRUE)] 
  df_ref[, End_Day := Start_Day + Clean_Days]
  
  unique_meps_people = unique(df_ref[, c(id_col, demo_cols), with = FALSE])
  unique_meps_people[, Grab_Day := sample(1:365, .N, replace = TRUE)]
  
  df_ref[unique_meps_people, Grab_Day := i.Grab_Day, on = id_col]
  
  df_ref[, Is_Active := (Grab_Day >= Start_Day & Grab_Day <= End_Day)]
  
  active_ref <- df_ref[Is_Active == TRUE]
  
  people_with_drugs <- unique(active_ref[[id_col]])
  zero_drug_people <- unique_meps_people[!get(id_col) %in% people_with_drugs]
  
  # ==============================================================================
  # STEP B: COMPUTE TRUE POINT-IN-TIME DISTRIBUTIONS
  # ==============================================================================
  message("Calculating point-in-time co-occurrence probabilities...")
  
  pit_counts <- active_ref[, .(num_drugs = .N), by = c(id_col, demo_cols)]
  
  if(nrow(zero_drug_people) > 0) {
    zero_counts <- zero_drug_people[, .(num_drugs = 0), by = c(id_col, demo_cols)]
    pit_counts <- rbind(pit_counts, zero_counts)
  }
  
  num_drugs_dist <- pit_counts[, .(n = .N), by = c(demo_cols, "num_drugs")]
  setorderv(num_drugs_dist, "num_drugs")
  num_drugs_dist[, prob := n / sum(n), by = demo_cols]
  num_drugs_dist[, cum_prob := cumsum(prob), by = demo_cols]
  
  global_counts_dist <- pit_counts[, .(n = .N), by = num_drugs]
  setorderv(global_counts_dist, "num_drugs")
  global_counts_dist[, prob := n / sum(n)]
  global_counts_dist[, cum_prob := cumsum(prob)]
  
  # ==============================================================================
  # STEP C: ASSIGN POINT-IN-TIME COUNTS TO SYNTHETIC POPULATION
  # ==============================================================================
  message("Assigning point-in-time drug volume to synthetic population...")
  
  unique_targets <- unique(df_target[, ..demo_cols])
  unique_targets[, Profile_ID := .I]
  unique_targets[, rand := runif(.N)]
  
  target_volume_lookup <- merge(unique_targets, num_drugs_dist, by = demo_cols,
                                all.x = TRUE, allow.cartesian = TRUE)
  
  matched_profiles <- target_volume_lookup[!is.na(num_drugs)][rand <= cum_prob, 
                                                              .SD[1], 
                                                              by = Profile_ID][, .(Profile_ID, num_drugs)]
  
  unmatched_p_ids <- setdiff(unique_targets$Profile_ID, matched_profiles$Profile_ID)
  unmatched_profiles <- data.table()
  if(length(unmatched_p_ids) > 0) {
    fallback_seeds <- unique_targets[Profile_ID %in% unmatched_p_ids,
                                     .(Profile_ID, rand)]
    global_indices <- findInterval(fallback_seeds$rand, c(0,
                                                          global_counts_dist$cum_prob))
    global_indices[global_indices == 0] <- 1
    global_indices[global_indices > nrow(global_counts_dist)] <- nrow(global_counts_dist)
    
    unmatched_profiles <- data.table(
      Profile_ID = fallback_seeds$Profile_ID,
      num_drugs = global_counts_dist$num_drugs[global_indices]
    )
  }
  
  profile_counts <- rbind(matched_profiles, unmatched_profiles)
  unique_targets <- merge(unique_targets, profile_counts,
                          by = "Profile_ID", all.x = TRUE)
  
  df_target <- merge(df_target, unique_targets[, c(demo_cols, "num_drugs"), 
                                               with = FALSE], by = demo_cols,
                     all.x = TRUE)
  df_target[is.na(num_drugs), num_drugs := 0]
  
  zero_drug_population <- df_target[num_drugs == 0]
  active_target_population <- df_target[num_drugs > 0]
  
  if(nrow(active_target_population) == 0) {
    if(nrow(zero_drug_population) > 0) {
      for(col in rx_cols) zero_drug_population[, (col) := "NONE"]
    }
    return(zero_drug_population)
  }
  
  message("Expanding dataset structure for active point-in-time users...")
  df_expanded <- active_target_population[rep(1:.N, num_drugs)]
  df_expanded[, Row_ID := .I]
  
  # ==============================================================================
  # STEP D: ASSIGN TRUE POINT-IN-TIME RX FEATURE BLOCKS
  # ==============================================================================
  message("Sampling active co-occurring point-in-time feature blocks...")
  
  rx_joint_lookup <- active_ref[, .(n = .N), by = c(demo_cols, rx_cols)]
  setorderv(rx_joint_lookup, rx_cols) 
  rx_joint_lookup[, prob := n / sum(n), by = demo_cols]
  
  rx_profiles <- unique(active_ref[, ..rx_cols])
  rx_profiles[, rx_profile_id := .I]
  rx_joint_lookup <- merge(rx_joint_lookup, rx_profiles, by = rx_cols)
  rx_joint_lookup[, cum_prob := cumsum(prob), by = demo_cols]
  
  global_rx_dist <- active_ref[, .(n = .N), by = rx_cols]
  setorderv(global_rx_dist, rx_cols)
  global_rx_dist <- merge(global_rx_dist, rx_profiles,
                          by = rx_cols)[, prob := n / sum(n)]
  global_rx_dist[, cum_prob := cumsum(prob)]
  
  df_expanded[, rand_rx := runif(.N)]
  expanded_working <- df_expanded[, c("Row_ID", "rand_rx", demo_cols),
                                  with = FALSE]
  
  expanded_rx_lookup <- merge(expanded_working, rx_joint_lookup,
                              by = demo_cols, all.x = TRUE, allow.cartesian = TRUE)
  matched_rx <- expanded_rx_lookup[!is.na(rx_profile_id)][rand_rx <= cum_prob,
                                                          .SD[1], by = Row_ID][, .(Row_ID, rx_profile_id)]
  
  unmatched_row_ids <- setdiff(df_expanded$Row_ID, matched_rx$Row_ID)
  unmatched_rx <- data.table()
  if(length(unmatched_row_ids) > 0) {
    fallback_row_seeds <- df_expanded[Row_ID %in% unmatched_row_ids,
                                      .(Row_ID, rand_rx)]
    global_rx_indices <- findInterval(fallback_row_seeds$rand_rx, c(0,
                                                                    global_rx_dist$cum_prob))
    global_rx_indices[global_rx_indices == 0] <- 1
    global_rx_indices[global_rx_indices > nrow(global_rx_dist)] <- nrow(global_rx_dist)
    
    unmatched_rx <- data.table(
      Row_ID = fallback_row_seeds$Row_ID,
      rx_profile_id = global_rx_dist$rx_profile_id[global_rx_indices]
    )
  }
  
  final_profile_assignments <- rbind(matched_rx, unmatched_rx)
  df_expanded <- merge(df_expanded, final_profile_assignments,
                       by = "Row_ID", all.x = TRUE)
  df_expanded[is.na(rx_profile_id), rx_profile_id := 1]
  
  df_expanded <- merge(df_expanded, rx_profiles, by = "rx_profile_id",
                       all.x = TRUE)
  df_expanded[, c("Row_ID", "rx_profile_id", "rand_rx") := NULL]
  
  if(nrow(zero_drug_population) > 0) {
    for(col in rx_cols) zero_drug_population[, (col) := "NONE"]
    df_expanded <- rbind(df_expanded, zero_drug_population, fill = TRUE)
  }
  
  # ==============================================================================
  # CRITICAL UPDATE: HARD STRUCTURAL DE-DUPLICATION GUARD
  # ==============================================================================
  message("Executing anti-contamination checks on unmedicated rows...")
  
  # Flag every individual row where all target rx columns equal "NONE"
  df_expanded[, Is_None_Row := (rowSums(df_expanded[, ..rx_cols] == "NONE") == length(rx_cols))]
  
  # Calculate how many rows total each individual person has in the combined pool
  df_expanded[, Total_Person_Rows := .N, by = Synth_Person_ID]
  
  # Drop "NONE" rows ONLY if the person has multiple rows (meaning they have real drugs elsewhere)
  df_expanded <- df_expanded[!(Is_None_Row == TRUE & Total_Person_Rows > 1)]
  
  # Clean up the check markers
  df_expanded[, c("Is_None_Row", "Total_Person_Rows") := NULL]
  
  # Clean up reference attributes
  df_ref[, c("Clean_Days", "Start_Day", "End_Day", "Grab_Day", "Is_Active") := NULL]
  
  message("Point-in-time simulation complete!")
  return(df_expanded)
}

# ==============================================================================
# 3. PROCESSING & EXPORT EXECUTIONS
# ==============================================================================

# Run the point-in-time polypharmacy assignment execution
synth_pop_prescrip <- assign_polypharmacy_rx(
  df_ref    = meps_set, 
  df_target = synth_pop, 
  demo_cols = demo_cols, 
  rx_cols   = rx_cols,
  id_col    = person_id_col
)

# 4. RESTORE EXECUTION SCHEMA FOR YOUR MACHINE LEARNING MODEL
# We substitute the categorical labels back for your exact original values here
synth_pop_final <- synth_pop_prescrip %>%
  mutate(
    Age = as.integer(Original_Age),
    Family_income = as.numeric(Original_Income),
    Observation_ID = row_number()
  ) %>%
  select(Observation_ID, Synth_Person_ID, Age, Sex, Family_income, 
         Insurance_coverage, Race_ethnicity, Quantity, Form, Strength, Day_Supply)

# Clean up structural temporary columns left behind in memory
synth_pop[, c("Original_Age", "Original_Income") := NULL]

# final shuffle to prevent saving patterns used in downstream ML
set.seed(123)
synth_pop_final <- synth_pop_final[sample(1:.N)]

# Save file out to disk (AGAIN, CHANGE FOR YOUR COMMUNITY)
#fwrite(synth_pop_final, "clark_county_nv_demo_rx.csv")
fwrite(synth_pop_final, "sandwich_ma_demo_rx.csv")
#fwrite(synth_pop_final, "urbana_champaign_il_demo_rx.csv")

message("Data processing complete.")

# ==============================================================================
# 4. POST-RUN AUDIT VALIDATION
# ==============================================================================
message("\n=== VERIFICATION: FINAL PROCESSED SYNTHETIC TARGET DISTRIBUTION ===")
person_counts <- as.data.table(synth_pop_final)[, .(row_count = .N),
                                                by = Synth_Person_ID]
single_row_ids <- person_counts[row_count == 1, Synth_Person_ID]
single_row_data <- as.data.table(synth_pop_final)[Synth_Person_ID %in% single_row_ids]

count_none <- nrow(single_row_data[Strength == "NONE"])
count_one  <- length(single_row_ids) - count_none
count_poly <- uniqueN(synth_pop_final$Synth_Person_ID) - length(single_row_ids)
total_p    <- uniqueN(synth_pop_final$Synth_Person_ID)

message("Synthetic 0 Drugs (NONE Base): ", round(count_none / total_p * 100, 2),
        "%")
message("Synthetic 1 Drug  (Active Concur): ", round(count_one / total_p * 100, 2),
        "%")
message("Synthetic 2+ Drugs (Polypharmacy):  ", round(count_poly / total_p * 100, 2),
        "%")
message("====================================================================\n")