################################################################################
# Title: meps_processor_2021.R

# Description:
# Loads, cleans and exports data from the Medical Expenditure Panel Survey
# 2021 Prescribed Medicines File and 2021 Full Year Population Characteristics
# File.

# Functions used: 
#     * remove_drugs():
#       * Input: the full list of drugs and the list of drugs that need to be
#                removed
#       * Output: the truncated drug list

#     * fix_drug_coding():
#       * Input: dataframe, column, column name, incorrect column entry, and
#                correct column entry
#       * Output: dataframe with correct entry in row/column of interest

#     * remove_entries(): 
#       * Input: dataframe, column, incorrect column entry
#       * Output: dataframe with row with incorrect entry removed

################################################################################

# install packages and load libraries
library("stringdist")
library("data.table")
library('dplyr')
library("readxl")
library("stringr")
library("tidyr")
library("purrr")

# load functions library
source("meps_processor_functions.R")

################################################################################
# Load and clean Medical Expenditure Panel Survey (MEPS) 2021 data.
################################################################################

# load MEPS dataset (all columns read in as characters)
meps = read.delim("data/MEPS_data_2021.txt", header=TRUE, sep = "\t",
                  colClasses = "character", dec = ".")

# extract desired data columns
meps = meps %>%
  select(c(DUPERSID, RXDRGNAM:RXDAYSUP, RXSF21X:RXXP21X))

# update to clearer column names
colnames(meps)[1:9] = c("Person ID", "Drug", "NDC", "Quantity", "Form",
                        "Form Units", "Strength", "Units", "Day_Supply")
colnames(meps)[10:20] = c("Amount paid self or family",
                          "Amount paid Medicare",
                          "Amount paid Medicaid",
                          "Amount paid private insurance",
                          "Amount paid veterans/CHAMPVA",
                          "Amount paid Tricare",
                          "Amount paid other federal",
                          "Amount paid state or local",
                          "Amount paid worker's comp",
                          "Amount paid other insurance",
                          "Sum of all payments")

# replace missing values (-15, -8, and -7 in original data) with NA
# make sure all correctly imputed
meps = meps %>%
  mutate(across(colnames(meps), ~ case_when(. =="-15" ~ NA,
                                            . =="-8" ~ NA,
                                            . == "-7" ~ NA,
                                            .=="-14" ~ NA,
                                            .=="-1" ~ NA,
                                            .default = .)))

# remove whitespace around entries
meps = meps %>%
  mutate(across(everything(), ~trimws(.)))

# remove unnecessary commas from strength, drug
meps$Drug = gsub(",", "", meps$Drug)
meps$Strength = gsub(",", "", meps$Strength)

# remove administration route labels from drug names (this prevents
# different forms of the same drug from being counted twice) and add to own
# column instead - ophthalmic, otic, nasal, topical usually
meps = meps %>%
  mutate(Administration_route = case_when(grepl("OPHTHALMIC", Drug)==TRUE ~ "OPHTHALMIC",
                                          grepl("OTIC", Drug)==TRUE ~ "OTIC",
                                          grepl("NASAL",Drug)==TRUE ~ "NASAL",
                                          grepl("TOPICAL",Drug)==TRUE ~ "TOPICAL",
                                          grepl("ORAL",Drug)==TRUE ~ "ORAL",
                                          grepl("SUBLINGUAL",Drug)==TRUE ~ "SUBLINGUAL",
                                          grepl("BUCCAL",Drug)==TRUE ~ "BUCCAL",
                                          grepl("RECTAL",Drug)==TRUE ~ "RECTAL",
                                          grepl("INTRAVENOUS",Drug)==TRUE ~ "INTRAVENOUS",
                                          grepl(" IV",Drug)==TRUE ~ "IV",
                                          grepl("IV ",Drug)==TRUE ~ "IV",
                                          grepl("INTRAMUSCULAR",Drug)==TRUE ~ "INTRAMUSCULAR",
                                          grepl("SUBCUTANEOUS",Drug)==TRUE ~ "SUBCUTANEOUS",
                                          grepl("INTRANASAL",Drug)==TRUE ~ "INTRANASAL",
                                          grepl("INHALED",Drug)==TRUE ~ "INHALED",
                                          grepl("INHALATIONAL",Drug)==TRUE ~ "INHALATIONAL",
                                          grepl("INHALANT",Drug)==TRUE ~ "INHALANT",
                                          grepl("VAGINAL",Drug)==TRUE ~ "VAGINAL",
                                          grepl("TRANSDERMAL",Drug)==TRUE ~ "TRANSDERMAL",
                                          .default=NA))

meps$Drug = gsub(paste0("\\b", "OPHTHALMIC", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "OTIC", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "NASAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "TOPICAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "ORAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "SUBLINGUAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "BUCCAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "RECTAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "INTRAVENOUS", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "IV ", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "INTRAMUSCULAR", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "SUBCUTANEOUS", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "INTRANASAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "INHALANT", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "INHALED", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "INHALATIONAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "VAGINAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "TRANSDERMAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)

# group and remove commonly miscoded drug units
# TODO: will need to check for others in any new dataset
miscoded_units = c("OTHER", "U/ML", "U/ML/U/ML", "UNIT", "UNIT/ML", "UNIT/GM",
                   "U/GM", "UT/ML", "MCG/OTHER", "%/OTHER","DOSE","")
for (unit in miscoded_units){
  meps = remove_entries(meps, meps$Units, unit)
}

# remove unnecessary info from units and/or fix spelling/punctuation to be
# standard across all entries
fix_units_df = data.frame(miscoded_units = c("MCG/INH", "MG/ACT", "MCG/mg/Act",
                                             "MG/GM", "mg/Act", "GM/SCOOP",
                                             "MG/mg/Act", "MCG/BLIST","MCG/SPRAY",
                                             "MG/SPRAY", "MCG/ACT", "GM", "GM/ML",
                                             "MCG/MCG/ACT","MCG/DOSE","MG/MG/ML"),
                          correct_units = c("MCG", "MG", "MCG/MG", "MG", "MG",
                                            "G","MG/MG", "MCG", "MCG", "MG", "MCG",
                                            "G", "G/ML", "MCG/MCG","MCG","MG/MG/MG"))
for (i in seq_len(nrow(fix_units_df))){
  meps = fix_drug_coding(meps, meps$Units, "Units", fix_units_df[i,1],
                         fix_units_df[i,2])
}

# make Form variable clearer (assume extended release not relevant for simplicity)
meps = meps %>%
  mutate(Form = case_when(Form=="CAP-Capsule" ~ "CAPSULE",
                          Form=="INH-Inahler" ~ "INHALER",
                          Form=="INH-Inhalant" ~ "INHALER",
                          Form=="CAP-CAPLETS" ~ "CAPSULE",
                          Form=="TAB" ~ "TABLET",
                          Form=="TABS" ~ "TABLET",
                          Form=="SUSP" ~ "SUSPENSION",
                          Form=="CPDR" ~ "CAPSULE",
                          Form=="SHA" ~ "SHAMPOO",
                          Form=="CAPS"~"CAPSULE",
                          Form=="SOLN" ~ "SOLUTION",
                          Form=="TB12" ~ "TABLET",
                          Form=="TBEC" ~ "TABLET",
                          Form=="TB24"~"TABLET",
                          Form=="PACK" ~ "PACKET",
                          Form=="Patches" ~ "PATCH",
                          Form=="CREA" ~ "CREAM",
                          Form=="Capsule" ~ "CAPSULE",
                          Form=="TBCR" ~ "TABLET",
                          Form=="SYR" ~ "SYRINGE",
                          Form=="SOL" ~ "SOLUTION",
                          Form=="Inhaler" ~ "INHALER",
                          Form=="AERS" ~ "AEROSOL",
                          Form=="NEBU" ~ "NEBULIZED SOLUTION",
                          Form=="LIQD" ~ "LIQUID",
                          Form=="LPOP" ~"LOLLIPOP",
                          Form=="SYRP" ~ "SYRUP",
                          Form=="CRE" ~ "CREAM",
                          Form=="SUSR" ~ "SUSPENSION",
                          Form=="AERO" ~ "AEROSOL",
                          Form=="TBDP" ~ "TABLET",
                          Form=="CPEP" ~ "CAPSULE",
                          Form=="OINT" ~ "OINTMENT",
                          Form=="PTCH" ~ "PATCH",
                          Form=="CPSP" ~ "CAPSULE",
                          Form=="CP24" ~ "CAPSULE",
                          Form=="SUS" ~ "SUSPENSION",
                          Form=="SUPP" ~ "SUPPOSITORY",
                          Form=="CHEW" ~ "CHEW",
                          Form=="SOLR" ~ "SOLUTION",
                          Form=="POW" ~ "POWDER",
                          Form=="SUBL" ~ "SUBLINGUAL",
                          Form=="CP12" ~ "CAPSULE",
                          Form=="PTTW" ~ "2 WEEK PATCH",
                          Form=="LIQ" ~ "LIQUID",
                          Form=="Drops" ~ "DROPS",
                          Form=="EMUL" ~ "EMULSION",
                          Form=="PT72" ~ "72 HR PATCH",
                          Form=="LOTN" ~ "LOTION",
                          Form=="CPCR" ~ "CAPSULE",
                          Form=="IV" ~ "IV",
                          Form=="PT24" ~ "24 HR PATCH",
                          Form=="PTWK" ~ "1 WEEK PATCH",
                          Form=="LOT" ~ "LOTION",
                          Form=="INHA" ~ "INHALER",
                          Form=="SHAM" ~ "SHAMPOO",
                          Form=="AEPB" ~ "AEROSOL",
                          Form=="LQCR" ~ "LIQUID",
                          Form=="ELIX" ~ "ELIXIR",
                          Form=="PAS" ~ "PASTE",
                          Form=="PDR" ~ "POWDER",
                          Form=="OIN" ~ "OINMENT",
                          Form=="SOLG" ~ "GEL",
                          Form=="CONC" ~ "CONCENTRATE",
                          Form=="ELI" ~ "ELIXIR",
                          Form=="ECT" ~ "TABLET",
                          Form=="POWD" ~ "POWDER",
                          Form=="INJ" ~ "INJECTABLE",
                          Form=="SUP" ~ "SUPPOSITORY",
                          Form=="Eye drops" ~ "DROPS",
                          Form=="ENEM" ~ "ENEMA",
                          Form=="SPR" ~ "SPIRIT",
                          Form=="PSTE" ~ "PASTE",
                          Form=="Nose drops" ~ "DROPS",
                          Form=="AER" ~ "AEROSOL",
                          Form=="TBEF" ~ "TABLET",
                          Form=="PLLT" ~ "PELLET",
                          Form=="TINC" ~ "TINCTURE",
                          Form=="TROC" ~ "TROCHE",
                          Form=="NEB" ~ "NEBULIZED SOLUTION",
                          Form=="Wash" ~ "WASH",
                          Form=="GRA" ~ "GRANULES",
                          Form=="AERB" ~ "AEROSOL",
                          Form=="INST" ~ "INSERT",
                          Form=="Topical Gel" ~ "GEL",
                          Form=="FOA" ~ "FOAM",
                          Form=="Aerosol" ~ "AEROSOL",
                          Form=="CTB" ~ "TABLET"
                          
  ))

# remove miscoded drugs or drugs with vague/missing info or otherwise miscoded
meps = meps[is.na(meps$Units)==FALSE,]
meps = meps[is.na(meps$Drug)==FALSE,]
meps = remove_entries(meps, meps$Form, "OTHER")
meps = remove_entries(meps, meps$Units, "ML")
meps = remove_entries(meps, meps$Units, "ML/ML")
meps = meps %>% 
  mutate(Units = case_when(NDC=="68788810303" ~ "MG/MG",
                           .default=Units))

# replace unknown form abbreviations by NA or unclear form type
meps <- meps %>%
  mutate(Form = na_if(Form, "SRN")) %>%
  mutate(Form = na_if(Form,"NDL"))

# Remove whitespace
meps = meps %>%
  mutate_all(~str_squish(.))

# fix miscoded units for combination drugs
meps = meps %>%
  mutate(Units = case_when((str_count(Drug, "/") == 1 | str_count(Drug, "-") == 1) & Units=="MG" ~ "MG/MG",
                           (str_count(Drug, "/") == 2 | str_count(Drug, "-") == 2) & Units=="MG" ~ "MG/MG/MG",
                           (str_count(Drug, "/") == 1 | str_count(Drug, "-") == 1) & Units=="MCG" ~ "MCG/MCG",
                           (str_count(Drug, "/") == 2 | str_count(Drug, "-") == 2) & Units=="MCG" ~ "MCG/MCG/MCG",
                           (str_count(Drug, "/") == 1 | str_count(Drug, "-") == 1) & Units=="%" ~ "%/%",
                           (str_count(Drug, "/") == 2 | str_count(Drug, "-") == 2) & Units=="ML" ~ "ML/ML/ML",
                           (str_count(Drug, "/") == 1 | str_count(Drug, "-") == 1) & Units=="G" ~ "G/G",
                           (str_count(Drug, "/") == 2 | str_count(Drug, "-") == 2) & Units=="G" ~ "G/G/G",
                           .default = Units))

# fix specific cases with units failing to update for combo drugs
meps = meps %>%
  mutate(Units = case_when(NDC=="00173088710" ~ "MCG/MCG/MCG",
                           NDC=="00173088714" ~ "MCG/MCG/MCG",
                           .default=Units))

# remove rows where strength has no slash/dash and drug does
meps = meps %>%
  filter(!((grepl("/",Drug)==TRUE | grepl("-",Drug)==TRUE) & (grepl("/",Strength)==FALSE & grepl("-",Strength)==FALSE)))

# remove rows where drug has more delimiters than strength
meps = meps %>%
  filter(Strength != "500-6500")

# Separate combination drugs (and find ones that are miscoded as combination
# drugs)

# fix mismatches between slashes and dashes across Drug, Strength, Units columns
# (that is, make it either all slashes or all dashes)
meps = meps %>%
  mutate(Strength = case_when(grepl("-", Drug)==TRUE &
                                grepl("/", Strength) == TRUE ~
                                str_replace_all(Strength, "/", "-"),
                              .default = Strength))

meps = meps %>%
  mutate(Strength = case_when(grepl("/", Drug)==TRUE &
                                grepl("-", Strength) == TRUE ~
                                str_replace_all(Strength, "-", "/"),
                              .default = Strength))

meps = meps %>%
  mutate(Units = case_when(grepl("-", Drug)==TRUE & grepl("/", Units) == TRUE &
                             grepl("ML",Units)==FALSE & grepl("HR",Units)==FALSE ~ 
                             str_replace_all(Units, "/", "-"),
                           .default = Units))

meps = meps %>%
  mutate(Units = case_when(grepl("/", Drug)==TRUE & grepl("-", Units) == TRUE &
                             grepl("ML",Units)==FALSE & grepl("HR",Units)==FALSE ~
                             str_replace_all(Units, "-", "/"),
                           .default = Units))

# Remove drugs with dashes that are not combination drugs and will be removed
# anyway
drug_names_to_remove <- c("OMEGA-3 POLYUNSATURATED FATTY ACIDS")
meps <- meps %>% filter(!Drug %in% drug_names_to_remove)

# Remove whitespace
meps = meps %>%
  mutate_all(~str_squish(.))

# Handle dash/slash delimiters 

# Make dummy column of all zeros initially
meps = meps %>%
  mutate(Is_Combo = rep(0, nrow(meps)))

# Make a duplicate row with Is_Combo set to 1 if drug is a combination drug

# Identify rows where 'Drug' contains a dash/slash
rows_with_delim = meps %>% filter(grepl("[-/]", Drug))

# Duplicate these rows and set 'Is_Combo' to 1
duplicated_rows = rows_with_delim %>% mutate(Is_Combo = 1)

# Combine the original dataframe with the duplicated rows
meps = bind_rows(meps, duplicated_rows) %>% arrange(row_number())

# Conditionally split the Units column based on the presence of a delimiter in the
# Drug column 
meps = meps %>%
  mutate(
    Units_Parts = if_else(
      grepl("[-/]", Drug),  
      str_split(Units, "[-/]", simplify = TRUE),  
      Units  
    )
  )

# Split the Drug, Strength columns into parts before and after the dash or slash
meps = meps %>%
  mutate(
    Drug_Parts = str_split(Drug, "[-/]", simplify = TRUE),
    Strength_Parts = str_split(Strength, "[-/]", simplify = TRUE)
  )

# Use if_else to conditionally assign the correct part based on the Is_Combo column
meps <- meps %>%
  mutate(
    Drug = if_else(Is_Combo == 0, Drug_Parts[, 1],
                   if_else(Is_Combo == 1, Drug_Parts[, 2], Drug)),
    Strength = if_else(Is_Combo == 0, Strength_Parts[, 1],
                       if_else(Is_Combo == 1, Strength_Parts[, 2], Strength)),
    Units = if_else(Is_Combo == 0, Units_Parts[, 1],
                    if_else(Is_Combo == 1, Units_Parts[, 2], Units))
  )

# Drop the helper columns
meps <- meps %>%
  select(-Is_Combo, -Drug_Parts, -Strength_Parts, -Units_Parts)

# fix units that did not split correctly
meps = meps %>%
  mutate(Units=case_when(Units=="MCG/MCG" ~ "MCG",
                         Units=="MG/MG" ~ "MG",
                         .default=Units))

# Generate drug list to loop through

# get list of all drugs listed in Drug column
drugs = unique(meps$Drug)

# remove vitamins, minerals, compounds from food, etc
food_drugs = unlist(as.list(read_excel("data/drugs_to_remove.xlsx", sheet=1,
                                       col_names=FALSE)))
drugs = remove_drugs(drugs, food_drugs)

# remove hormones and other naturally occurring compounds in the body too
inbody_drugs = unlist(as.list(read_excel("data/drugs_to_remove.xlsx", sheet=2,
                                         col_names=FALSE)))
drugs = remove_drugs(drugs, inbody_drugs)

# remove vague names and non-distinct structures
vague_drugs = unlist(as.list(read_excel("data/drugs_to_remove.xlsx", sheet=3,
                                        col_names=FALSE)))
drugs = remove_drugs(drugs, vague_drugs)

# remove missing drug names from drug list
drugs = na.omit(drugs)

# remove any reintroduced whitespace from drugs vector
drugs = trimws(drugs)

# remove any duplicates introduced
drugs = unique(drugs)

# replace empty strings with NA
meps <- meps %>%
  mutate(across(everything(), ~na_if(., "")))

# ensure all units NA are removed
meps = meps[is.na(meps$Units)==FALSE,]

# initialize larger data frame to store cleaned data
meps_clean = vector("list", length = length(drugs))

# loop through drug list to complete data cleaning on MEPS dataset
for (i in 1:length(drugs)){
  drug_data = meps %>%
    filter(Drug==drugs[i])
  
  # Find avg Day_Supply and Quantity values for each drug (to be used for
  # imputing)
  not_missing_daysup = drug_data[is.na(drug_data$Day_Supply)==FALSE,]
  avg_daysup = sum(as.numeric(not_missing_daysup$Day_Supply))/
    length(not_missing_daysup$Day_Supply)
  not_missing_quanty = drug_data[is.na(drug_data$Quantity)==FALSE,]
  avg_quanty = sum(as.numeric(not_missing_quanty$Quantity))/
    length(not_missing_quanty$Quantity)
  not_missing_strength = drug_data[is.na(drug_data$Strength)==FALSE,]
  avg_strength = sum(as.numeric(not_missing_strength$Strength))/
    length(not_missing_strength$Strength)
  
  # remove entries where strength is missing for over 95% of entries
  if (sum(is.na(drug_data$Strength)==TRUE) > 0.95*dim(drug_data)[1]){
    drug_data = subset(drug_data, is.na(drug_data$Strength)==FALSE)
  }
  else {
    drug_data = drug_data %>%
      mutate(Strength = case_when(is.na(Strength)==TRUE ~ avg_strength,
                                    .default=as.numeric(Strength)))
  }
  
  # remove entries where Day_Supply missing for >95% of entries (imputing not
  # reliable in this case)
  if (sum(is.na(drug_data$Day_Supply)==TRUE) > 0.95*dim(drug_data)[1]){
    drug_data = subset(drug_data, is.na(drug_data$Day_Supply)==FALSE)
  }
  else {
    drug_data = drug_data %>%
      mutate(Day_Supply = case_when(is.na(Day_Supply)==TRUE ~ avg_daysup,
                                    .default=as.numeric(Day_Supply)))
  }
  
  # remove drugs where Quantity missing for >95% of entries (imputing not
  # reliable)
  if (sum(is.na(drug_data$Quantity)==TRUE) > 0.95*dim(drug_data)[1]){
    drug_data = subset(drug_data, is.na(drug_data$Quantity)==FALSE)
  }
  else {
    drug_data = drug_data %>%
      mutate(Quantity = case_when(is.na(Quantity)==TRUE ~ avg_quanty,
                                  .default=as.numeric(Quantity)))
  }
  
  # convert strength, Day_Supply, quantity to correct data type
  drug_data = drug_data %>%
    mutate(Day_Supply = as.numeric(Day_Supply)) %>%
    mutate(Quantity = as.numeric(Quantity)) %>%
    mutate(Strength = as.numeric(Strength))
  
  # Conversions to correct units in Strength column (assume in 1mL for solution
  # conversions)
  
  # convert MG to MCG
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units == "MG" ~ Strength*1000,
                                .default=Strength))
  
  # convert MG/ML to MCG (mass and volume often directly given (e.g. 3 mg/20 mL);
  # if volume not given, assuming 1 mL)
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units == "MG/ML" ~ Strength*1000,
                                .default=Strength))
  
  # convert MCG/HR to MCG
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="MCG/HR" ~ Strength*24,
                                .default=Strength))
  
  # Convert % to MCG (1% solution means 1 g of drug per 100 mL of solution - for
  # example, 1% solution is 1 g drug per 100 mL solvent)
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units == "%" ~ Strength*1000000,
                                .default=Strength))
  
  # Convert G/ML to MCG (mass and volume often directly given (e.g. 3 g/20 mL);
  # if volume not given, assuming 1 mL)
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units == "G/ML" ~ Strength*1000000,
                                .default=Strength))
  
  # Convert G to MCG
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="G" ~ Strength*1000000,
                                .default=Strength))
  
  # Convert MG/24HR to MCG
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="MG/24HR" ~ Strength*1000,
                                .default=Strength))
  
  # Convert MG/HR to MCG
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="MG/HR" ~ Strength*24*1000,
                                .default=Strength))
  
  # Convert IU to MCG (100 IU insulin glargine is 3.64 mcg)
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="IU" & Drug=="INSULIN GLARGINE" ~ Strength*3.64/100,
                                .default=Strength))
  
  # change all remaining units to MCG (everything is in MCG now, MCG is
  # micrograms or ug)
  drug_data = drug_data %>%
    mutate(Units = "MCG")
  
  # add cleaned data for each drug to its own data frame
  meps_clean[[i]] = drug_data
}

# bind each data frame for each drug into giant data frame
meps_clean = do.call(rbind, meps_clean)

# remove Units (all same units now)
meps_clean = subset(meps_clean, select=-c(Units))

# calculate daily frequency for each record
meps_clean = meps_clean %>%
  mutate(Daily_Frequency = Quantity/Day_Supply)

# calculate daily dosage for each record (Strength is mass of active ingredient)
meps_clean = meps_clean %>%
  mutate(Daily_Dosage = Strength*Daily_Frequency)

# Add year for integration
meps_clean = meps_clean %>%
  mutate(Year="2021")

# Load 2021 population data
meps_pop = read.delim("data/MEPS_pop_data_2021.txt", header=TRUE, sep = "\t",
                      colClasses = "character", dec = ".")

# prep meps_pop to link to meps_clean (note: PMED is prescription medication)
meps_pop = meps_pop %>%
  rename(`Person ID` = DUPERSID) %>%
  select(c("Person ID","AGE21X", "SEX", "AFRDPM42", "TTLP21X", "FAMINC21",
           "POVCAT21", "INSURC21","PMEDIN31","PMEDIN42","PMEDIN53","RACETHX",
           "RACEV2X"))

# handle missing values (all are negative valued reserved codes; -2, -10, -13
# not present in selected data)
meps_pop <- meps_pop %>%
  mutate(across(colnames(meps_pop), ~ case_when(
    as.character(.) %in% c("-15", "-8", "-7", "-1") ~ NA,  
    suppressWarnings(as.numeric(.)) %in% c(-15, -8, -7, -1) ~ NA,  
    TRUE ~ .  
  )))

# impute with average age grouped by gender/race/ethnicity
average_age_by_group <- meps_pop %>%
  group_by(SEX, RACEV2X, RACETHX) %>%
  summarize(mean_age = mean(as.numeric(AGE21X), na.rm = TRUE), .groups = 'drop')

meps_pop <- meps_pop %>%
  left_join(average_age_by_group, by = c("SEX", "RACEV2X", "RACETHX")) %>%
  mutate(AGE21X = ifelse(is.na(AGE21X), mean_age, AGE21X)) %>%
  select(-mean_age)

# impute with mode for AFRDPM42 (since one of the two categories dominates the
# other)
meps_pop <- meps_pop %>%
  mutate(AFRDPM42 = ifelse(is.na(AFRDPM42),
                           as.numeric(names(which.max(table(meps_pop$AFRDPM42,
                                                            useNA = "no")))),
                           AFRDPM42))

# impute with average for PMED31, PMED42, PMED53 and then average into 1 PMED column
# (note: the PMED_insurance column will be of data type numeric)
meps_pop$PMEDIN31[is.na(meps_pop$PMEDIN31)] <- mean(as.numeric(meps_pop$PMEDIN31),
                                                    na.rm = TRUE)
meps_pop$PMEDIN42[is.na(meps_pop$PMEDIN42)] <- mean(as.numeric(meps_pop$PMEDIN42),
                                                    na.rm = TRUE)
meps_pop$PMEDIN53[is.na(meps_pop$PMEDIN53)] <- mean(as.numeric(meps_pop$PMEDIN53),
                                                    na.rm = TRUE)
meps_pop = meps_pop %>%
  mutate(PMEDIN31 = as.numeric(PMEDIN31)) %>%
  mutate(PMEDIN42 = as.numeric(PMEDIN42)) %>%
  mutate(PMEDIN53 = as.numeric(PMEDIN53))
meps_pop$PMED_insurance <- rowMeans(meps_pop[, c("PMEDIN31", "PMEDIN42",
                                                 "PMEDIN53")])
# fix order and colnames
meps_pop = meps_pop %>%
  select(-c(PMEDIN31,PMEDIN42,PMEDIN53)) %>%
  relocate(RACETHX, .before=AFRDPM42) %>%
  relocate(RACEV2X, .after=RACETHX) %>%
  relocate(AFRDPM42, .after=POVCAT21)
colnames(meps_pop) = c("Person ID", "Age", "Sex", "Ethnicity", "Race",
                       "Personal_income","Family_income","Poverty_category",
                       "Can't_afford_PMED", "Insurance_coverage",
                       "PMED_insurance_coverage")

# remove negative values from income columns (zeros allowed)
meps_pop = meps_pop %>%
  filter(Personal_income >= 0) %>%
  filter(Family_income >= 0)

# Link by Person ID (retaining people in "No drug" category too)
meps_clean = merge(meps_clean,meps_pop,by="Person ID",all=TRUE)
meps_clean = meps_clean %>%
  mutate(Drug = ifelse(is.na(Drug), "No drugs", Drug))

# Add column for concentration calculation (assuming sewershed size is 100,000
# people, 310.404 L wastewater per person, divide by 4 to account for medication
# noncompliance and non-resident sewer flow)
meps_clean = meps_clean %>%
  mutate(Concentration = Daily_Dosage/(100000*310.404*4))

# export data as csv
write.csv(meps_clean,"data/meps_clean_2021.csv")