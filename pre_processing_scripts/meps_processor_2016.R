################################################################################
# Title: meps_processor_2016.R

# Description:
# Loads, cleans and exports data from the Medical Expenditure Panel Survey
# 2016 Prescribed Medicines File and 2016 Full Year Population Characteristics
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
# Load and clean Medical Expenditure Panel Survey (MEPS) 2016 prescription data.
################################################################################

# load MEPS dataset (all columns read in as characters)
meps = read.delim("data/MEPS_data_2016.txt", header=TRUE, sep = "\t",
                  colClasses = "character", dec = ".")

# extract desired data columns
meps = meps %>%
  select(c(DUPERSID, PANEL, RXDRGNAM:RXDAYSUP, TC1, TC2, TC3))

# create unique Person ID 
meps = meps %>%
  mutate(Person_ID = paste0(DUPERSID, PANEL)) %>%
  select(-c(DUPERSID, PANEL)) %>%
  relocate(Person_ID, .before=RXDRGNAM)

# update to clearer column names
colnames(meps)[1:12] = c("Person_ID", "Drug", "NDC", "Quantity", "Form",
                        "Form Units", "Strength", "Units", "Day_Supply",
                        "Therapeutic_class_1", "Therapeutic_class_2",
                        "Therapeutic_class_3")

# replace missing values with NA
meps <- meps %>%
  mutate(across(colnames(meps), ~ case_when(
    as.character(.) %in% c("-15", "-9", "-8", "-7", "-14", "-1") ~ NA,  
    suppressWarnings(as.numeric(.)) %in% c(-15, -9, -8, -7, -14, -1) ~ NA,  
    TRUE ~ .  
  )))

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
miscoded_units = c("OTHER", "U/ML", "U/ML/U/ML", "UNIT", "UNIT/ML", "UNIT/GM",
                   "U/GM", "UT/ML", "MCG/OTHER", "%/OTHER","DOSE","","UNIT/ML",
                   "OTHER/OTHER", "UNT/ML", "U/ML/ML")
for (unit in miscoded_units){
  meps = remove_entries(meps, meps$Units, unit)
}

# fix dataset specific miscoding
meps = meps %>%
  mutate(Drug = case_when(Drug=="ETHINYL ESTRADIOL-NORGESTIMATE" ~ "NORGESTIMATE-ETHINYL ESTRADIOL",
                          .default=Drug))

meps = meps %>%
  mutate(Strength = case_when(NDC=="54868619600" ~ "10/8",
                              .default=Strength)) %>%
  mutate(Units = case_when(NDC=="54868619600" ~ "MG/MG",
                           .default=Units))

meps <- meps %>%
  mutate(Drug = str_trim(Drug))

# remove unnecessary info from units and/or fix spelling/punctuation to be
# standard across all entries
fix_units_df = data.frame(miscoded_units = c("MCG/INH", "MG/ACT", "MCG/mg/Act",
                                             "MG/GM", "mg/Act", "GM/SCOOP",
                                             "MG/mg/Act", "MCG/BLIST","MCG/SPRAY",
                                             "MG/SPRAY", "MCG/ACT", "GM", "GM/ML",
                                             "MCG/MCG/ACT","MCG/DOSE","MCG/MCG/DOSE",
                                             "MG/MG/ML","MG/ML/ML","MG/ML/MG/ML",
                                             "MG/MCG/MG/MCG","MG/MG/MG","GM/DOSE"),
                          correct_units = c("MCG", "MG", "MCG/MG", "MG", "MG",
                                            "G","MG/MG", "MCG", "MCG", "MG", "MCG",
                                            "G", "G/ML", "MCG/MCG","MCG","MCG/MCG",
                                            "MG/MG","MG","MG/MG","MG/MCG","MG/MG",
                                            "G"))
for (i in seq_len(nrow(fix_units_df))){
  meps = fix_drug_coding(meps, meps$Units, "Units", fix_units_df[i,1],
                         fix_units_df[i,2])
}

# remove miscoded drugs or drugs with vague/missing info
meps = meps[is.na(meps$Units)==FALSE,]
meps = meps[is.na(meps$Drug)==FALSE,]
meps = remove_entries(meps, meps$Form, "OTHER")
meps = remove_entries(meps, meps$Units, "ML")
meps = remove_entries(meps, meps$NDC,"53002090072") #inaccurate/incomplete for
                                                    #this dataset
meps = remove_entries(meps, meps$NDC,"16590016705") #same reason here and below
meps = remove_entries(meps, meps$NDC, "69543025204")
meps = remove_entries(meps, meps$NDC, "63187004004")
meps = remove_entries(meps, meps$NDC, "00603085594")

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
                           (str_count(Drug, "/") == 1 | str_count(Drug, "-") == 1) & Units=="ML" ~ "ML/ML",
                           (str_count(Drug, "/") == 1 | str_count(Drug, "-") == 1) & Units=="IU" ~ "IU/IU",
                           .default = Units))

# remove rows where strength has no slash/dash and drug does
meps = meps %>%
  filter(!((grepl("/",Drug)==TRUE | grepl("-",Drug)==TRUE) & (grepl("/",Strength)==FALSE & grepl("-",Strength)==FALSE)))

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
drug_names_to_remove <- c("OMEGA-3 POLYUNSATURATED FATTY ACIDS",
                          "ALPHA-LIPOIC ACID","L-METHYLFOLATE")
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
  mutate(Units=case_when(Units=="MG/G" ~ "MG",
                         Units=="%/%" ~ "%",
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

# replace empty strings with NA
meps <- meps %>%
  mutate(across(everything(), ~na_if(., "")))

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

# ensure updated drug list
drugs = unique(drugs)

# initialize larger data frame to store cleaned data
meps_clean = vector("list", length = length(drugs))

# loop through drug list to complete data cleaning on MEPS dataset
for (i in 1:length(drugs)){
  drug_data = meps %>%
    filter(Drug==drugs[i])
  
  # if drug missing, skip to next drug
  if (nrow(drug_data) == 0) next
  
  # remove entries where strength is missing for over 95% of entries
  missing_pct_strength <- sum(is.na(drug_data$Strength)) / nrow(drug_data)
  if (missing_pct_strength > 0.95) {
    drug_data = drug_data[!is.na(drug_data$Strength), ]} 
  
  # if drug missing, skip to next drug
  if (nrow(drug_data) == 0) next
  
  # remove entries where Day_Supply missing for >95% of entries (imputing not
  # reliable in this case)
  missing_pct_daysupp <- sum(is.na(drug_data$Day_Supply)) / nrow(drug_data)
  if (missing_pct_daysupp > 0.95) {
    drug_data = drug_data[!is.na(drug_data$Day_Supply), ]} 
  
  # if drug missing, skip to next drug
  if (nrow(drug_data) == 0) next
  
  # remove drugs where Quantity missing for >95% of entries (imputing not
  # reliable)
  missing_pct_quant <- sum(is.na(drug_data$Quantity)) / nrow(drug_data)
  if (missing_pct_quant > 0.95) {
    drug_data = drug_data[!is.na(drug_data$Quantity), ]} 
  
  # if drug missing, skip to next drug
  if (nrow(drug_data) == 0) next
  
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
  
  # Convert MG/9HR to MCG
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="MG/9HR" ~ Strength*(24/9)*1000,
                                .default=Strength))
  
  # Convert IU to MCG (100 IU insulin glargine is 3.64 mcg;
  # 1 IU nystatin is 0.000333mg; 100 IU insulin aspart is 3.5 mg; 100 IU insulin
  # detemir is 14.2 mg) - IU/ML is done the same as MG/ML, so can be converted 
  # directly as though it's from IU alone
  drug_data = drug_data %>%
    mutate(Strength = case_when((Units=="IU" | Units=="IU/ML") & Drug=="INSULIN GLARGINE" ~ Strength*3.64/100,
                                Units=="IU" & Drug=="NYSTATIN" ~ Strength*0.000333*1000,
                                (Units=="IU" | Units=="IU/ML") & Drug=="INSULIN ASPART" ~ (Strength*3.5*1000)/100,
                                (Units=="IU" | Units=="IU/ML") & Drug=="INSULIN DETEMIR" ~ (Strength*14.2*1000)/100,
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

# drug list common to all 6 years
common_drugs = read.csv("common_drugs.csv")
common_drugs = as.list(common_drugs)
common_drugs <- as.character(common_drugs$Drug)
common_drugs <- trimws(common_drugs)
meps_clean$Drug <- trimws(as.character(meps_clean$Drug))

# filter meps_clean for only common drugs
meps_clean = meps_clean %>%
  filter(Drug %in% common_drugs)

# read in openFDA API's Form results
api_results = read.csv("meps_2016_routes.csv")
api_results$NDC <- as.character(api_results$NDC)
api_results = api_results %>%
  select(c("NDC", "route")) %>%
  distinct(NDC, .keep_all = TRUE)

# combine with API results
meps_clean = meps_clean %>%
  left_join(api_results, by="NDC") %>%
  mutate(Form = if_else(!is.na(route) & route != "", route, Form))

meps_clean <- meps_clean %>%
  mutate(Form = case_when(
    Form %in% c("TABS", "TAB", "SYRP", "CAPS", "SYR", "CAP-Capsule", "CP24", "TB24",
                "T12A", "CHEW", "ELI", "CTB", "TBEC", "TBDP", "TB12", "TBCR",
                "CAP-Caplets", "CP12", "CPSP", "CPDR", "CPEP", "CPCR", "LPOP",
                "T24A", "ELIX", "GER", "GUM", "LOZ") ~ "ORAL",
    Form %in% c("SHA", "SHAM", "Topical-Unspecified") ~ "TOPICAL",
    Form %in% c("INH-Inhaler", "INH-Inhalant", "NEBU",
                "RESPIRATORY (INHALATION)") ~ "INHALATION",
    Form %in% c("IV") ~ "INTRAVENOUS",
    Form %in% c("Eye drops") ~ "OPHTHALMIC",
    Form %in% c("IUD") ~ "INTRAUTERINE",
    Administration_route == "TOPICAL" ~ "TOPICAL",
    Administration_route == "NASAL" ~ "NASAL",
    Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Administration_route == "OTIC" ~ "OTIC",
    TRUE ~ Form   
  ))

# change Form variable to reflect full form name for merging with excretion data
# (those that are in the MEPS codebook but have been removed by steps above are not
#accounted for below)
meps_clean <- meps_clean %>%
  mutate(Form = case_when(NDC %in% c("57664014634","00093412573","00093412774",
                                     "00054372263","00054372250","00186062501",
                                     "00536012297","00536012285","59630075050",
                                     "65162069179","00054006447","59762305102",
                                     "00186061001","54838055550","00186401001",
                                     "00054039063","00186402001","66993041630",
                                     "66689040316","63824025412","66220072930",
                                     "64950035450","49884046565","00245003660",
                                     "60432061360","00185094098","49348022834",
                                     "00054372763","65628007010","51552135001",
                                     "54838055840","54868619600","47463079930",
                                     "60432021208","47335071086","00121174410",
                                     "00121075908","00121077504","50383004248",
                                     "00093611816","00603533815","00603533715",
                                     "50383004224","00591544243","00591544221",
                                     "50383004004","00591505221","00591505221",
                                     "68071153208","59746000103","00603459315",
                                     "50383024116","00781502207","59762444002",
                                     "31722057447","00009005604","68001000501",
                                     "51991065116","00095008921","00095008735",
                                     "00472023516","00603138458","00093610812",
                                     "45802068028","51672416108","00406800330",
                                     "54838055240","54838057280","00603085794",
                                     "00121057716","50383077916","00603137858",
                                     "50383079516","50383077932","00603137859",
                                     "60432003816","60432003864","50458059601",
                                     "00121057616","00245107230","00378615093",
                                     "00378615010","62175011841","60505006500",
                                     "55045307101","62037064001","68134036301",
                                     "68134020116","00472127016","00472127094",
                                     "00472125594","00472126194","00113089726",
                                     "45802020326","00603084154","00603084254",
                                     "66993047173","40085084296","00536100597",
                                     "42211010111","00078035752","65162064978",
                                     "00781627043","59762312001","00093202623",
                                     "59762314001","00093202694","00093202631",
                                     "00093202723","59762311001","59762313001",
                                     "68180012401","00093417773","00093417774",
                                     "68180012402","54569102400","00093417573",
                                     "42043014358","00093415580","00781615746",
                                     "00093416173","00093416178","00143988775",
                                     "00143988701","00143988801","00143988750",
                                     "00143988915","00093415573","00143988901",
                                     "00781615752","00143988601","00781615757",
                                     "00143988980","00781603955","00781604155",
                                     "00093416078","00093416176","00781604158",
                                     "00093415579","00143988650","00093415073",
                                     "00781603958","00781604146","00093415080",
                                     "43598020750","43598020752","24478020020",
                                     "24478019010","50419077701","68180039301",
                                     "50419077301","16714039201","68180072210",
                                     "68180072320","67253001341","00781607861",
                                     "67253001342","42043025167","68180072220",
                                     "42043025138","16714039302","42043025238",
                                     "16714039202","16714039301","00781607846",
                                     "42043025267","68180072310","62559044001",
                                     "68788910901","23490574100","68094058758",
                                     "68094058862","00121178105","49999058500",
                                     "43598020751","23490731201","63629149004",
                                     "00603083958","58657052016","57896018016",
                                     "00603082358","00603082394","00536077085",
                                     "50580013444","00006008062","68682070690",
                                     "00054353244","59762494001","54569606200",
                                     "55048002730","68382010601","00781224301",
                                     "55111053201","59762001601","68788899201",
                                     "65224032060","43376031060","00574012901") ~ "ORAL",
    NDC %in% c("49884093547","12547017157") ~ "TOPICAL",
    NDC %in% c("00378087399","00555101016","00378087299","00591351004","00378087116",
               "00378334053","47781029703","47781029803","00378911293",
               "49730011130") ~ "TRANSDERMAL",
    NDC %in% c("63323012310") ~ "INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTERIAL",
    NDC %in% c("00703367103") ~ "INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTERIAL/INTRATHECAL",
    NDC %in% c("61703035038") ~ "INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS",
    NDC %in% c("64679072801","00173047900","00781317207","54436002004","00169406012",
               "00169406013") ~ "SUBCUTANEOUS",
    NDC %in% c("00009307303") ~ "INTRAMUSCULAR/INTRA-ARTICULAR/SOFT TISSUE/INTRALESIONAL",
    NDC %in% c("00487950101","00487950103","00487950160","00378827093",
               "00378797052","00487980101","00487980125","00487980160",
               "00186198904","00093681673", "00591376830","00781751687",
               "00781751787","00591376730","00093681573", "00186198804",
               "00093681773","00173071920","00173071820","00173072020",
               "00085433401","00085433301","00173060202","00173060102",
               "00173087410","00085134102","00085134101","00085134107",
               "00186091706","00186091612","00173071920","35356016608",
               "68258303701","00186037028","59310057922","00173068220",
               "00085113201","00173068221","00597010061","00597008717",
               "54868564600","00186037028","54569646600","00085113201",
               "54868605100","35356016608") ~ "INHALATION",
    NDC %in% c("00641602325") ~ "INTRAVENOUS",
    NDC %in% c("59762453701", "59762453702","59148001971","00641604325",
               "59762453802","00009737611") ~ "INTRAMUSCULAR",
    NDC %in% c("00781652486", "00781652386", "00037024330","51525029403",
               "47335077991", "60505083305","51991081403", "60505084805",
               "51525023403") ~ "NASAL",
    NDC %in% c("57664014634", "00641149535", "00009001103","00009001104",
               "00641092825","76045010610","00641037625","63323025803",
               "00409379501","63323016530") ~ "INTRAVENOUS/INTRAMUSCULAR",
    NDC %in% c("42037010478") ~ "OTIC",
    NDC %in% c("24338030065") | Form=="SUBL" ~ "SUBLINGUAL",
    NDC %in% c("25010081756","24208000403","17478028811","61314022705","64980051405",
               "61314022505","64980051401","64980051415","76478000215","60758080105",
               "24208050307", "61314022710","00187149605","25010081566","17478028810",
               "61314022615","61314022715","17478028812","24208060203","17478062512",
               "49348014929","00378964532","41616022090","17478020910","17478020919",
               "47781013736","17478029010","00065026005","00065026025","49884004448",
               "00065027105","50383023210","00093761843","24208048510","00591248179",
               "00065027225","61314001910","60429011410","17478071310","24208043405",
               "64980051505","17478071311","64980051501","68180043501","50383018902",
               "61314027105", "60758061525","00023361525","24208058060","61314063305",
               "17478028310","64980051705","60505057501","00065427325","65862075705",
               "17478071410","61314065605","16571012025", "16571012050","61314065610",
               "61314065625","58118996805","24208041105","00023932115","00023932105",
               "00023932110","52959026505","61314014305","24208041110","61314014405",
               "61314014310","17478071510","61314014410","00023921105","00023921110",
               "17478071511","00172499716","61314014415","24208071510","00065401303",
               "16590035503","00065000603","00023320503","00023320505","00023320508",
               "00023916305","00065065435","00023916330","00023916360","00591248279",
               "76478000210","60505055104","24208048610","54569625100",
               "55045216805","21695098505","00574402435") ~ "OPHTHALMIC",
    NDC %in% c("40076031812","00093613932","00093613832","00713052612",
               "00591299239", "00591298539", "00713011812","00713013512",
               "00713010912","00574202007") ~ "RECTAL",
    TRUE ~ Form
  ))

# remove NDC that has mismatched drug record
meps_clean = meps_clean[meps_clean$NDC != "21695058960", ]
meps_clean = meps_clean[meps_clean$NDC != "54569543101", ]

# calculate daily frequency for each record
meps_clean = meps_clean %>%
  mutate(Daily_Frequency = Quantity/Day_Supply)

# calculate daily dosage for each record (Strength is mass of active ingredient)
meps_clean = meps_clean %>%
  mutate(Daily_Dosage = Strength*Daily_Frequency)

# decode therapeutic classes
meps_clean = meps_clean %>%
  mutate(Therapeutic_class_1 = case_when(Therapeutic_class_1=="1" ~ "Anti-infectives",
                                         Therapeutic_class_1=="20" ~ "Antineoplastics",
                                         Therapeutic_class_1=="28" ~ "Biologicals",
                                         Therapeutic_class_1=="40" ~ "Cardiovascular agents",
                                         Therapeutic_class_1=="57" ~ "Central nervous system agents",
                                         Therapeutic_class_1=="81" ~ "Coagulation modifiers",
                                         Therapeutic_class_1=="87" ~ "Gastrointestinal agents",
                                         Therapeutic_class_1=="97" ~ "Hormones/hormone modifiers",
                                         Therapeutic_class_1=="105" ~ "Miscellaneous agents",
                                         Therapeutic_class_1=="113" ~ "Genitourinary tract agents",
                                         Therapeutic_class_1=="115" ~ "Nutritional products",
                                         Therapeutic_class_1=="122" ~ "Respiratory agents",
                                         Therapeutic_class_1=="133" ~ "Topical agents",
                                         Therapeutic_class_1=="218" ~ "Alternative medicines",
                                         Therapeutic_class_1=="242" ~ "Psychotherapeutic agents",
                                         Therapeutic_class_1=="254" ~ "Immunologic agents",
                                         Therapeutic_class_1=="358" ~ "Metabolic agents",
                                         TRUE ~ as.character(Therapeutic_class_1)))

# define "Multiple" label if drug has >1 class (Indicated by Therapeutic_class_2
# and Therapeutic_class_3)
meps_clean <- meps_clean %>%
  mutate(Therapeutic_class_1 = case_when(
    !is.na(Therapeutic_class_1) & (!is.na(Therapeutic_class_2) | !is.na(Therapeutic_class_3)) ~ "Multiple",
    TRUE ~ as.character(Therapeutic_class_1))) %>%
  select(-c(Therapeutic_class_2, Therapeutic_class_3)) %>%
  rename(Therapeutic_class = Therapeutic_class_1)

################################################################################
# Load and integrate excretion data.
################################################################################

# load machine readable section of spreadsheet
excretion_data = read_excel("data/excretion_data.xlsx", sheet="Machine readable data")

# make administration column same format as form column in MEPS data
excretion_data <- excretion_data %>%
  mutate(Form = toupper(Form)) %>%
  rename(Drug = Pharmaceutical) %>% 
  rename(Excretion_percentage = Excretion_fraction)

excretion_data$Drug = trimws(toupper(excretion_data$Drug))

meps_clean = meps_clean %>%
  left_join(excretion_data %>% select(Drug, Form, Excretion_percentage),
            by = c("Drug", "Form"))

# fixing drugs that have forms in MEPS that do not match forms in excretion data
# by either 1) adding available excretion fraction if combo form (e.g., intravenous
# intramuscular), or 2) adding highest available excretion fraction of all
# available excretion forms in excretion data if form not available in excretion
# data (notably done only for common drugs between all 6 years) - only exception
# to the prioritization is if the drug seems miscoded (e.g., as oral mucosa instead
# of oral)
meps_clean = meps_clean %>%
  mutate(Excretion_percentage = case_when(NDC %in% c("00641092825","00641149535") ~ 30,
                                          NDC %in% c("49348014929") ~ 6,
                                          NDC  %in% c("00713011812") ~ 9,
                                          is.na(Excretion_percentage)==TRUE & Drug=="OLOPATADINE" ~ 77.2,
                                          is.na(Excretion_percentage)==TRUE & Drug=="METHYLPREDNISOLONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="HYDROCORTISONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="ASPIRIN" ~ 0,
                                          is.na(Excretion_percentage)==TRUE & Drug=="TIOTROPIUM" ~ 7,
                                          is.na(Excretion_percentage)==TRUE & Drug=="NITROGLYCERIN" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="DEXAMETHASONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="FLUCONAZOLE" ~ 80,
                                          is.na(Excretion_percentage)==TRUE & Drug=="KETOROLAC" ~ 66,
                                          is.na(Excretion_percentage)==TRUE & Drug=="DIPHENHYDRAMINE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="METHOTREXATE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="BISACODYL" ~ 100,
                                          .default=Excretion_percentage))

################################################################################
# Add population data
################################################################################

# Load 2016 population data
meps_pop = read.delim("data/MEPS_pop_data_2016.txt", header=TRUE, sep = "\t",
                      colClasses = "character", dec = ".")

# create unique Person ID 
meps_pop = meps_pop %>%
  mutate(Person_ID = paste0(DUPERSID, PANEL)) %>%
  select(-c(DUPERSID)) 

# create unique Household ID
meps_pop = meps_pop %>%
  mutate(Household_ID = paste0(DUID, PANEL)) %>%
  select(-c(DUID, PANEL)) 

# prep meps_pop to link to meps_clean
meps_pop = meps_pop %>%
  select(c("Person_ID","Household_ID", "AGE16X", "SEX", "FAMINC16",
           "INSCOV16","RACETHX")) # these include age, sex, family income,
                                  # insurance coverage, race/ethnicity in that
                                  # order

# handle missing values (all are negative valued reserved codes; -2, -10, -13
# not present in selected data; NA only for Age; handled in later scripts;
# all other variables had complete data after this step)
meps_pop <- meps_pop %>%
  mutate(across(colnames(meps_pop), ~ case_when(
    as.character(.) %in% c("-15", "-9", "-8", "-7", "-1") ~ NA,  
    suppressWarnings(as.numeric(.)) %in% c(-15, -9, -8, -7, -1) ~ NA,  
    TRUE ~ .  
  )))

# fix order and colnames
colnames(meps_pop) = c("Person_ID", "Household_ID", "Age", "Sex",
                       "Family_income", "Insurance_coverage",
                       "Race_ethnicity")

# remove negative values from income columns (zeros allowed)
meps_pop = meps_pop %>%
  filter(Family_income >= 0)

# Link by Person ID (retaining people in "No prescriptions" category too)
meps_clean = full_join(meps_clean,meps_pop,by="Person_ID")
meps_clean = meps_clean %>%
  mutate(Drug = ifelse(is.na(Drug), "No prescriptions", Drug)) %>%
  filter(!(Person_ID %in% c("1764710121","1492210121","1764710321",
                            "1764710421")))

# Add year for integration
meps_clean = meps_clean %>%
  mutate(Year="2016")

# export data as csv
write.csv(meps_clean,"data/meps_clean_2016.csv")