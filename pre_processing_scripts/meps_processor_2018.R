################################################################################
# Title: meps_processor_2018.R

# Description:
# Loads, cleans and exports data from the Medical Expenditure Panel Survey
# 2018 Prescribed Medicines File and 2018 Full Year Population Characteristics
# File

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
library("data.table")

# load functions library
source("meps_processor_functions.R")

################################################################################
# Load and clean Medical Expenditure Panel Survey (MEPS) 2018 prescription data.
################################################################################

# load MEPS dataset (all columns read in as characters)
meps = read.delim("data/MEPS_data_2018.txt", header=TRUE, sep = "\t",
                  colClasses = "character", dec = ".")

# extract desired data columns
meps = meps %>%
  select(c(DUPERSID, RXDRGNAM:RXDAYSUP, TC1, TC2, TC3))

# update to clearer column names
colnames(meps)[1:12] = c("Person_ID", "Drug", "NDC", "Quantity", "Form",
                        "Form Units", "Strength", "Units", "Day_Supply",
                        "Therapeutic_class_1", "Therapeutic_class_2",
                        "Therapeutic_class_3")

# replace missing values with NA
meps <- meps %>%
  mutate(across(colnames(meps), ~ case_when(
    as.character(.) %in% c("-15", "-8", "-7", "-14", "-1") ~ NA,  
    suppressWarnings(as.numeric(.)) %in% c(-15, -8, -7, -14, -1) ~ NA,  
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
                   "OTHER/OTHER","MG/OTHER","CC/OTHER")
for (unit in miscoded_units){
  meps = remove_entries(meps, meps$Units, unit)
}

# remove unnecessary info from units and/or fix spelling/punctuation to be
# standard across all entries
fix_units_df = data.frame(miscoded_units = c("MCG/INH", "MG/ACT", "MCG/mg/Act",
                                             "MG/GM", "mg/Act", "GM/SCOOP",
                                             "MG/mg/Act", "MCG/BLIST","MCG/SPRAY",
                                             "MG/SPRAY", "MCG/ACT", "GM", "GM/ML",
                                             "MCG/MCG/ACT","MCG/DOSE","MCG/MCG/DOSE",
                                             "MG/MG/ML","MG/ML/ML","MG/DROP","MG/GR"),
                          correct_units = c("MCG", "MG", "MCG/MG", "MG", "MG",
                                            "G","MG/MG", "MCG", "MCG", "MG", "MCG",
                                            "G", "G/ML", "MCG/MCG","MCG","MCG/MCG",
                                            "MG/MG","MG","MG","MG/G"))
for (i in seq_len(nrow(fix_units_df))){
  meps = fix_drug_coding(meps, meps$Units, "Units", fix_units_df[i,1],
                         fix_units_df[i,2])
}

# fix dataset specific miscoding
meps = meps %>%
  filter(Units != "MCG/%") %>%
  mutate(Units = case_when(Units=="MG/HR/MG/HR" ~ "MG/HR",
                           .default=Units))

# remove miscoded drugs or drugs with vague/missing info
meps = meps[is.na(meps$Units)==FALSE,]
meps = meps[is.na(meps$Drug)==FALSE,]
meps = remove_entries(meps, meps$Form, "OTHER")
meps = remove_entries(meps, meps$Units, "ML")

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
                           (str_count(Drug, "/") == 1 | str_count(Drug, "-") == 1) & Units=="ML" ~ "ML/ML",
                           (str_count(Drug, "/") == 1 | str_count(Drug, "-") == 1) & Units=="G" ~ "G/G",
                           (str_count(Drug, "/") == 2 | str_count(Drug, "-") == 2) & Units=="G" ~ "G/G/G",
                           .default = Units))

# remove rows where strength has no slash/dash and drug does
meps = meps %>%
  filter(!((grepl("/",Drug)==TRUE | grepl("-",Drug)==TRUE) & (grepl("/",Strength)==FALSE & grepl("-",Strength)==FALSE)))

# fix dataset specific miscoding
meps = meps %>%
  filter(Drug != "ACETAMINOPHEN/DEXTROMETHORPHAN/PE") %>%
  mutate(Strength = case_when(Drug=="ACETAMINOPHEN/BUTALBITAL/CAFFEINE" ~ "325/50/40",
                              .default=Strength)) %>%
  mutate(Units = case_when(Drug=="ACETAMINOPHEN/BUTALBITAL/CAFFEINE" ~ "MG/MG/MG",
                           .default=Units))
meps = meps %>%
  mutate(Drug = case_when(Drug=="BROMPHENIRAMINE/DEXTROMETHORPHAN/PSE" ~ "BROMPHENIRAMINE/DEXTROMETHORPHAN/PSEUDOEPHEDRINE",
                          .default=Drug)) %>%
  mutate(Strength = case_when(Strength=="30/2/10/5" ~ "2/10/30",
                              .default=Strength)) %>%
  mutate(Strength = case_when(NDC=="50383080316" ~ "6.25-15",
                              .default=Strength)) %>%
  mutate(Units = case_when(NDC=="68788645301" ~ "MG/MG",
                           NDC=="50383080316" ~ "MG/MG",
                           NDC=="24208048610" ~ "MG/MG",
                           .default=Units))

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
                          "ALPHA-LIPOIC ACID")
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
  mutate(Units=case_when(Units=="%/%" ~ "%",
                         Units=="MG/MG" ~ "MG",
                         Units=="MG/G" ~ "MG",
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

# ensure final drug list
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
  
  # Convert MG/9HR to MCG
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="MG/9HR" ~ Strength*(24/9)*1000,
                                .default=Strength))
  
  # Convert MG/HR to MCG
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="MG/HR" ~ Strength*24*1000,
                                .default=Strength))
  
  # Convert IU to MCG (100 IU insulin glargine is 3.64 mcg)
  drug_data = drug_data %>%
    mutate(Strength = case_when((Units=="IU" | Units=="IU/ML") & Drug=="INSULIN GLARGINE" ~ Strength*3.64/100,
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
api_results = read.csv("meps_2018_from_api.csv")
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
    Form %in% c("TABS", "CAP-Capsule", "CAPS", "TAB","TB24","TB12",
                "CHEW","CAP-Caplets","SYRP","TBEC","CPDR","TBDP",
                "CP24","TBCR","CPCR","ELIX","CPEP","CTB","ECC","T12A",
                "SYR","GELC","ELI","CP12","CPSP","T24A") ~ "ORAL",
    Form %in% c("NEBU","INH-Inhalant","INH-Inhaler") ~ "INHALATION",
    Form %in% c("Eye drops") ~ "OPHTHALMIC",
    Form %in% c("ENEM") ~ "RECTAL",
    Administration_route == "TOPICAL" ~ "TOPICAL",
    Administration_route == "NASAL" ~ "NASAL",
    Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Administration_route == "OTIC" ~ "OTIC",
    Form == "INTRAMUSCULAR, INTRAVENOUS" ~ "INTRAVENOUS/INTRAMUSCULAR",
    Form =="INTRAMUSCULAR, INTRAVENOUS, SUBCUTANEOUS" ~ "INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS",
    TRUE ~ Form   
  ))

# change Form variable to reflect full form name for merging with excretion data
# (those that are in the MEPS codebook but have been removed by steps above are not
#accounted for below); note that 8 and 9 digit NDCs are miscoded and route is
# curated to fix this
meps_clean <- meps_clean %>%
  mutate(Form = case_when(NDC %in% c("121072104","54838055550","51672409103",
                                     "65162069179","16714067102","54372250",
                                     "54372263","50383028616","54318863",
                                     "52652400101","60432070605","54039063",
                                     "536012297","536012285","54329446",
                                     "60432061360","54040444","54051750",
                                     "406800330","54051744","574015330",
                                     "51991083704","54838057280","23155029251",
                                     "54838055240","51991083716","13668002907",
                                     "13668059607","54317757","54317763",
                                     "50383004248","121075908","50383004224",
                                     "60432021208","50383004004","13925016604",
                                     "71201223","603116158","45802068028",
                                     "116200116","51991065116","50383024116",
                                     "121079916","50383077932","60432003816",
                                     "50383077916","603137858","121057716",
                                     "50383079516","93417573","93417774",
                                     "93417773","42043014338","68180012402",
                                     "93416178","65862007101","143988701",
                                     "93416173","143988775","781615757",
                                     "65862007175","93415573","143988675",
                                     "781615746","143988915","781604155",
                                     "93416176","93415580","143988901",
                                     "143988750","65862070701","93415579",
                                     "93415080","57237003301","143988980",
                                     "93416078","781615657","65862070755",
                                     "93202623","59762313001","93202631",
                                     "59762312001","93202723","59762314001",
                                     "59762311001","93202694","24478021030",
                                     "16714039301","68180072320","68180072310",
                                     "781607861","65862021860","68180072220",
                                     "42043025267","16714039201","93413764",
                                     "16714039202","54000285","62559044001",
                                     "50090022000","68788994801","67253014940",
                                     "68788684001","63629344902","187073030",
                                     "187073090","62175061746","8060701",
                                     "62175031037","406012701","62175031137",
                                     "54868521702","50268053115","87606413",
                                     "591084510","64679073409","62037083201",
                                     "54868572900","378459510","49884082501",
                                     "54868244001","536100410","10135017362",
                                     "63824004124","143998250","54569569000",
                                     "53217028601","54569281400","603083994",
                                     "121148800","121148810","54838013940",
                                     "57237003375","68134020116","65628007010",
                                     "65628007003","45802095243","50383058416",
                                     "49348050034","472125594","472127016",
                                     "472176094","45802020326","472008216",
                                     "78035752","54019959","93202631","603906354",
                                     "60432021208","60687024940","59762305102",
                                     "66993041630","186404001","50383070530",
                                     "527176836","54317644","59746000103",
                                     "68382091634","781502207","603459315",
                                     "68001000501","591079021","591505221",
                                     "603533715","591544221","603533815",
                                     "603533731","591544243","62175018146",
                                     "8060701","62175061746","62175018143",
                                     "8060704","68084084825","24478013001",
                                     "70165020030","536077097","536077085",
                                     "58657052016","603083994","54838014470",
                                     "54838014440","50383061804","51079045220",
                                     "24510011010","93412774","93412773",
                                     "93412573","65862059601","59762001601",
                                     "47335067513","68382059505","832107130",
                                     "832107430","68382010601") ~ "ORAL",
                          NDC %in% c("52565001859","168020160","472098792",
                                     "713070153","43199003160","63646050050",
                                     "63646050025","60432013350","50383026650",
                                     "51672129302","68462053253","51672406301",
                                     "50383041906","51672530200","713031788",
                                     "50383026650","65162083366","63481068447",
                                     "69097052444","42291025611","54868618100",
                                     "781708035","66993096245","713063737",
                                     "299382060","17478071110","17478071130",
                                     "25021067376","168020260","168020230",
                                     "59762374301","59762374302","49884093547",
                                     "51672129401","68462040355","299590660",
                                     "51672137706","54868599100","93630195",
                                     "472012645","66993088445","574206130",
                                     "60793041130","591352530","603188016",
                                     "574206302","234057504","234057508",
                                     "713063337","168032346","69315030230") ~ "TOPICAL",
                          NDC %in% c("68968555403","68968555503","378087399",
                                     "591350804","51862045504","378087299",
                                     "378910493","47781029803","47781029603",
                                     "378910293") ~ "TRANSDERMAL",
                          NDC %in% c("50383077504","54350049") ~ "MUCOSA",
                          NDC %in% c("703367103") ~ "INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTERIAL/INTRATHECAL",
                          NDC %in% c("781317414","169406013","169406012",
                                     "169280015","169406013","54569650700",
                                     "60429099505") ~ "SUBCUTANEOUS",
                          NDC %in% c("76204010025","487980125","378797052",
                                     "591379860","378797093","487980130",
                                     "173068220","59310057922","85113204",
                                     "85113201","597010061","597016061",
                                     "597008717","59310058020","173087610",
                                     "173087410","173060202","173060002",
                                     "173060102","85134102","85134101",
                                     "186091612","186091706","33261089301",
                                     "173068254","173069704","186037228",
                                     "186037220","781751687","93681673",
                                     "69097032187","487970130","69097031987",
                                     "186198804","115168974","115168774",
                                     "781751587","173071920","173071820",
                                     "173072020","85433301","85433401") ~ "INHALATION",
                          NDC %in% c("59762453701","703680101","548540000",
                                     "59762453802","59148007280") ~ "INTRAMUSCULAR",
                          NDC %in% c("69543038625","9001104","9001103") ~ "INTRAVENOUS/INTRAMUSCULAR",
                          NDC %in% c("24208039830","54004641","54004544","24208039915",
                                     "527185943","781652386","527181843",
                                     "51525029403","51991081403",
                                     "60505084805","85075605","60505082901",
                                     "54327099","60505082901","50383070016",
                                     "60432026415","00135057603","00536109170",
                                     "781635587","60505083001","85128801",
                                     "60505616701","54327099","536109170",
                                     "536109194","50383070016","52959095616",
                                     "49999098216","68788967801","55700068016",
                                     "54868554500","54569578000","60505082901",
                                     "62451008130","54868504900") ~ "NASAL",
                          NDC %in% c("42037010478","42037010479","60505036301",
                                     "24208041005","61314001510","50383002505",
                                     "60505036302") ~ "OTIC",
                         Form %in% c("SUBL","Sublingual") ~ "SUBLINGUAL",
                          NDC %in% c("61314014305","17478071510","24208041110",
                                     "17478071511","61314014310","23932105",
                                     "23932115","23917705","23932110",
                                     "61314014405","17478071512","61314014410",
                                     "23917710","61314014415","59762033302",
                                     "61314054701","64980051625","13830304",
                                     "24208046325","61314054703","517083001",
                                     "781713593","65401303","65427325",
                                     "61314027105","61314027225","65027225",
                                     "64980051705","93768432","60505057501",
                                     "70069000701","60429095705","17478010505",
                                     "17478071410","61314065605","61314065625",
                                     "16571012025","23320503","23320508",
                                     "23320505","68180042901","60758018805",
                                     "17478028310","64980051505","24208043405",
                                     "17478071310","24208043410","17478071311",
                                     "61314064305","17478029010","187149605",
                                     "17478028810","17478028812","17478028811",
                                     "60758077305","17478020911","60505100301",
                                     "17478020910","47335022090","60505100302",
                                     "50383023210","61314054701","61314054703",
                                     "51552143903","54569613500","61314063705",
                                     "71384050105","61314063710","61314063705",
                                     "60758011905","61314063715","61314063710",
                                     "17478071510","17478071511","61314014315",
                                     "54569633400","61314014310","24208046325",
                                     "61314054701","54569625100","59762033302",
                                     "61314054703","50090124000","17478051919",
                                     "65027105","24208044425","23320505",
                                     "50436675201","55700003805","17478029010",
                                     "60505059801","50090185200","60505059802",
                                     "50383002105","17478020910","17478020919",
                                     "68788738405","60429011410") ~ "OPHTHALMIC",
                          NDC %in% c("93613832","68682065220","187065820",
                                     "93613732","713010908","713010912",
                                     "51672529701","591299239","45802075930",
                                     "713016550","713050312","713050324",
                                     "574709012","57470931") ~ "RECTAL",
                          NDC %in% c("781707787") ~ "VAGINAL",
                          TRUE ~ Form
  ))

# drop records without NDC or admin route
meps_clean <- meps_clean %>%
  filter(!(is.na(NDC) & is.na(Administration_route)))
meps_clean = meps_clean[meps_clean$NDC != "43353056960", ]
meps_clean <- meps_clean %>%
  filter(!if_all(everything(), is.na))

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
  mutate(Excretion_percentage = case_when(is.na(Excretion_percentage)==TRUE & Drug=="METHYLPREDNISOLONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="TIOTROPIUM" ~ 7,
                                          is.na(Excretion_percentage)==TRUE & Drug=="AMLODIPINE" ~ 10,
                                          is.na(Excretion_percentage)==TRUE & Drug=="ASPIRIN" ~ 0,
                                          is.na(Excretion_percentage)==TRUE & Drug=="BISACODYL" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="OLOPATADINE" ~ 77.2,
                                          is.na(Excretion_percentage)==TRUE & Drug=="ACETAMINOPHEN" ~ 9,
                                          is.na(Excretion_percentage)==TRUE & Drug=="HYDROCORTISONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="DEXAMETHASONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="LIDOCAINE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="NITROGLYCERIN" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="FLUCONAZOLE" ~ 80,
                                          is.na(Excretion_percentage)==TRUE & Drug=="KETOROLAC" ~ 66,
                                          is.na(Excretion_percentage)==TRUE & Drug=="METHOTREXATE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="DIPHENHYDRAMINE" ~ 100,
                                          .default=Excretion_percentage))

################################################################################
# add population data
################################################################################

# Load 2018 population data
meps_pop = read.delim("data/MEPS_pop_data_2018.txt", header=TRUE, sep = "\t",
                      colClasses = "character", dec = ".")

# prep meps_pop to link to meps_clean
meps_pop = meps_pop %>%
  rename(`Person_ID` = DUPERSID) %>%
  rename(`Household_ID` = DUID) %>%
  select(c("Person_ID","Household_ID", "AGE18X", "SEX", "FAMINC18",
           "INSCOV18","RACETHX")) # these include age, sex, family income,
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
  filter(!(Person_ID %in% c("2294109101","2293024102","2296468102",
                            "2298954101","2293024101","2298119101")))

# Add year for integration
meps_clean = meps_clean %>%
  mutate(Year="2018")

# export data as csv
fwrite(meps_clean, "data/meps_clean_2018.csv")