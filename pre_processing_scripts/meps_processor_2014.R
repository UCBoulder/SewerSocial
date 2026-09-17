################################################################################
# Title: meps_processor_2014.R

# Description:
# Loads, cleans and exports data from the Medical Expenditure Panel Survey
# 2014 Prescribed Medicines File and 2014 Full Year Population Characteristics
# File. Integrates manually curated data from drug labels available from
# Drugs@FDA database.

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
# Load and clean Medical Expenditure Panel Survey (MEPS) 2014 prescription data.
################################################################################

# load MEPS dataset (all columns read in as characters)
meps = read.delim("data/MEPS_data_2014.txt", header=TRUE, sep = "\t",
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
    as.character(.) %in% c("-15","-9", "-8", "-7", "-14", "-1") ~ NA,  
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

# NOTE: all the corrections to miscoded info below are curated against known
# dosages/units

# remove ambiguous drug units
miscoded_units = c("OTHER", "U/ML", "U/ML/U/ML", "UNIT", "UNIT/ML", "UNIT/GM",
                   "U/GM", "UT/ML", "MCG/OTHER", "%/OTHER","DOSE","","UNIT/ML",
                   "B CELL", "MG/OTHER","OTHER/OTHER", "U/GM/MG","UNT/ML",
                   "U/ML/ML","U")
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
                                             "MG/MG/ML","MG/ML/ML","MG/DROP",
                                             "%/G","GM/DOSE","GR","%/INH"),
                          correct_units = c("MCG", "MG", "MCG/MG", "MG", "MG",
                                            "G","MG/MG", "MCG", "MCG", "MG", "MCG",
                                            "G", "G/ML", "MCG/MCG","MCG","MCG/MCG",
                                            "MG/MG","MG","MG", "%", "G","G","%"))
for (i in seq_len(nrow(fix_units_df))){
  meps = fix_drug_coding(meps, meps$Units, "Units", fix_units_df[i,1],
                         fix_units_df[i,2])
}

# adjust specific cases to this dataset for MG/ML/MG/ML units
meps = meps %>%
  mutate(Units = case_when(Units=="MG/ML/MG/ML" & Drug=="ACETAMINOPHEN" ~ "MG",
                           Units=="MG/ML/MG/ML" & Drug!= "ACETAMINOPHEN" ~ "MG/MG",
                           .default=Units))

# remove miscoded drugs or drugs with vague/missing info
meps = meps[is.na(meps$Units)==FALSE,]
meps = meps[is.na(meps$Drug)==FALSE,]
meps = remove_entries(meps, meps$Form, "OTHER")
meps = remove_entries(meps, meps$Units, "ML")
meps = remove_entries(meps, meps$Units, "ML/ML")
meps = remove_entries(meps, meps$NDC, "24208063562") #inaccurate/incomplete
#for this dataset
meps = remove_entries(meps, meps$NDC, "54868593600") #same reason
meps = remove_entries(meps, meps$NDC, "45802070111") # mismatch between reported drug in MEPS and NDC
meps = remove_entries(meps, meps$NDC, "51021077014") # non-existent NDC

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

# remove rows where strength has no slash/dash and drug does
meps = meps %>%
  filter(!((grepl("/",Drug)==TRUE | grepl("-",Drug)==TRUE) & (grepl("/",Strength)==FALSE & grepl("-",Strength)==FALSE)))

# drop combo drugs with only 1 strength listed for more than 1 drug (not clear 
# which drug the strength is for), fix miscoded drugs
meps = meps %>%
  filter(Drug != "ACETAMINOPHEN/BUTALBITAL/CAFFEINE") %>%
  filter(Drug != "ACETAMINOPHEN/CPM/DEXTROMETHORPHAN/PE") %>%
  mutate(Drug = case_when(Drug=="INSULIN ISOPHANE (NPH)" ~ "INSULIN ISOPHANE",
                          .default=Drug))

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

# fix units for non-combination drugs that did not split correctly
meps = meps %>%
  mutate(Units = case_when(Units=="MG/MG" ~ "MG",
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

# ensure all units NA are removed
meps = meps[is.na(meps$Units)==FALSE,]

# remove any duplicates introduced
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
  # example, 1% solution is 1 g drug per 100 mL solvent, assume 100mL each)
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
  # 1 IU insulin isophane is 0.035 mg; 1 IU nystatin is 0.000333mg; 762 IU
  # of neomycin per mg)
  drug_data = drug_data %>%
    mutate(Strength = case_when(Units=="IU" & Drug=="INSULIN GLARGINE" ~ Strength*3.64/100,
                                Units=="IU" & Drug=="NYSTATIN" ~ Strength*0.000333*1000,
                                Units=="IU" & Drug=="INSULIN ISOPHANE" ~ Strength*0.035*1000,
                                Units=="IU" & Drug=="NEOMYCIN" ~ Strength*1000/762,
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

# change Form variable to reflect full form name for merging with excretion data
# (those that are in the MEPS codebook but have been removed by steps above are not
#accounted for below)
meps_clean <- meps_clean %>%
  mutate(Form = case_when(
    Form %in% c("TABS", "TB24", "TAB", "CAPS", "CAP", "CAP-Capsule", "TBEC", "CPDR", 
                "TBDP", "T12A", "SYRP", "CP24", "TBCR", "CPCR", "TB12", "CHEW", 
                "CPEP", "ECT", "CPSP", "CTB", "SYR", "CP12", "PACK", "TBED", "CHER",
                "SOCT", "TBPK", "TBDD", "CSDR", "C", "CA", "PAK", "ECC", "TAM",
                "CAP-Caplets", "ELI", "SYP", "CER", "GELC", "BLO", "ELIX") ~ "ORAL",
    Form %in% c("AEPB", "INH-Inhaler", "INH-Inhalant", "NEBU", "NEB", "AERB", "ACC", "ACT",
                "Diskus") ~ "INHALATION",
    Form %in% c("AERO", "AER", "AERS", "AE") & Administration_route == "NASAL" ~ "NASAL",
    Form %in% c("AERO", "AER", "AERS", "AE") & Drug %in% c("FLUTICASONE", "ALBUTEROL", "TIOTROPIUM",
                                                           "MOMETASONE", "LEVALBUTEROL", "CICLESONIDE",
                                                           "SALMETEROL", "BECLOMETHASONE",
                                                           "IPRATROPIUM") ~ "INHALATION",
    Form %in% c("SUSP", "SUS", "SUSR") & Administration_route == "NASAL" ~ "NASAL",
    Form %in% c("SUSP", "SUS", "SUSR") & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Form %in% c("SUSP", "SUS", "SUSR") & Administration_route == "OTIC" ~ "OTIC",
    Form %in% c("SUSP", "SUS", "SUSR") & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form %in% c("SUSP", "SUS", "SUSR") & Drug %in% c("IBUPROFEN", "ACETAMINOPHEN", "CIPROFLOXACIN",
                                                     "FAMOTIDINE", "CEFDINIR", "NAPROXEN", "AMOXICILLIN",
                                                     "AZITHROMYCIN", "NITROFURANTOIN", "OXCARBAZEPINE",
                                                     "OSELTAMIVIR", "PHENYTOIN", "PAROXETINE", "LANSOPRAZOLE",
                                                     "FEXOFENADINE", "TACROLIMUS", "FLUCONAZOLE","ERYTHROMYCIN",
                                                     "METHYLPHENIDATE", "CLARITHROMYCIN","CEFUROXIME",
                                                     "CARBAMAZEPINE", "CEPHALEXIN", "ACYCLOVIR") ~ "ORAL",
    Form %in% c("SUSP", "SUS", "SUSR") & Drug == "BUDESONIDE" ~ "INHALATION",
    Form %in% c("SUSP", "SUS", "SUSR") & Drug == "METHYLPREDNISOLONE" ~ "INTRAMUSCULAR/INTRA-ARTICULAR/SOFT TISSUE/INTRALESIONAL",
    Form %in% c("SUSP", "SUS", "SUSR", "SUSY") & Drug %in% c("MEDROXYPROGESTERONE","RISPERIDONE") ~ "INTRAMUSCULAR",
    Form == "SUSR" & Drug == "EXENATIDE" ~ "SUBCUTANEOUS",
    Form %in% c("SOLN", "SOL", "SOLR") & Administration_route == "NASAL" ~ "NASAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Form %in% c("SOLN", "SOL", "SOLR") & Administration_route == "OTIC" ~ "OTIC",
    Form %in% c("SOLN", "SOL", "SOLR") & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug %in% c("METFORMIN", "OXYCODONE", "FUROSEMIDE", "GABAPENTIN",
                                                     "PREDNISONE", "ASPIRIN", "PROMETHAZINE",
                                                     "ONDANSETRON", "ARIPIPRAZOLE", "LEVETIRACETAM",
                                                     "CETIRIZINE", "ACETAMINOPHEN","DICYCLOMINE",
                                                     "FLUOXETINE","LEVOCETIRIZINE","OMEPRAZOLE",
                                                     "LEVOFLOXACIN","HYDROXYZINE", "DIAZEPAM", "METOCLOPRAMIDE",
                                                     "PENICILLIN V POTASSIUM", "LORATADINE", "ALENDRONATE",
                                                     "LANSOPRAZOLE",
                                                     "MORPHINE", "LACTULOSE", "LEVOCARNITINE") ~ "ORAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug=="CLINDAMYCIN" & Administration_route != "TOPICAL" ~ "ORAL",
    Form %in% c("SOLN","SOL") & NDC %in% c("00093611816","00093611887", "00121075908","50383004248",
                              "60432021208","00093611887") ~ "ORAL",
    Form=="SOLN" & NDC %in% c("24208071510") ~ "OPHTHALMIC",
    Form == "SOL" & NDC == "00517090225" ~ "INTRAVENOUS/INTRAMUSCULAR",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "SUMATRIPTAN" ~ "NASAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug %in% c("BUDESONIDE", "ALBUTEROL") ~ "INHALATION",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "IPRATROPIUM" & Administration_route != "NASAL" ~ "INHALATION",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug %in% c("HALOPERIDOL") ~ "INTRAMUSCULAR",
    Form == "SOLN" & NDC %in% c("00409379601") ~ "INTRAMUSCULAR",
    Form == "SOLN" & NDC %in% c("17478020810", "17478020910", "17478020919",
                                  "60758077305","61314012605",
                                  "61314012610") ~ "OPHTHALMIC",
    Form == "SOLN" & NDC == "00054317763" ~ "ORAL",
    Form == "SOLN" & NDC == "63323016501" ~ "INTRAVENOUS/INTRAMUSCULAR",
    Form == "SOLR" & Drug=="HYDROCORTISONE" ~ "INTRAVENOUS/INTRAMUSCULAR",
    Form == "SOLN" & NDC =="00703367103" ~ "INTRA-ARTERIAL/INTRAMUSCULAR/INTRATHECAL/INTRAVENOUS",
    Form == "SOLN" & NDC == "63323012310" ~ "INTRAMUSCULAR/INTRAVENOUS/INTRA-ARTERIAL",
    Form %in% c("CREA", "SHA", "SHAM", "LOT", "LOTN", "PSTE", "OIL", "SWAB", "PADS") ~ "TOPICAL",
    Form == "CRE" & NDC != "00281032630" ~ "TOPICAL",
    Form == "CRE" & NDC == "00281032630" ~ "TRANSDERMAL",
    Form %in% c("OINT", "OIN") & Administration_route== "TOPICAL" ~  "TOPICAL",
    Form %in% c("OINT", "OIN") & Administration_route== "OPHTHALMIC" ~  "OPHTHALMIC",
    Form %in% c("OINT", "OIN") & Drug== "NITROGLYCERIN" ~  "TRANSDERMAL",
    Form %in% c("Eye drops", "EMUL") ~ "OPHTHALMIC",
    Form == "Drops" & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Form == "Drops" & Administration_route == "OTIC" ~ "OTIC",
    Form == "SPR" & Administration_route == "NASAL" ~ "NASAL",
    Form == "SPR" & Drug == "FLUTICASONE" & Administration_route != "NASAL" ~ "IHNALATION",
    Form == "SPR" & Drug == "NITROGLYCERIN" ~ "SUBLINGUAL",
    Form %in% c("SUBL", "Sublingual", "SUB") ~ "SUBLINGUAL",
    Form %in% c("PTWK", "PT24", "PTCH", "Patches") ~ "TRANSDERMAL",
    Form == "FILM" ~ "BUCCAL",
    Form == "Rinse" ~ "ORAL MUCOSA",
    Form == "CONC" ~ "ORAL",
    Form %in% c("TDM", "PAD") ~ "TOPICAL",
    Form == "POW" ~ "ORAL",
    Form == "POWD" & Drug == "MICONAZOLE" ~ "TOPICAL",
    Form == "POWD" & Drug == "CHOLESTYRAMINE" ~ "ORAL",
    Form == "INJ" & NDC == "54868402100" ~ "INTRAVENOUS/INTRAMUSCULAR",
    Form == "INJ" & NDC == "00641604201" ~ "INTRAVENOUS/INTRAMUSCULAR",
    Form == "INJ" & NDC %in% c("68258898501","54868142901") ~ "SUBCUTANEOUS",
    Form == "SOSY" ~ "SUBCUTANEOUS",
    Form %in% c("ENEM", "KIT") ~ "RECTAL",
    Form %in% c("SUP", "SUPP") & Drug == "HYDROCORTISONE" ~ "TOPICAL",
    Form == "SUPP" ~ "RECTAL",
    Form == "FOAM" ~ "TOPICAL",
    Form %in% c("LIQD", "LIQ") & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form %in% c("LIQD", "LIQ") & Drug %in% c("DIPHENHYDRAMINE","ACETAMINOPHEN",
                                             "GUAIFENESIN","LOPERAMIDE",
                                             "IBUPROFEN") ~ "ORAL",
    Form %in% c("Pen", "PEN", "SOAJ", "SOPN") ~ "SUBCUTANEOUS",
    Form ==  "GEL" & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form ==  "GEL" & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Form == 'GEL' & Drug == "DIAZEPAM" ~ 'RECTAL',
    Form == "SOLG" & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    NDC =="50419042301" ~ "INTRAUTERINE",
    Form == "SOLR" & Drug=="CLINDAMYCIN" ~ "ORAL",
    Form == "SPR" & NDC=="00173071920" ~ "INHALATION",
    Form %in% c("SOLN", "SOL") & Drug=="IPRATROPIUM" ~ "INHALATION",
    Form == "PT72" ~ "TRANSDERMAL",
    TRUE ~ Form   
  ))

# remove NDC that has mismatched drug record
meps_clean = meps_clean[meps_clean$NDC != "45802042578", ]

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
  mutate(Excretion_percentage = case_when(NDC=="54868402100" ~ 30,
                                          NDC %in% c("00065027225","00065027105",
                                                     "35356047725") ~ 77.2,
                                          NDC  %in% c("00603082358","00603082354",
                                                     "00603082381","00185064901",
                                                     "00185064801","00603082394",
                                                     "00603333932","00603334021",
                                                     "00603333921","00185064910",
                                                     "00603334032","00185064810",
                                                     "49348047104","00536359701",
                                                     "00247007310","00536077085") ~ 100,
                                          NDC %in% c("45802073032","00713016412",
                                                     "45802073233") ~ 9,
                                          NDC %in% c("23490600225","54868069101",
                                                     "00071041824","00071041813") ~ 100,
                                          NDC == "00009001103" ~ 100,
                                          NDC== "60793041130" ~ 100,
                                          NDC %in% c("00597007541","00597007547",
                                                     "00597007575") ~ 7,
                                          NDC == "00009307303" ~ 100,
                                          NDC %in% c("00641604201","00517090225") ~ 66,
                                          NDC == "00597002402" ~ 100,
                                          NDC == "63323016501" ~ 100,
                                          NDC %in% c("63323012310","00703367103") ~ 100,
                                          NDC %in% c("23490707101","55045263001") ~ 91,
                                          NDC %in% c("00713010912","45802057278") ~ 100,
                                          .default=Excretion_percentage))

################################################################################
# Add population data
################################################################################

# Load 2014 population data
meps_pop = read.delim("data/MEPS_pop_data_2014.txt", header=TRUE, sep = "\t",
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
  select(c("Person_ID","Household_ID", "AGE14X", "SEX", "FAMINC14",
           "INSCOV14","RACETHX")) # these include age, sex, family income,
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

meps_clean$Person_ID = trimws(meps_clean$Person_ID)
meps_pop$Person_ID = trimws(meps_pop$Person_ID)

# Link by Person ID (retaining people in "No prescriptions" category too)
meps_clean = full_join(meps_clean,meps_pop,by="Person_ID")
meps_clean = meps_clean %>%
  mutate(Drug = ifelse(is.na(Drug), "No prescriptions", Drug)) %>%
  filter(!(Person_ID %in% c("6505210119","6916310419","6916310119","4038110118",
                            "4038110218","6916310319","6916310219")))

# Add year for integration
meps_clean = meps_clean %>%
  mutate(Year="2014")
  
# export data as csv
write.csv(meps_clean,"data/meps_clean_2014.csv")