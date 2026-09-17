################################################################################
# Title: meps_processor_2020.R

# Description:
# Loads, cleans and exports data from the Medical Expenditure Panel Survey
# 2020 Prescribed Medicines File and 2020 Full Year Population Characteristics
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
# Load and clean Medical Expenditure Panel Survey (MEPS) 2020 prescription data.
################################################################################
# load MEPS dataset (all columns read in as characters)
meps = read.delim("data/MEPS_data_2020.txt", header=TRUE, sep = "\t",
                  colClasses = "character", dec = ".")

# extract desired data columns
meps = meps %>%
  select(c(DUPERSID, RXDRGNAM:RXDAYSUP, TC1, TC2, TC3))

# update to clearer column names
colnames(meps)[1:12] = c("Person_ID", "Drug", "NDC", "Quantity", "Form",
                        "Form Units", "Strength", "Units", "Day_Supply",
                        "Therapeutic_class_1", "Therapeutic_class_2",
                        "Therapeutic_class_3")

# replace missing values (-15, -8, and -7 in original data) with NA
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

# remove administration route labels from drug names (this prevents
# different forms of the same drug from being counted twice) and add to own
# column instead
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
                                          grepl("INHALED",Drug)==TRUE ~ "INHALATION",
                                          grepl("INHALATIONAL",Drug)==TRUE ~ "INHALATION",
                                          grepl("INHALANT",Drug)==TRUE ~ "INHALATION",
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
meps$Drug = gsub(paste0("\\b", "INHALATION", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "INHALATION", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "INHALATION", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "VAGINAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)
meps$Drug = gsub(paste0("\\b", "TRANSDERMAL", "\\b"), "", meps$Drug,
                 ignore.case = TRUE)

# remove ambiguous drug units
miscoded_units = c("OTHER", "U/ML", "U/ML/U/ML", "UNIT", "UNIT/ML", "UNIT/GM",
                   "U/GM", "UT/ML", "MCG/OTHER", "%/OTHER")
for (unit in miscoded_units){
  meps = remove_entries(meps, meps$Units, unit)
}

# remove unnecessary info from units and/or fix spelling/punctuation to be
# standard across all entries
fix_units_df = data.frame(miscoded_units = c("MCG/INH", "MG/ACT", "MCG/mg/Act",
                                             "MG/GM", "mg/Act", "GM/SCOOP",
                                             "MG/mg/Act", "MCG/BLIST",
                                             "MCG/SPRAY","MG/SPRAY", "MCG/ACT",
                                             "GM", "GM/ML","MG/ML/MG/ML"),
                          correct_units = c("MCG", "MG", "MCG/MG", "MG/G", "MG",
                                            "G","MG/MG", "MCG", "MCG", "MG",
                                            "MCG","G", "G/ML","MG/ML"))
for (i in seq_len(nrow(fix_units_df))){
  meps = fix_drug_coding(meps, meps$Units, "Units", fix_units_df[i,1],
                         fix_units_df[i,2])
}

# remove unnecessary commas
meps$Drug = gsub(",", "", meps$Drug)
meps$Strength = gsub(",", "", meps$Strength)

# NOTE: all the corrections to miscoded info below are curated against known
# dosages/units

# remove miscoded drugs or drugs with vague/missing info
meps = meps[is.na(meps$Units)==FALSE,]
meps = meps[is.na(meps$Drug)==FALSE,]
meps = remove_entries(meps, meps$Form, "OTHER")
meps = remove_entries(meps, meps$NDC, "53002090672")
meps = remove_entries(meps, meps$NDC, "70147031316")
meps = remove_entries(meps, meps$Units, "ML")

# fix cases with miscoded strengths
meps = meps %>%
  mutate(Strength = case_when(Strength=="160-4.5/1" ~ "160-4.5",
                              .default=Strength))
meps = meps %>%
  mutate(Strength = case_when(Strength=="1.25/1.25/1.25/1.25" ~ "1.25/1.25",
                              .default=Strength))

# fix miscoded albuterol (108 mcg albuterol sulfate = 90 mcg albuterol)
meps = meps %>%
  mutate(Strength = case_when(Drug=="ALBUTEROL" & Strength=="108/1" ~ "90",
                              Drug=="ALBUTEROL" & Strength=="108" ~ "90",
                              .default=Strength))

# fix albuterol units
meps = meps %>%
  mutate(Units = case_when(Drug == "ALBUTEROL" & Units=="MCG/MG" ~ "MCG",
                           .default=Units))
meps = meps %>%
  mutate(Units = case_when(Drug == "ALBUTEROL" & Strength=="0.09/1" ~ "MG",
                           .default=Units))
meps = meps %>% 
  mutate(Units = case_when(Drug=="ALBUTEROL" & Strength=="90" ~ "MCG",
                           .default=Units))

# Remove whitespace
meps = meps %>%
  mutate_all(~str_squish(.))

# fix miscoded units for combination drugs
meps = meps %>%
  mutate(Units = case_when((grepl("/", Drug)==TRUE | grepl("-", Drug)==TRUE)
                           & Units=="MG" ~ "MG/MG",
                           .default = Units))
meps = meps %>%
  mutate(Units = case_when((grepl("-", Drug)==TRUE | grepl("/", Drug)==TRUE)
                           & Units=="MCG" ~ "MCG/MCG",
                           .default = Units))
meps = meps %>%
  mutate(Units = case_when((grepl("-", Drug)==TRUE | grepl("/", Drug)==TRUE)
                           & Units=="G" ~ "G/G",
                           .default = Units))
meps = meps %>%
  mutate(Units = case_when((grepl("-", Drug)==TRUE | grepl("/", Drug)==TRUE)
                           & Units=="%" ~ "%/%",
                           .default = Units))
meps = meps %>%
  mutate(Units = case_when((grepl("/", Drug)==TRUE | grepl("-", Drug)==TRUE)
                           & Units=="ML" ~ "ML/ML/ML",
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

# Remove drug names with dashes that are on the drugs_to_remove list early to 
# avoid causing issues during combination drug handling
drug_names_to_remove = c("OMEGA-3 POLYUNSATURATED FATTY ACIDS",
                         "ALPHA-LIPOIC ACID","ANTI-INFECTIVES",
                         "THIAZIDE AND THIAZIDE-LIKE DIURETICS",
                         "STEROIDS WITH ANTI-INFECTIVES",
                         "NONSTEROIDAL ANTI-INFLAMMATORY AGENTS")
meps = meps %>% filter(!Drug %in% drug_names_to_remove)

# Remove whitespace
meps = meps %>%
  mutate_all(~str_squish(.))

# Handle dash delimiters - no slash delimiters after data curation

# Make dummy column of all zeros initially
meps = meps %>%
  mutate(Is_Combo = rep(0, nrow(meps)))

# Make a duplicate row with Is_Combo set to 1 if drug is a combination drug

# Identify rows where 'Drug' contains a dash
rows_with_dash = meps %>% filter(grepl("-", Drug))

# Duplicate these rows and set 'Is_Combo' to 1
duplicated_rows = rows_with_dash %>% mutate(Is_Combo = 1)

# Combine the original dataframe with the duplicated rows
meps = bind_rows(meps, duplicated_rows) %>% arrange(row_number())

# Conditionally split the Units column based on the presence of a dash in the
# Drug column (all units with slash are not combos)
meps = meps %>%
  mutate(
    Units_Parts = if_else(
      grepl("-", Drug),  
      str_split(Units, "-", simplify = TRUE),  
      Units  
    )
  )

# Split the Drug, Strength columns into parts before and after the dash or slash
meps = meps %>%
  mutate(
    Drug_Parts = str_split(Drug, "-", simplify = TRUE),
    Strength_Parts = str_split(Strength, "[-/]", simplify = TRUE)
  )

# Use if_else to conditionally assign the correct part based on the Is_Combo
# column
meps = meps %>%
  mutate(
    Drug = if_else(Is_Combo == 0, Drug_Parts[, 1],
                   if_else(Is_Combo == 1, Drug_Parts[, 2],Drug)),
    Strength = if_else(Is_Combo == 0, Strength_Parts[, 1],
                       if_else(Is_Combo == 1, Strength_Parts[, 2], Strength)),
    Units = if_else(Is_Combo == 0, Units_Parts[, 1],
                    if_else(Is_Combo == 1, Units_Parts[, 2],Units))
  )

# Drop the helper columns
meps = meps %>%
  select(-Is_Combo, -Drug_Parts, -Strength_Parts, -Units_Parts)

# Generate drug list to loop through

# get list of all drugs listed in Drug column
drugs = unique(meps$Drug)

# remove missing drug names from drug list
drugs = na.omit(drugs)

# remove any reintroduced whitespace from drugs vector
drugs = trimws(drugs)

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

# remove any duplicates introduced
drugs = unique(drugs)

# remove extra periods from data
meps$Strength = gsub("\\.+$", "", meps$Strength)

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
  
  # Conversions to correct units in Strength column
  
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
meps_clean <- meps_clean %>%
  mutate(Form = case_when(
    Form %in% c("TABS", "TB24", "TAB", "CAPS", "CAP-Capsule", "TBEC", "CPDR", 
                "TBDP", "T12A", "SYRP", "CP24", "TBCR", "CPCR", "TB12", "CHEW", 
                "CPEP", "ECT", "CPSP", "CTB", "SYR", "CP12", "PACK", "TBED", "CHER", "SOCT", "TBPK", "TBDD", "CSDR") ~ "ORAL",
    Form %in% c("AEPB", "INH-Inhaler", "INH-Inhalant", "NEBU", "AERB", "ACC") ~ "INHALATION",
    Form %in% c("AERO", "AER", "AERS") & Administration_route == "NASAL" ~ "NASAL",
    Form %in% c("AERO", "AER", "AERS") & Drug %in% c("FLUTICASONE", "ALBUTEROL", "TIOTROPIUM", "MOMETASONE", "LEVALBUTEROL", "CICLESONIDE", "SALMETEROL") ~ "INHALATION",
    Form %in% c("SUSP", "SUS", "SUSR") & Administration_route == "NASAL" ~ "NASAL",
    Form %in% c("SUSP", "SUS", "SUSR") & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Form %in% c("SUSP", "SUS", "SUSR") & Administration_route == "OTIC" ~ "OTIC",
    Form %in% c("SUSP", "SUS", "SUSR") & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form %in% c("SUSP", "SUS", "SUSR") & Drug %in% c("IBUPROFEN", "ACETAMINOPHEN", "CIPROFLOXACIN", "FAMOTIDINE", "CEFDINIR", "NAPROXEN", "AMOXICILLIN", "AZITHROMYCIN", "NITROFURANTOIN", "OXCARBAZEPINE", "OSELTAMIVIR", "CARBAMAZEPINE", "CEPHALEXIN") ~ "ORAL",
    Form %in% c("SUSP", "SUS", "SUSR") & Drug == "BUDESONIDE" ~ "INHALATION",
    Form %in% c("SUSP", "SUS", "SUSR") & Drug == "METHYLPREDNISOLONE" ~ "INTRAMUSCULAR/INTRA-ARTICULAR/SOFT TISSUE/INTRALESIONAL",
    Form %in% c("SUSP", "SUS", "SUSR", "SUSY") & Drug == "MEDROXYPROGESTERONE" ~ "INTRAMUSCULAR",
    Form %in% c("SOLN", "SOL", "SOLR") & Administration_route == "NASAL" ~ "NASAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Form %in% c("SOLN", "SOL", "SOLR") & Administration_route == "OTIC" ~ "OTIC",
    Form %in% c("SOLN", "SOL", "SOLR") & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug %in% c("METFORMIN", "OXYCODONE", "FUROSEMIDE", "GABAPENTIN", "PREDNISONE", "ASPIRIN", "PROMETHAZINE", "PREDNISOLONE", "ONDANSETRON", "ARIPIPRAZOLE", "LEVETIRACETAM", "CETIRIZINE", "PENICILLIN V POTASSIUM", "LORATADINE", "ALENDRONATE", "MORPHINE", "LACTULOSE") ~ "ORAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "SUMATRIPTAN" ~ "NASAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "ACETAMINOPHEN" & NDC == "43825010201" ~ "INTRAVENOUS",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "ACETAMINOPHEN" & NDC %in% c("00536012297", "00904701416") ~ "ORAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "BUDESONIDE" ~ "INHALATION",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "PANTOPRAZOLE" ~ "INTRAVENOUS",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug %in% c("KETOROLAC", "HALOPERIDOL") ~ "INTRAMUSCULAR",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "NITROGLYCERIN" ~ "SUBLINGUAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "METHOTREXATE" & NDC != "63323012310" & NDC != "00143951910" ~ "INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "METHOTREXATE" & NDC == "63323012310" ~ "INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTERIAL",
    Form %in% c("SOLN", "SOL", "SOLR") & Drug == "METHOTREXATE" & NDC == "00143951910" ~ "INTRA-ARTERIAL/INTRATHECAL/INTRAVENOUS/INTRAMUSCULAR",
    Form %in% c("CREA", "CRE", "SHA", "SHAM", "LOT", "LOTN", "PSTE", "OIL", "SWAB", "PADS") ~ "TOPICAL",
    Form %in% c("OINT", "OIN") & Administration_route== "TOPICAL" ~  "TOPICAL",
    Form %in% c("OINT", "OIN") & Administration_route== "OPHTHALMIC" ~  "OPHTHALMIC",
    Form %in% c("OINT", "OIN") & Drug== "NITROGLYCERIN" ~  "TRANSDERMAL",
    Form %in% c("Eye drops", "EMUL") ~ "OPHTHALMIC",
    Form == "Drops" & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Form == "Drops" & Administration_route == "OTIC" ~ "OTIC",
    Form == "SPR" & Administration_route == "NASAL" ~ "NASAL",
    Form == "SPR" & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form == "SPR" & Drug == "DIPHENHYDRAMINE" ~ "TOPICAL",
    Form %in% c("SUBL", "Sublingual") ~ "SUBLINGUAL",
    Form %in% c("PTWK", "PT24", "PTCH") ~ "TRANSDERMAL",
    Form == "FILM" ~ "BUCCAL",
    Form == "CONC" ~ "ORAL",
    Form == "INJ" & NDC == "00409131209" ~ "INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS",
    Form == "INJ" & Drug == "DULAGLUTIDE" ~ "SUBCUTANEOUS",
    Form == "INJ" & Drug == "HALOPERIDOL" ~ "INTRAMUSCULAR",
    Form == "SUPP" | NDC == "45802092341" ~ "RECTAL",
    Form == "FOAM" & Drug == "BUDESONIDE" ~ "RECTAL",
    Form == "FOAM" & Drug != "BUDESONIDE" ~ "TOPICAL",
    Form %in% c("LIQD", "LIQ") & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form %in% c("LIQD", "LIQ") & Drug %in% c("ACETAMINOPHEN", "DOCUSATE", "DIPHENHYDRAMINE", "LOPERAMIDE", "GUAIFENESIN", "PSEUDOEPHEDRINE", "TRIPROLIDINE", "DEXTROMETHORPHAN") ~ "ORAL",
    Form %in% c("Pen", "SOAJ", "SOPN") ~ "SUBCUTANEOUS",
    Form ==  "SRER" & Drug == "METHYLPHENIDATE" ~ "ORAL",
    Form %in%  c("SRER", "PRSY") & Drug == "ARIPIPRAZOLE" ~ "INTRAMUSCULAR",
    Form ==  "PRSY" & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form ==  "GEL" & Administration_route == "TOPICAL" ~ "TOPICAL",
    Form ==  "GEL" & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    Form == "SOLG" & Administration_route == "OPHTHALMIC" ~ "OPHTHALMIC",
    TRUE ~ Form   
  ))

# remove fluconazole with NDC 49884093547 because NDC corresponds to diclofenac
# and other data inaccuracies unclear for these rows
meps_clean = meps_clean[meps_clean$NDC != "49884093547", ]

# calculate daily frequency for each record
meps_clean = meps_clean %>%
  mutate(Daily_Frequency = Quantity/Day_Supply)

# calculate daily dosage for each record
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
# data
meps_clean = meps_clean %>%
  mutate(Excretion_percentage = case_when(NDC=="00409131209" ~ 5,
                                          NDC %in% c("61703035010","61703035038") ~ 100,
                                          NDC=="45802073030" ~ 9,
                                          NDC=="57896044305" ~ 100,
                                          NDC=="00009307303" ~ 100,
                                          NDC=="45802010901" ~ 100,
                                          NDC %in% c("00597007541", "00597007547") ~ 7,
                                          NDC %in% c("00904530760", "00536121429",
                                                     "00904530780", "00904530680",
                                                     "49348004534", "58657052816",
                                                     "00904530660", "00536077085",
                                                     "54838013540", "00536101001") ~ 100,
                                          NDC=="63323012310" ~ 100,
                                          NDC=="00143951910" ~ 100,
                                          NDC=="12547017004" ~ 100,
                                          NDC %in% c("64980051705", "60505057501",
                                                     "59651006605", "00065427325",
                                                     "61314027105", "70069000701",
                                                     "60505058604", "62332050203",
                                                     "00093768432", "61314027225",
                                                     "65862075705", "62332050105") ~ 77.2,
                                          .default=Excretion_percentage))

################################################################################
# add in population data
################################################################################

# Load 2020 population data
meps_pop = read.delim("data/MEPS_pop_data_2020.txt", header=TRUE, sep = "\t",
                      colClasses = "character", dec = ".")

# prep meps_pop to link to meps_clean
meps_pop = meps_pop %>%
  rename(`Person_ID` = DUPERSID) %>%
  rename(`Household_ID` = DUID) %>%
  select(c("Person_ID","Household_ID", "AGE20X", "SEX", "FAMINC20",
           "INSCOV20","RACETHX")) # these include age, sex, family income,
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
  filter(!(Person_ID %in% c("2320963101","2323677101","2323677102",
                            "2465567101")))

# Add year for integration
meps_clean = meps_clean %>%
  mutate(Year="2020")

# export data as csv
write.csv(meps_clean,"data/meps_clean_2020.csv")