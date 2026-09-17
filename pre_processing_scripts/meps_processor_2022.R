################################################################################
# Title: meps_processor_2022.R

# Description:
# Loads, cleans and exports data from the Medical Expenditure Panel Survey
# 2022 Prescribed Medicines File and 2022 Full Year Population Characteristics
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
# Load and clean Medical Expenditure Panel Survey (MEPS) 2022 prescription data.
################################################################################

# load MEPS dataset (all columns read in as characters)
meps = read.delim("data/MEPS_data_2022.txt", header=TRUE, sep = "\t",
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
meps$Drug = gsub(paste0("\\b", " IV", "\\b"), "", meps$Drug,
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
# will need to check for others in any new dataset
miscoded_units = c("OTHER", "U/ML", "U/ML/U/ML", "UNIT", "UNIT/ML", "UNIT/GM",
                   "U/GM", "UT/ML", "MCG/OTHER", "%/OTHER","DOSE","","UNIT/ML")
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
                                             "MG/MG/ML","MG/ML/ML"),
                          correct_units = c("MCG", "MG", "MCG/MG", "MG", "MG",
                                            "G","MG/MG", "MCG", "MCG", "MG", "MCG",
                                            "G", "G/ML", "MCG/MCG","MCG","MCG/MCG",
                                            "MG/MG","MG"))
for (i in seq_len(nrow(fix_units_df))){
  meps = fix_drug_coding(meps, meps$Units, "Units", fix_units_df[i,1],
                         fix_units_df[i,2])
}

# remove miscoded drugs or drugs with vague/missing info or otherwise incorrect
meps = meps[is.na(meps$Units)==FALSE,]
meps = meps[is.na(meps$Drug)==FALSE,]
meps = remove_entries(meps, meps$Form, "OTHER")
meps = remove_entries(meps, meps$Units, "ML")
meps = meps %>%
  mutate(Units = case_when(NDC=="17856158601" ~ "MG/MG",
                           .default=Units))

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

# Handle dash/slash delimiters - check IU for insulin glargine, MG/MG ones

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
  mutate(Units=case_when(Units=="MG/MG" ~ "MG",
                         .default=Units))

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

# replace empty strings with NA
meps <- meps %>%
  mutate(across(everything(), ~na_if(., "")))

# final drug list
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

  # Convert IU to MCG (100 IU insulin glargine is 3.64 mcg; these are the only
  # ones left with IU units)
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

# take care of imprecise scientific notation strengths 

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
api_results = read.csv("meps_2022_from_api.csv")
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
    Form %in% c("TAB","TABS","TB24","CPDR","TBEC","CAP-Capsule",
                "CER","TBDP","CAPS","CP24","CPSP","TBCR","CHEW",
                "TB12","C12A","T12A","CPCR","ELIX","CPEP","ECT",
                "CTB", "SYRP","CP12","CAP-Caplets","SYR") ~ "ORAL",
    Form %in% c("INH-Inhaler","INH-Inhalant","RESPIRATORY (INHALATION)",
                "NEBU") ~ "INHALATION",
    Form %in% c("INTRAOCULAR") ~ "OPHTHALMIC",
    Form %in% c() ~ "TOPICAL",
    Form %in% c() ~ "INTRAVENOUS",
    Form %in% c() ~ "OTIC",
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
#accounted for below)
meps_clean <- meps_clean %>%
  mutate(Form = case_when(NDC %in% c("00054329446","00054006447","16714067102",
                                     "00406800330","27808008201","00054023763",
                                     "00054023749","00121072104","00054039063",
                                     "62559015116","00054317757","42192060816",
                                     "42571036007","00121075908","50383004224",
                                     "50383004248","00904701420","00904701416",
                                     "31722056424","54838057280","51991083716",
                                     "00904676520","45802097426","54838055240",
                                     "00121087316","00527512570","50383079516",
                                     "50383077916","70408014634","60687025240",
                                     "50383077931","00904530920","00904530909",
                                     "45802089726","68001043594","00045012304",
                                     "00904676620","70677011801","45802020326",
                                     "50383081016","36800064526","00536130485",
                                     "00904584709","58657050916","00536131485",
                                     "00904676320","00536126659","00245107130",
                                     "68094050359","68788826003","00093413664",
                                     "00143988815","00143988915","00093416173",
                                     "00143988701","00143988775","65862070655",
                                     "00781615746","00093416178","00143988901",
                                     "65862007175","65862007150","00781604155",
                                     "00093415580","65862007101","00143988601",
                                     "00143988801","00781604146","00143988980",
                                     "00781615757","00143988750","00781615752",
                                     "65862070701","00093202631","70710146002",
                                     "59762312001","59762314001","00093202623",
                                     "59762313001","68180065701","68180044102",
                                     "00093417774","67877054588","68180015001",
                                     "50419077301","65862021960","68180072304",
                                     "16714039301","65862021801","67877054898",
                                     "67877054798","00093413764","65862021901",
                                     "68180072305","16714039202","67877054888",
                                     "65862021860","65862059601","00093412773",
                                     "59762305102","69097052834","00186402001",
                                     "69097052934","00003376474","70954005930",
                                     "00603533815","70954005830","00591505221",
                                     "00603533715","00603533731","00603533831",
                                     "00603459315","42806040021","59762444002",
                                     "59746000103","68382091634","72647033104",
                                     "00781502207","68001000501","45963043864",
                                     "00113057826","00093300993") ~ "ORAL",
                          NDC %in% c("12547017157") ~ "TOPICAL",
                          NDC %in% c("68968555203","00378910493",
                                     "43598089530","00597003134",
                                     "00591350904","00591351004",
                                     "00378087399","00597003334",
                                     "00591350804","51862045504") ~ "TRANSDERMAL",
                          NDC %in% c() ~ "INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTERIAL",
                          NDC %in% c("72572012001") ~ "INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTICULAR/SOFT TISSUE/INTRALESIONAL",
                          NDC %in% c("61703035038") ~ "INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS",
                          NDC %in% c("00143951910") ~ "INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS/INTRATHECAL",
                          NDC %in% c("16729027730") ~ "INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTERIAL/INTRATHECAL",
                          NDC %in% c() ~ "INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS/INTRA-ARTERIAL",
                          NDC %in% c("53217013201") ~ "INTRAMUSCULAR/INTRASYNOVIAL/SOFT TISSUE",
                          NDC %in% c("55150017301","55111069312",
                                     "00173047800","00169280015",
                                     "00169406013","00169406012") ~ "SUBCUTANEOUS",
                          NDC %in% c("70518261300","00093681773",
                                     "69097032187","00115168974",
                                     "00093681673","00093681555",
                                     "76282064138","68180098430",
                                     "00487970130","00487960101",
                                     "00093681573","16714001930",
                                     "76282064238","47335063249",
                                     "00173087410","00173060002",
                                     "00173060202","00173088810",
                                     "00173087610","00186091612",
                                     "00186091706","00173071920",
                                     "00173071820","00173072020",
                                     "66993008096","66993007896",
                                     "66993007996","54569566300",
                                     "50090605800","70518285400",
                                     "68788722906","50090573500",
                                     "68788747702","54569637000",
                                     "00093317431","00173068220",
                                     "00781729685","59310057922",
                                     "00054074287","00254100752",
                                     "66758095985","00597016061",
                                     "00597010061","00378932232",
                                     "00597010051") ~ "INHALATION",
                          NDC %in% c("72266011901","00548540000",
                                     "00548540025","00703680104",
                                     "00703680101","66993037083",
                                     "65757040103","59148007280",
                                     "50458030611","66993037179",
                                     "59762453802") ~ "INTRAMUSCULAR",
                          NDC %in% c("00409177805","63323039810",
                                     "00008092360","00008092355") ~ "INTRAVENOUS",
                          NDC %in% c("63323028002","00409610210",
                                     "63323066401","72611072225",
                                     "76045010610") ~ "INTRAVENOUS/INTRAMUSCULAR",
                          NDC %in% c("69097064448","45802061906",
                                     "00527185943") ~ "NASAL",
                          NDC %in% c() ~ "OTIC",
                          NDC %in% c("45802021001") | Form %in% c("SUBL","Sublingual") ~ "SUBLINGUAL",
                          NDC %in% c() ~ "OPHTHALMIC",
                          NDC %in% c("58914030180","65649065103",
                                     "45802075930") ~ "RECTAL",
                          TRUE ~ Form
  ))

# filter the records with unknowable forms
meps_clean <- meps_clean %>%
  filter(!(is.na(NDC) & is.na(Administration_route)))

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
# to the prioritization is if expert curation indicates otherwise (e.g., subcutaneous
# being similar to intravenous)
meps_clean = meps_clean %>%
  mutate(Excretion_percentage = case_when(is.na(Excretion_percentage)==TRUE & Drug=="FUROSEMIDE" ~ 70,
                                          is.na(Excretion_percentage)==TRUE & Drug=="DEXAMETHASONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="DIPHENHYDRAMINE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="TRIAMCINOLONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="KETOROLAC" ~ 66,
                                          is.na(Excretion_percentage)==TRUE & Drug=="OLOPATADINE" ~ 77.2,
                                          is.na(Excretion_percentage)==TRUE & Drug=="NITROGLYCERIN" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="METHYLPREDNISOLONE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="METHOTREXATE" ~ 100,
                                          is.na(Excretion_percentage)==TRUE & Drug=="TIOTROPIUM" ~ 7,
                                          .default=Excretion_percentage))

################################################################################
# add in population data
################################################################################

# Load 2022 population data
meps_pop = read.delim("data/MEPS_pop_data_2022.txt", header=TRUE, sep = "\t",
                  colClasses = "character", dec = ".")

# prep meps_pop to link to meps_clean
meps_pop = meps_pop %>%
  rename(`Person_ID` = DUPERSID) %>%
  rename(`Household_ID` = DUID) %>%
  select(c("Person_ID","Household_ID", "AGE22X", "SEX", "FAMINC22",
           "INSCOV22","RACETHX")) # these include age, sex, family income,
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
  filter(!(Person_ID %in% c("2794838101","2683576102","2791491101","2791491103",
                            "2683096102","2791491102","2683096101","2683096103",
                            "2683096104","2683096105","2683096106")))

# Add year for integration
meps_clean = meps_clean %>%
  mutate(Year="2022")

# export data as csv
fwrite(meps_clean, "data/meps_clean_2022.csv")