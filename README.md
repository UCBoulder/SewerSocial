# PharmShed: Modeling pharmaceutical influent concentrations in diverse sewersheds

The purpose of this study is to predict pharmaceutical influent concentrations in sewersheds with diverse demographics. Data are from the Agency for Healthcare Research and Quality's Medical Expenditure Panel Survey (MEPS) for the years 2014, 2016, 2018, 2020, 2021, and 2022. These data include household-reported prescription consumption data for a nationally representative sample of Americans and demographic data for the person to whom each medication is prescribed (notably, these data are separate but linked by a common identifier column). The data from MEPS are used to train 3 models: a decision tree model, a random forest model, and a k-nearest neighbors (KNN) model. Principal component analysis is performed for the KNN model to improve model performance. The models all predict the pharmaceuticals prescribed from each person's demographics with varying accuracy. Ultimately, the model output will be used to back-calculate the wastewater influent concentration of each pharmaceutical in a sewershed and compared with literature-reported influent concentrations for various pharmaceuticals.

## Pacakge Dependencies
The code for this study is written in both R and Python. For the R scripts, the following packages are required: stringdist, data.table, tidyverse, readxl, haven. For the Python scripts, the following packages are required: pandas, numpy, matplotlib, seaborn, plotly, sklearn.

## Utility of Each Program
- meps_processor_yyyy.R - each script with this naming format takes in the given year's MEPS prescription and demographic data as MEPS_data_yyyy.txt and MEPS_pop_data_yyyy.txt, respectively. These are pre-processing scripts that clean the data and integrate the prescription data and the demographic data by the linking variable Person ID. The output is a single, cleaned csv file called mesp_clean_yyyy.csv that combines the prescription and demographic data for the given year.
- meps_processor_functions.R - functions used in the meps_processor_yyyy.R scripts.
- meps_scraper.R - converts the MEPS data files for 2014 and 2016 from ssp format to txt format. Takes in MEPS_data_2014.ssp, MEPS_data_2016.ssp, MEPS_pop_data_2014.ssp, MEPS_pop_data_2016.ssp and returns txt versions of each.
- integrator.R - combines the cleaned data files (meps_clean_yyyy.csv) for each year into one file. Returns integrated_data.csv.
- decision_tree_model.ipynb - takes in integrated_data.csv and trains and tests a decision tree model to predict drug name from demographic data.
- knn_model.ipynb - takes in each meps_clean_yyyy.csv file and trains and tests a multi-output regressor KNN model to predict drug name and concentration from demographic data.
- random_forest.ipynb - takes in integrated_data.csv and trains and tests a random forest model to predict drug name from demographic data.

## Data/Functional Dependencies
The data are too large to upload directly to GitHub, but they are available [here](https://drive.google.com/file/d/12m6rsRkr8EXtKQ7z9PEojX0gaLQEkSgG/view?usp=sharing). The data directory can be downloaded as a zip file and extracted into the same directory as the scripts. Note that the MEPS_data_yyyy.txt and MEPS_pop_data_yyyy.txt files for years 2018, 2020, 2021, and 2022 were directly downloaded as xlsx files from the [MEPS Prescribed Medicines files website](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_results.jsp?cboDataYear=All&cboDataTypeY=2%2CHousehold+Event+File&buttonYearandDataType=Search&cboPufNumber=All&SearchTitle=Prescribed+Medicines) and the [MEPS Full-Year Population Characteristics website](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_results.jsp?cboDataYear=All&cboDataTypeY=1%2CHousehold+Full+Year+File&buttonYearandDataType=Search&cboPufNumber=All&SearchTitle=Population+Characteristics). They were converted to txt files using Excel's file conversion functionality. The MEPS_data_yyyy.txt and MEPS_pop_data_yyyy.txt files for years 2014 and 2016 were downloaded from the same websites as ssp files and are converted to txt files via the meps_scraper.R script. The following lists the data files and functions required for each script:
- meps_processor_2014.R: MEPS_data_2014.txt, MEPS_pop_data_2014.txt, drugs_to_remove.xlsx, meps_processor_functions.R
- meps_processor_2016.R: MEPS_data_2016.txt, MEPS_pop_data_2016.txt, drugs_to_remove.xlsx, meps_processor_functions.R
- meps_processor_2018.R: MEPS_data_2018.txt, MEPS_pop_data_2018.txt, drugs_to_remove.xlsx, meps_processor_functions.R
- meps_processor_2020.R: MEPS_data_2020.txt, MEPS_pop_data_2020.txt, drugs_to_remove.xlsx, meps_processor_functions.R
- meps_processor_2021.R: MEPS_data_2021.txt, MEPS_pop_data_2021.txt, drugs_to_remove.xlsx, meps_processor_functions.R
- meps_processor_2022.R: MEPS_data_2022.txt, MEPS_pop_data_2022.txt, drugs_to_remove.xlsx, meps_processor_functions.R
- meps_scraper.R: MEPS_data_2014.ssp, MEPS_data_2016.ssp, MEPS_pop_data_2014.ssp, MEPS_pop_data_2016.ssp
- integrator.R: meps_clean_2014.csv, meps_clean_2016.csv, meps_clean_2018.csv, meps_clean_2020.csv, meps_clean_2021.csv, meps_clean_2022.csv
- decision_tree_model.ipynb: integrated_data.csv
- knn_model.ipynb: meps_clean_2014.csv, meps_clean_2016.csv, meps_clean_2018.csv, meps_clean_2020.csv, meps_clean_2021.csv, meps_clean_2022.csv
- random_forest.ipynb: integrated_data.csv