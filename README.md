# PharmShed: Predicting wastewater pharmaceutical concentrations at the sewer-shed scale from community demographics

The purpose of this study is to predict active pharmaceutical ingredient (API) influent concentrations and mass loads in sewersheds with diverse demographics. Data are from the Agency for Healthcare Research and Quality's Medical Expenditure Panel Survey (MEPS) for the years 2014, 2016, 2018, 2020, 2021, and 2022. These data include household-reported prescription consumption data for a nationally representative sample of Americans and demographic data for the person to whom each medication is prescribed (notably, these data are separate but linked by a common identifier column). The data from 2014-2021 are used for training with cross-validation and testing, while the data from 2022 are reserved for additional validation. External data are used for testing the model-predicted wastewater mass loads against literature-reported mass loads. In total, 5 base models (XGBoost, TabICL, RealMLP, k-nearest neighbor with Hassanat distance, and support vector machine) are trained on the 2014-2021 data. From this training, a specialist map is created for each API, linking it to the model that provides the highest recall. This is used to build a winner-takes-all ensemble model from the 5 base models. The ensemble model drops any APIs for which recall is less than 75%. Validation and testing scripts for the 2022 and external data are additionally provided. Scripts performing statistical analyses on model outputs are also provided. 

# Package Dependencies
The code for this study is written in both R and Python. For the Python scripts, load the requirements.txt file to access the Python environment needed to run all models and scripts in Python. For the R scripts, use the .Rprofile and renv.lock and run renv::restore() in the console in RStudio.

# Utility of Each Program
## Pre-processing scripts:
- get_ndc_codes_shareable.R - queries the openFDA API for excretion route data for each drug.
- meps_processor_{year}.R - each script with this naming format takes in the given year's MEPS prescription and demographic data as MEPS_data_{year}.txt and MEPS_pop_data_{year}.txt, respectively. These are pre-processing scripts that clean the data and integrate the prescription data and the demographic data by the linking variable Person ID. The output is a single, cleaned csv file called meps_clean_{year}.csv that combines the prescription and demographic data for the given year.
- meps_processor_functions.R - functions used in the meps_processor_{year}.R scripts.
- meps_scraper.R - converts the MEPS data files for 2014 and 2016 from ssp format to txt format. Takes in MEPS_data_2014.ssp, MEPS_data_2016.ssp, MEPS_pop_data_2014.ssp, MEPS_pop_data_2016.ssp and returns txt versions of each.
- integrator.R - combines the cleaned data files (meps_clean_{year}.csv) for each year into one file. Returns integrated_data.csv.
- superdataset.ipynb - merges demographic and prescription datasets for 2014-2021 (the training data).
- super_data_2022.ipynb - merges demographic and prescription datasets for 2022 (the validation data).
- correlation_checker.R - checks for correlations between features in super_integrated_data.csv.

## Base model scripts:
- ann_super_dataset.ipynb - trains the Approximate Nearest Neighbors model with Hassanat distance metric on the 2014-2021 data.
- realmlp_super_dataset.ipynb - trains the RealMLP model on the 2014-2021 data.
- svm_super_dataset.ipynb - trains the SVM model on the 2014-2021 data.
- tabicl_model.py - trains the TabICL model on the 2014-2021 data.
- xgboost_super_dataset.ipynb - trains the XGBoost model on the 2014-2021 data.
- xgboost_super_threshold_tuning.ipynb - applies threshold tuning to the XGBoost model.

## Base model validation scripts:
- ann_internal_validation.py - implements the saved Approximate Nearest Neighbors model on the 2022 validation dataset.
- realmlp_internal_validation.py - implements the saved RealMLP model on the 2022 validation dataset.
- svm_internal_validation.py - implements the saved SVM model on the 2022 validation dataset.
- tabicl_internal_validation.py - implements the saved TabICL model on the 2022 validation dataset.
- xgboost_internal_validation.py - implements the saved XGBoost model on the 2022 validation dataset.

## Ensemble and full validation script:
- ensemble_training_internal_validation_original.py - constructs the ensemble based on historical recall scores and conducts validation for the full ensemble on the 2022 validation dataset.

## Synthetic external data generation scripts:
- app.R - SeweRx, an R Shiny app developed in [this paper](https://doi.org/10.1002/wer.70357), which contains the PharmUse and PharmFlush modules described in the paper. It also includes a Synthetic Population tab that generates the synthetic demographic datasets used in the synthetic external data generation for PharmShed. The current, beta version of the app is available [here](nessagracesapps.shinyapps.io/shiny_app/). Future versions of the app will include more user-friendly features and instructions on how to integrate with the upcoming PharmShed Python package. The app will eventually be hosted on its own server, and the website to access it is subject to change when this happens. The current version is size-restricted because it uses the free version of shinyapps.io's server.
- prescription_imputation_function.R - uses the MEPS demographic and prescription feature joint distributions to simulate the prescription feature distribution in a given community's synthetic external demographic data generated by app.R.

## External data model application scripts:
- synth-data-base-model-run-threshold-loop.py - runs the base models for external, synthetically generated data for a given community.
- synth-data-facility-subsets.py - randomly samples from the community-level population data generated in synth-data-base-model-run-threshold-loop.py to ensure predictions are for datasets of the correct population size for each facility.
- ensemble_model_application_loop.py - applies the ensemble model for loaded external base model output generated by synth-data-base-model-run-threshold-loop.py and synth-data-facility-subsets.py.

## Statistical analysis scripts:
- analyze_city_differences.py - conducts chi-square and Kruskal-Wallis tests to determine if there are statistically significant differences in prescription rate or mass load of APIs between the communities of Sandwich, MA, Urbana-Champaign, IL, and Clark County, NV. Also calculates fold change in mass load between these communities.
- analyze_drug_patterns_updated.py - conducts chi-square and Mann-Whitney U tests to determine if there are statistically significant differences in features between APIs included in the model and excluded from the model. Also checks if there are statistically significant differences in features for those on a prescription versus in the "no prescriptions" class. Finally, calculates fold changes in numeric features between APIs and performs a logistic regression to compute odds ratios of having different features.

# Data/Functional Dependencies
The MEPS data are available [here](https://data.mendeley.com/preview/wfmjbyjmk2?a=1820bd8a-0208-4a4a-b9bf-6dddd230003a). The data directory can be downloaded as a zip file and extracted into the same directory as the scripts. The following provides instructions to generate/access the data files and functions required for each script:

## Pre-processing scripts:
- get_ndc_codes_shareable.R:
    - Obtain an API key from [this link](https://open.fda.gov/apis/authentication/). A csv file must be uploaded in the script under section 4, Execution. This can be any csv file with an NDC column.
- meps_processor_2014.R:
    - Use the link to the MEPS data above to access MEPS_data_2014.txt and MEPS_pop_data_2014.txt. Access drugs_to_remove.xlsx and in the data_dependencies folder. meps_processor_functions.R is included in the pre_processing_scripts folder.
- meps_processor_2016.R:
    - Use the link to the MEPS data above to access MEPS_data_2016.txt and MEPS_pop_data_2016.txt. Access drugs_to_remove.xlsx and in the data_dependencies folder. meps_processor_functions.R is included in the pre_processing_scripts folder.
- meps_processor_2018.R:
    - Use the link to the MEPS data above to access MEPS_data_2018.txt and MEPS_pop_data_2018.txt. Access drugs_to_remove.xlsx and in the data_dependencies folder. meps_processor_functions.R is included in the pre_processing_scripts folder.
- meps_processor_2020.R:
    - Use the link to the MEPS data above to access MEPS_data_2020.txt and MEPS_pop_data_2020.txt. Access drugs_to_remove.xlsx and in the data_dependencies folder. meps_processor_functions.R is included in the pre_processing_scripts folder.
- meps_processor_2021.R:
    - Use the link to the MEPS data above to access MEPS_data_2021.txt and MEPS_pop_data_2021.txt. Access drugs_to_remove.xlsx and in the data_dependencies folder. meps_processor_functions.R is included in the pre_processing_scripts folder.
- meps_processor_2022.R:
    - Use the link to the MEPS data above to access MEPS_data_2022.txt and MEPS_pop_data_2022.txt. Access drugs_to_remove.xlsx and in the data_dependencies folder. meps_processor_functions.R is included in the pre_processing_scripts folder.
- meps_scraper.R:
    - Download MEPS_data_2014.ssp, MEPS_data_2016.ssp, MEPS_pop_data_2014.ssp, and MEPS_pop_data_2016.ssp directly as described in the link to the MEPS data.
- integrator.R:
    - Run all meps_processor_{year}.R scripts to generate meps_clean_2014.csv, meps_clean_2016.csv, meps_clean_2018.csv, meps_clean_2020.csv, meps_clean_2021.csv,and meps_clean_2022.csv.
- superdataset.ipynb: 
    - Run meps_processor_2014.R through meps_processor_2022.R. Then run integrator.R. This will give the integrated_data.csv and prescription.csv datasets required to run superdataset.ipynb.
- super_data_2022.ipynb:
    - Run meps_processor_2014.R through meps_processor_2022.R. Then run integrator.R. This will give the data_2022.csv and prescription_2022.csv datasets required to run super_data_2022.ipynb.
- correlation_checker.R: 
    - Run superdataset.ipynb to generate super_integrated_data.csv.

## Base model scripts:
- ann_super_dataset.ipynb:
    - Run superdataset.ipynb to generate the super_integrated_data.csv dataset.
- realmlp_super_dataset.ipynb:
    - Run superdataset.ipynb to generate the super_integrated_data.csv dataset.
- svm_super_dataset.ipynb: 
    - Run superdataset.ipynb to generate the super_integrated_data.csv dataset.
- tabicl_model.py:
    - Run superdataset.ipynb to generate the super_integrated_data.csv dataset.
- xgboost_super_dataset.ipynb:
    - Run superdataset.ipynb to generate the super_integrated_data.csv dataset.
- xgboost_super_threshold_tuning.ipynb:
    - Run superdataset.ipynb to generate the super_integrated_data.csv dataset. Run super_data_2022.ipynb to generate the super_data_2022.csv dataset. Run xgboost_super_dataset.ipynb to generate xgboost_all_versions_comparison.csv, xgboost_super_2022_per_drug_metrics.csv, xgboost_super_age_medians.joblib, xgboost_super_strength_medians.joblib, xgboost_super_day_supply_medians.joblib, and xgboost_super_final_model.ubj.

## Base model validation scripts:
- ann_internal_validation.py:
    - Run super_data_2022.ipynb to generate the super_data_2022.csv dataset. Run xgboost_super_dataset.ipynb to generate xgboost_super_label_encoder.joblib (same across all models). Run ann_super_dataset.ipynb to generate knn_svm_preprocessor.joblib (saved in that notebook as knn_super_preprocessor.joblib and same between ANN/KNN and SVM models), knn_final_prebuilt_index.joblib, knn_super_clf_only.joblib, knn_super_age_medians.joblib, knn_super_strength_medians.joblib, and knn_super_day_supply_medians.joblib.
- realmlp_internal_validation.py:
    - Run super_data_2022.ipynb to generate the super_data_2022.csv dataset. Run xgboost_super_dataset.ipynb to generate xgboost_super_label_encoder.joblib (same across all models). Run realmlp_super_dataset.ipynb to generate realmlp_super_final_model.joblib, realmlp_super_age_medians.joblib, realmlp_super_strength_medians.joblib, and realmlp_super_day_supply_medians.joblib.
- svm_internal_validation.py:
    - Run super_data_2022.ipynb to generate the super_data_2022.csv dataset. Run xgboost_super_dataset.ipynb to generate xgboost_super_label_encoder.joblib (same across all models). Run svm_super_dataset.ipynb to generate knn_svm_preprocessor.joblib (saved in that notebook as svm_super_preprocessor.joblib and same between ANN/KNN and SVM models), svm_super_final_model.joblib, svm_super_age_medians.joblib, svm_super_strength_medians.joblib, and svm_super_day_supply_medians.joblib.
- tabicl_internal_validation.py:
    - Run super_data_2022.ipynb to generate the super_data_2022.csv dataset. Run xgboost_super_dataset.ipynb to generate xgboost_super_label_encoder.joblib (same across all models). Run tabicl_model.py to generate tabicl_trained_model.pkl, tabicl_super_age_medians.joblib, tabicl_super_strength_medians.joblib, and tabicl_super_day_supply_medians.joblib.
- xgboost_internal_validation.py: 
    - Run super_data_2022.ipynb to generate the super_data_2022.csv dataset. Run xgboost_super_dataset.ipynb to generate xgboost_super_label_encoder.joblib, xgboost_super_final_model.ubj, xgboost_super_age_medians.joblib, xgboost_super_strength_medians.joblib, and xgboost_super_day_supply_medians.joblib. Run xgboost_super_threshold_tuning.ipynb to generate xgboost_threshold_map.joblib. 

## Ensemble and full validation script:
- ensemble_training_internal_validation_original.py:
    - Run all base model validation scripts to generate the _super_proba_2022.csv and _super_validation_summary.csv files. Run all base model scripts to generate the _super_per_drug_recall.csv, xgboost_super_label_encoder.joblib, and xgboost_threshold_map.joblib files. Run super_data_2022.ipynb to generate the super_data_2022.csv dataset. 

## Synthetic external data generation scripts:
- app.R:
    - Access pharmuse.csv from data_dependencies. The pharmflush_functions.R file is included in the synthetic_external_data_generation_scripts folder with app.R. The file test_file.csv that is in that same folder can be used to upload to the Shiny app once deployed.
- prescription_imputation_function.R:
    - Run superdataset.ipynb to generate the super_integrated_data.csv dataset. Use the Synthetic Population tab in app.R to generate files of the form synthetic_population_{city}.csv.

## External data model application scripts:
- synth-data-base-model-run-threshold-loop.py:
    - Run prescription_imputation_function.R to generate the files of the form CITY_demo_rx.csv. Run all base model scripts to generate the final model files xgboost_super_final_model.ubj, knn_super_clf_only.joblib, svm_super_final_model.joblib, tabicl_trained_model.pkl, and realmlp_super_final_model.joblib. The base model scripts will also generate knn_svm_preprocessor.joblib, xgboost_super_label_encoder.joblib, knn_final_prebuilt_index.joblib, and xgboost_threshold_map.joblib. 

- synth-data-facility-subsets.py:
    - Run synth-data-base-model-run-threshold-loop.py to generate files in the form {city name}_{model name}_threshold_results.csv.gz.

- ensemble_model_application_loop.py:
    - Run prescription_imputation_function.R to generate the files of the form CITY_demo_rx.csv. Run synth-data-base-model-run-threshold-loop.py to generate files in the form {city name}_{model name}_threshold_results.csv.gz. Run synth-data-facility-subsets.py to generate files in the form {city name}_{facility name}_{model name}_threshold_results.csv.gz. Run xgboost_super_dataset.ipynb to generate xgboost_super_label_encoder.joblib. Run ensemble_training_internal_validation_original.py to generate specialist_map_ensemble.joblib and comparison_per_drug.joblib. Access pharmuse.csv via data_dependencies folder (more info in [this paper](https://doi.org/10.1002/wer.70357), where PharmUse is Table S4). 

## Statistical analysis scripts:
- analyze_city_differences.py:
    - Run ensemble_model_application_loop.py. This generates clark_county_nv_indiv_predictions_and_mass.csv, urbana_champaign_il_indiv_predictions_and_mass.csv, and sandwich_ma_indiv_predictions_and_mass.csv. These can be used to run the analyze_city_differences.py script. Notably, these CSV files can be swapped for other external data if other communities are simulated in ensemble_model_application_loop.py. 
- analyze_drug_patterns_updated.py:
    - Run super_data_2022.ipynb to generate super_data_2022.csv. Access common_years_drug_list.xlsx from data_dependencies folder.