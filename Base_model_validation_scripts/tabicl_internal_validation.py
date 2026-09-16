# This script runs the saved TabICL base model on the 2022 validation dataset.

# parallelize code
import os
import torch

# let OS manage threads and don't let PyTorch conflict with GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# check GPU availability
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True 

# Install necessary libraries
import pandas as pd
import numpy as np 
import joblib
from tabicl import TabICLClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report
)
from permetrics import ClassificationMetric
from sklearn.utils import shuffle
import gc

# Define functions to apply the medians saved from training for imputation
def _nearest_year(year, available_years):
    if year in available_years:
        return year
    return min(available_years, key=lambda y: abs(y - year))
 
def _assign_income_bracket(df, income_bin_edges, available_years):
    income_bracket = pd.Series(index=df.index, dtype='float64')
    for year, group in df.groupby('Year'):
        effective_year = _nearest_year(year, available_years)
        edges = income_bin_edges.get(effective_year)
        if edges is not None:
            binned = pd.cut(group['Family_income'], bins=edges, labels=False, include_lowest=True)
            income_bracket.loc[group.index] = binned + 1
    return income_bracket
 
def apply_age_medians(df, medians):
    df = df.copy()
    available_years = medians['available_years']
 
    df['income_bracket'] = _assign_income_bracket(df, medians['income_bin_edges'], available_years)
 
    effective_year = df['Year'].apply(lambda y: _nearest_year(y, available_years))
 
    hh_idx    = pd.MultiIndex.from_arrays([effective_year, df['Household_ID']])
    hh_lookup = pd.Series(hh_idx.map(medians['hh_medians']), index=df.index)
 
    grp_idx    = pd.MultiIndex.from_arrays([effective_year, df['income_bracket'], df['Insurance_coverage']])
    grp_lookup = pd.Series(grp_idx.map(medians['grp_medians']), index=df.index)
 
    yr_lookup = effective_year.map(medians['yr_medians'])
 
    df['Age'] = df['Age'].fillna(hh_lookup)
    df['Age'] = df['Age'].fillna(grp_lookup)
    df['Age'] = df['Age'].fillna(yr_lookup)
    df['Age'] = df['Age'].fillna(medians['global_median'])
    df.drop(columns=['income_bracket'], inplace=True)
    return df

def apply_strength_medians(df, medians):
    df = df.copy()
    mask = df['Drug'] != 'no prescriptions'

    yr_drug_idx = pd.MultiIndex.from_frame(df.loc[mask, ['Year', 'Drug']])
    yr_drug_lookup = pd.Series(yr_drug_idx.map(medians['yr_drug_medians']), index=df.loc[mask].index)
    drug_lookup = df.loc[mask, 'Drug'].map(medians['drug_medians'])

    df.loc[mask, 'Strength'] = df.loc[mask, 'Strength'].fillna(yr_drug_lookup)
    df.loc[mask, 'Strength'] = df.loc[mask, 'Strength'].fillna(drug_lookup)
    df.loc[mask, 'Strength'] = df.loc[mask, 'Strength'].fillna(medians['global_median'])
    return df

def apply_day_supply_medians(df, medians):
    df = df.copy()
    mask = df['Drug'] != 'no prescriptions'

    yr_drug_idx = pd.MultiIndex.from_frame(df.loc[mask, ['Year', 'Drug']])
    yr_drug_lookup = pd.Series(yr_drug_idx.map(medians['yr_drug_medians']), index=df.loc[mask].index)
    drug_lookup = df.loc[mask, 'Drug'].map(medians['drug_medians'])

    df.loc[mask, 'Day_Supply'] = df.loc[mask, 'Day_Supply'].fillna(yr_drug_lookup)
    df.loc[mask, 'Day_Supply'] = df.loc[mask, 'Day_Supply'].fillna(drug_lookup)
    df.loc[mask, 'Day_Supply'] = df.loc[mask, 'Day_Supply'].fillna(medians['global_median'])
    return df

def run_meps_2022_tabicl_pipeline(data_path, output_filename, encoder_path, model_path):
    print(f"\n--- Loading Components ---")
    le = joblib.load(encoder_path)
    model = joblib.load(model_path)

    numeric_features = ['Age', 'Family_income', 'Quantity', 'Strength', 'Day_Supply']
    categorical_features = ['Sex', 'Insurance_coverage', 'Race_ethnicity', 'Form']
    feature_cols = numeric_features + categorical_features

    # Load 2022 super dataset
    print(f"Reading dataset: {data_path}")
    data_2022 = pd.read_csv(data_path, sep=None, engine='python', encoding='utf-8-sig')

    # downcast data to preserve memory
    for col in data_2022.select_dtypes(include=['float64']).columns:
        data_2022[col] = data_2022[col].astype('float32')
    for col in data_2022.select_dtypes(include=['int64']).columns:
        data_2022[col] = data_2022[col].astype('int32')
    
    if 'Unnamed: 0' in data_2022.columns:
        data_2022 = data_2022.drop(columns=['Unnamed: 0'])

    # Same sentinel filling as training
    mask = data_2022['Drug'] == 'no prescriptions'
    data_2022.loc[mask, 'Form'] = data_2022.loc[mask, 'Form'].fillna('-1')

    # Drop any drugs in 2022 dataset not mapped during model training
    unseen = set(data_2022['Drug'].unique()) - set(le.classes_)
    if unseen:
        print(f"Unseen drugs in 2022 (will be dropped): {unseen}")
    data_2022 = data_2022[data_2022['Drug'].isin(le.classes_)].reset_index(drop=True)
    print(f"Rows after initial class filter: {len(data_2022):,}")

    # Cast feature types
    categorical_features = ['Sex', 'Insurance_coverage', 'Race_ethnicity', 'Form']
    for col in categorical_features:
        data_2022[col] = data_2022[col].astype('category')

    # Sequential hierarchical median imputation steps 
    print("Applying hierarchical missing value imputations...")
    age_medians = joblib.load('tabicl_super_age_medians.joblib')
    strength_medians = joblib.load('tabicl_super_strength_medians.joblib')
    day_supply_medians = joblib.load('tabicl_super_day_supply_medians.joblib')

     # Guard against a stale medians that predates the
    # nearest-year fallback logic. Without this check, a missing key would only
    # surface as a KeyError deep inside apply_age_medians, which is harder to
    # diagnose than a clear message here.
    required_age_keys = {
        'income_bin_edges', 'hh_medians', 'grp_medians',
        'yr_medians', 'global_median', 'available_years',
    }
    missing_keys = required_age_keys - set(age_medians.keys())
    assert not missing_keys, (
        f"tabicl_super_age_medians.joblib is missing required key(s): {missing_keys}. "
        "This usually means the file is stale — re-run fit_age_medians(super_df) "
        "in the training script and re-save tabicl_super_age_medians.joblib before "
        "running this inference script."
    )

    data_2022 = apply_age_medians(data_2022, age_medians)
    data_2022 = apply_strength_medians(data_2022, strength_medians)
    data_2022 = apply_day_supply_medians(data_2022, day_supply_medians)

    # set sentinel value for "no prescriptions" rows
    data_2022.fillna(-1, inplace=True)

    # Isolate targets and identifiers before transforming features
    y_2022_encoded = le.transform(data_2022['Drug'])
    obs_ids = data_2022['Observation_ID'].values if 'Observation_ID' in data_2022.columns else None

    X_2022 = data_2022[feature_cols].copy()

    print("Processing predictions in memory-safe batches...")

    # Winner-takes-all approach to select winners per row and compute performance metrics
    drug_class_list = list(le.classes_)
    if 'no prescriptions' in drug_class_list:
        no_presc_idx = drug_class_list.index('no prescriptions')
    else:
        no_presc_idx = None
        print("WARNING: 'no prescriptions' not found in le.classes_; "
                "low-confidence rows will keep their raw argmax instead of "
                "falling back to a no-prescriptions prediction.")

    # Accumulate hard predictions (small: one int per row) so we can score
    # against the 2022 ground truth after streaming, without holding the
    # full (rows x 217) probability matrix in memory.
    y_pred_all = np.empty(len(X_2022), dtype=int)

    first_batch = True

    # Batched Graph Querying & Optimized CSR Matrix Prediction Steps
    for i in range(0, len(X_2022), BATCH_SIZE):
        batch_x = X_2022[i : i + BATCH_SIZE]
        
        proba = model.predict_proba(batch_x)

        # Derive hard predictions from the RAW probabilities, before the
        # thresholding step below zeroes out low-confidence entries. 
        row_max = proba.max(axis=1)
        batch_pred_idx = proba.argmax(axis=1)
        if no_presc_idx is not None:
            batch_pred_idx[row_max <= THRESHOLD] = no_presc_idx
        y_pred_all[i : i + len(batch_x)] = batch_pred_idx
        
        proba_df = pd.DataFrame(proba, columns=le.classes_)
        if obs_ids is not None:
            proba_df.insert(0, 'Observation_ID', obs_ids[i : i + BATCH_SIZE])

        # Apply standard thresholding fallback filters
        prob_cols = [c for c in proba_df.columns if c != 'Observation_ID']
        proba_df[prob_cols] = proba_df[prob_cols].where(proba_df[prob_cols] >= THRESHOLD, 0.0)

        # Incrementally stream batch output out to gzip disk target
        mode = 'w' if first_batch else 'a'
        proba_df.to_csv(
            output_filename, 
            mode=mode, 
            index=False, 
            header=first_batch, 
            compression='gzip',
        )
        first_batch = False

        del batch_x, proba_df
        gc.collect()

    print(f"Inference Completed! Saved out to: {output_filename}")

    # validation metrics
    print("\nScoring predictions against 2022 ground truth...")

    acc   = accuracy_score(y_2022_encoded, y_pred_all)
    kappa = cohen_kappa_score(y_2022_encoded, y_pred_all)
    mcc   = matthews_corrcoef(y_2022_encoded, y_pred_all)

    evaluator      = ClassificationMetric(y_2022_encoded, y_pred_all)
    macro_prec     = evaluator.precision_score(average='macro')
    micro_prec     = evaluator.precision_score(average='micro')
    macro_recall   = evaluator.recall_score(average='macro')
    micro_recall   = evaluator.recall_score(average='micro')
    macro_f1       = evaluator.f1_score(average='macro')
    micro_f1       = evaluator.f1_score(average='micro')
    macro_f2       = evaluator.fbeta_score(beta=2, average='macro')
    micro_f2       = evaluator.fbeta_score(beta=2, average='micro')

    report = classification_report(
        y_2022_encoded, y_pred_all,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    per_drug_recall = pd.Series({
        drug: report[drug]['recall'] for drug in le.classes_ if drug in report
    })
    drugs_recall_ge_075 = int((per_drug_recall >= 0.75).sum())
    print(f"Drugs with 2022 recall >= 0.75: {drugs_recall_ge_075} / {len(per_drug_recall)}")

    validation_summary = pd.DataFrame([{
        'model':               MODEL_NAME,
        'dataset':             'MEPS_2022_internal_validation',
        'accuracy':            acc,
        'cohen_kappa':         kappa,
        'mcc':                 mcc,
        'macro_precision':     macro_prec,
        'micro_precision':     micro_prec,
        'macro_recall':        macro_recall,
        'micro_recall':        micro_recall,
        'macro_f1':            macro_f1,
        'micro_f1':            micro_f1,
        'macro_f2':            macro_f2,
        'micro_f2':            micro_f2,
        'drugs_recall_ge_075': drugs_recall_ge_075,
        'total_drugs':         len(per_drug_recall),
    }])
    validation_summary.to_csv(VALIDATION_SUMMARY_FILE, index=False)
    print(f"Saved validation summary to: {VALIDATION_SUMMARY_FILE}")


if __name__ == "__main__":
    # GLOBAL SYSTEM CONFIGURATIONS
    BATCH_SIZE = 100000
    THRESHOLD = 1 / 217
    MODEL_NAME = "TabICL_Super"
    
    # Model configuration mappings
    LABEL_ENCODER_PATH = "xgboost_super_label_encoder.joblib"
    TABICL_MODEL_PATH = "tabicl_trained_model.pkl"
    
    # Input files matching the MEPS 2022 validation block environment
    MY_2022_DATA = "super_data_2022.csv"
    OUTPUT_FILE = "tabicl_super_proba_2022.csv.gz"
    VALIDATION_SUMMARY_FILE = "tabicl_super_validation_summary.csv"

    # Start target run
    run_meps_2022_tabicl_pipeline(
        data_path=MY_2022_DATA,
        output_filename=OUTPUT_FILE,
        encoder_path=LABEL_ENCODER_PATH,
        model_path=TABICL_MODEL_PATH
    )