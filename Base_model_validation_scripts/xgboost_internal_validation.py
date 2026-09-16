# This script runs the saved XGBoost base model on the 2022 validation dataset.

# Load all required libraries.
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import gc
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report
)
from permetrics import ClassificationMetric
import warnings
warnings.filterwarnings('ignore')

# Confirm versions for reproducibility
import sklearn
print('XGBoost version:     ', xgb.__version__)
print('pandas version:      ', pd.__version__)
print('numpy version:       ', np.__version__)
print('scikit-learn version:', sklearn.__version__)

# Hard stop if XGBoost version is below 2.1.0.
# Native categorical support and learned missing directions
# require XGBoost 2.1+ — earlier versions will silently produce wrong results.
from packaging import version
assert version.parse(xgb.__version__) >= version.parse('2.1.0'), \
    f"ERROR: XGBoost 2.1+ required, found {xgb.__version__} — run: pip install 'xgboost>=2.1.0'"

print('\nAll version checks passed.')

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

def run_meps_2022_xgboost_pipeline(data_path, output_filename, encoder_path, model_path, threshold_map_path):
    print(f"\n--- Loading Components ---")
    le = joblib.load(encoder_path)
    model = xgb.Booster()
    model.load_model(model_path)

    feature_cols = ['Age', 'Sex', 'Family_income', 'Insurance_coverage', 'Race_ethnicity', 
                    'Quantity', 'Form', 'Strength', 'Day_Supply']

    # Load 2022 super dataset 
    print(f"Reading dataset: {data_path}")
    data_2022 = pd.read_csv(data_path, sep=None, engine='python', encoding='utf-8-sig')
    
    if 'Unnamed: 0' in data_2022.columns:
        data_2022 = data_2022.drop(columns=['Unnamed: 0'])

    # Same sentinel filling as training
    numeric_rx_cols = ['Quantity', 'Strength', 'Day_Supply']
    mask = data_2022['Drug'] == 'no prescriptions'
    data_2022.loc[mask, 'Form'] = data_2022.loc[mask, 'Form'].fillna('-1')
    data_2022.loc[mask, numeric_rx_cols] = data_2022.loc[mask, numeric_rx_cols].fillna(-1)

    # Drop any drugs in 2022 dataset not mapped during model training
    unseen = set(data_2022['Drug'].unique()) - set(le.classes_)
    if unseen:
        print(f"Unseen drugs in 2022 (will be dropped): {unseen}")
    data_2022 = data_2022[data_2022['Drug'].isin(le.classes_)].reset_index(drop=True)
    print(f"Rows after initial class filter: {len(data_2022):,}")

    # Sequential hierarchical median imputation steps
    print("Applying hierarchical missing value imputations...")
    age_medians = joblib.load('xgboost_super_age_medians.joblib')
    strength_medians = joblib.load('xgboost_super_strength_medians.joblib')
    day_supply_medians = joblib.load('xgboost_super_day_supply_medians.joblib')

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
        f"xgboost_super_age_medians.joblib is missing required key(s): {missing_keys}. "
        "This usually means the file is stale — re-run fit_age_medians(super_df) "
        "in the training script and re-save xgboost_super_age_medians.joblib before "
        "running this inference script."
    )

    data_2022 = apply_age_medians(data_2022, age_medians)
    data_2022 = apply_strength_medians(data_2022, strength_medians)
    data_2022 = apply_day_supply_medians(data_2022, day_supply_medians)

    # Isolate targets and identifiers before transforming features
    obs_ids = data_2022['Observation_ID'].values if 'Observation_ID' in data_2022.columns else None

    X_2022 = data_2022[feature_cols].copy()
    # cast features
    categorical_cols = ['Form', 'Sex', 'Insurance_coverage', 'Race_ethnicity']
    for col in categorical_cols:
        X_2022[col] = X_2022[col].astype('category')

    # prep for dmatrix
    y_2022_encoded = le.transform(data_2022['Drug'])

    print("Processing predictions in memory-safe batches...")

    first_batch = True

    # Batched Graph Querying & Optimized CSR Matrix Prediction Steps
    for i in range(0, len(X_2022), BATCH_SIZE):
        batch_x = X_2022[i : i + BATCH_SIZE]
        batch_y = y_2022_encoded[i : i + BATCH_SIZE]
        
        batch_dmatrix = xgb.DMatrix(batch_x, label=batch_y, enable_categorical=True)

        proba = model.predict(batch_dmatrix)
        
        proba_df = pd.DataFrame(proba, columns=le.classes_)
        if obs_ids is not None:
            proba_df.insert(0, 'Observation_ID', obs_ids[i : i + BATCH_SIZE])

        # Apply standard thresholding fallback filters
        prob_cols = [c for c in proba_df.columns if c != 'Observation_ID']
        proba_df[prob_cols] = proba_df[prob_cols].where(proba_df[prob_cols] >= THRESHOLD, 0.0)

        # Apply threshold tuning
        t_map = joblib.load(threshold_map_path)
        t_vector = np.array([t_map.get(drug, THRESHOLD) for drug in le.classes_])

        probs = proba_df[prob_cols].values 
        adjusted_probs = probs / t_vector[np.newaxis, :]
        max_vals = adjusted_probs.max(axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0 
        final_probs = adjusted_probs / max_vals
        final_probs[final_probs < THRESHOLD] = 0.0
        proba_df[prob_cols] = final_probs

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

        del batch_x, proba_df, batch_dmatrix
        gc.collect()

    print(f"Inference Completed! Saved out to: {output_filename}")

if __name__ == "__main__":
    # GLOBAL SYSTEM CONFIGURATIONS
    BATCH_SIZE = 100000
    THRESHOLD = 1 / 217
    
    # Model configuration mappings
    LABEL_ENCODER_PATH = "xgboost_super_label_encoder.joblib"
    XGBOOST_MODEL_PATH = "xgboost_super_final_model.ubj"
    THRESHOLD_MAP_PATH = "xgboost_threshold_map.joblib"
    
    # Input files matching the MEPS 2022 validation block environment
    MY_2022_DATA = "super_data_2022.csv"
    OUTPUT_FILE = "xgboost_super_proba_2022.csv.gz"

    # Start target run
    run_meps_2022_xgboost_pipeline(
        data_path=MY_2022_DATA,
        output_filename=OUTPUT_FILE,
        encoder_path=LABEL_ENCODER_PATH,
        model_path=XGBOOST_MODEL_PATH,
        threshold_map_path=THRESHOLD_MAP_PATH
    )