# This script applies the exact preprocessing/imputation schema from the 2022 notebook cell
# and runs inference for XGBoost base model.
# Compatible with Python 3.12.7.

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

# ==========================================
# EXACT IMPUTATION FUNCTIONS FROM NOTEBOOK
# ==========================================
def impute_age(df):
    df = df.copy()
    df['income_bracket'] = (
        df.groupby('Year')['Family_income']
        .transform(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop') + 1)
    )
    hh_meds    = df.groupby(['Year', 'Household_ID'])['Age'].transform('median')
    grp_meds   = df.groupby(['Year', 'income_bracket', 'Insurance_coverage'])['Age'].transform('median')
    yr_meds    = df.groupby('Year')['Age'].transform('median')
    global_med = df['Age'].median()
    df['Age']  = df['Age'].fillna(hh_meds)
    df['Age']  = df['Age'].fillna(grp_meds)
    df['Age']  = df['Age'].fillna(yr_meds)
    df['Age']  = df['Age'].fillna(global_med)
    df.drop(columns=['income_bracket'], inplace=True)
    return df

def impute_strength(df):
    df = df.copy()
    mask = df['Drug'] != 'no prescriptions'
    yr_drug_meds = df.groupby(['Year', 'Drug'])['Strength'].transform('median')
    drug_meds = df.groupby('Drug')['Strength'].transform('median')
    global_med   = df.loc[mask, 'Strength'].median()

    df.loc[mask, 'Strength'] = df.loc[mask, 'Strength'].fillna(yr_drug_meds)
    df.loc[mask, 'Strength'] = df.loc[mask, 'Strength'].fillna(drug_meds)
    df.loc[mask, 'Strength'] = df.loc[mask, 'Strength'].fillna(global_med)
    return df

def impute_day_supply(df):
    df = df.copy()
    mask = df['Drug'] != 'no prescriptions'
    yr_drug_meds = df.groupby(['Year', 'Drug'])['Day_Supply'].transform('median')
    drug_meds = df.groupby('Drug')['Day_Supply'].transform('median')
    global_med   = df.loc[mask, 'Day_Supply'].median()

    df.loc[mask, 'Day_Supply'] = df.loc[mask, 'Day_Supply'].fillna(yr_drug_meds)
    df.loc[mask, 'Day_Supply'] = df.loc[mask, 'Day_Supply'].fillna(drug_meds)
    df.loc[mask, 'Day_Supply'] = df.loc[mask, 'Day_Supply'].fillna(global_med)
    return df

def run_meps_2022_xgboost_pipeline(data_path, output_filename, encoder_path, model_path, threshold_map_path):
    print(f"\n--- Loading Components ---")
    le = joblib.load(encoder_path)
    model = xgb.Booster()
    model.load_model(model_path)

    feature_cols = ['Age', 'Sex', 'Family_income', 'Insurance_coverage', 'Race_ethnicity', 
                    'Quantity', 'Form', 'Strength', 'Day_Supply']

    # 1. Load 2022 super dataset matching original notebook block
    print(f"Reading dataset: {data_path}")
    data_2022 = pd.read_csv(data_path, sep=None, engine='python', encoding='utf-8-sig')
    
    if 'Unnamed: 0' in data_2022.columns:
        data_2022 = data_2022.drop(columns=['Unnamed: 0'])

    # 2. Same sentinel filling as notebook training configuration
    numeric_rx_cols = ['Quantity', 'Strength', 'Day_Supply']
    mask = data_2022['Drug'] == 'no prescriptions'
    data_2022.loc[mask, 'Form'] = data_2022.loc[mask, 'Form'].fillna('-1')
    data_2022.loc[mask, numeric_rx_cols] = data_2022.loc[mask, numeric_rx_cols].fillna(-1)

    # 3. Drop any drugs in 2022 dataset not mapped during model training
    unseen = set(data_2022['Drug'].unique()) - set(le.classes_)
    if unseen:
        print(f"Unseen drugs in 2022 (will be dropped): {unseen}")
    data_2022 = data_2022[data_2022['Drug'].isin(le.classes_)].reset_index(drop=True)
    print(f"Rows after initial class filter: {len(data_2022):,}")

    # 4. Sequential hierarchical median imputation steps from notebook
    print("Applying hierarchical missing value imputations...")
    data_2022 = impute_age(data_2022)
    data_2022 = impute_strength(data_2022)
    data_2022 = impute_day_supply(data_2022)

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

    # 7. Batched Graph Querying & Optimized CSR Matrix Prediction Steps
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
    # --- GLOBAL SYSTEM CONFIGURATIONS ---
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