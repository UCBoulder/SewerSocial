# This script applies the exact preprocessing/imputation schema from the RealMLP notebook
# and runs inference using memory-safe batch streaming with a pre-saved model.
# Compatible with Python 3.12.7.

import joblib
import numpy as np
import pandas as pd
import os
import torch
import gc
from sklearn.utils import shuffle
from pytabkit import RealMLP_TD_Classifier

# Automatically select the best hardware acceleration available
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


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


def run_meps_2022_realmlp_pipeline(data_path, output_filename, encoder_path, model_path):
    print(f"\n--- Loading Components (Device: {DEVICE}) ---")
    le = joblib.load(encoder_path)
    
    # Load your pre-trained RealMLP-TD classifier checkpoint
    model = joblib.load(model_path)
    
    # Dynamically set runtime evaluation device context matching hardware
    if hasattr(model, 'device'):
        model.device = DEVICE

    feature_cols = ['Age', 'Sex', 'Family_income', 'Insurance_coverage', 'Race_ethnicity', 
                    'Quantity', 'Form', 'Strength', 'Day_Supply']
    categorical_cols = ['Sex', 'Insurance_coverage', 'Race_ethnicity', 'Form']

    # 1. Load 2022 super dataset matching original notebook configuration
    print(f"Reading dataset: {data_path}")
    data_2022 = pd.read_csv(data_path, sep=None, engine='python', encoding='utf-8-sig')
    
    if 'Unnamed: 0' in data_2022.columns:
        data_2022 = data_2022.drop(columns=['Unnamed: 0'])

    # 2. Structural sentinel filling for "no prescriptions" entries
    mask = data_2022['Drug'] == 'no prescriptions'
    data_2022.loc[mask, 'Form'] = data_2022.loc[mask, 'Form'].fillna('-1')

    # 3. Drop unseen classes not encountered during model training phase
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

    # fill sentinel value for "no prescriptions" numeric columns
    data_2022.fillna(-1, inplace=True)

    # 5. Ensure categorical columns are explicitly cast to 'category' dtypes
    # RealMLP uses pandas dtypes internally to detect and apply its own built-in encodings
    print("Enforcing internal RealMLP categorical schema dtypes...")
    for col in categorical_cols:
        data_2022[col] = data_2022[col].astype(str).astype('category')

    # Isolate targets and identifiers before transforming features
    y_2022_encoded = le.transform(data_2022['Drug'])
    obs_ids = data_2022['Observation_ID'].values if 'Observation_ID' in data_2022.columns else None

    X_2022 = data_2022[feature_cols].copy()

    print(f"Processed feature matrix configuration shape: {X_2022.shape}")
    print("Processing predictions in memory-safe batches...")

    first_batch = True

    # 7. Batched Model Querying & Output Streaming Loop
    for i in range(0, len(X_2022), BATCH_SIZE):
        batch_x = X_2022.iloc[i : i + BATCH_SIZE]
        
        # RealMLP built-in predict_proba framework handles scaling internally
        proba = model.predict_proba(batch_x)

        proba_df = pd.DataFrame(proba, columns=le.classes_)
        if obs_ids is not None:
            proba_df.insert(0, 'Observation_ID', obs_ids[i : i + BATCH_SIZE])

        # Apply standard thresholding fallback filters
        prob_cols = [c for c in proba_df.columns if c != 'Observation_ID']
        proba_df[prob_cols] = proba_df[prob_cols].where(proba_df[prob_cols] >= THRESHOLD, 0.0)

        # Incrementally stream batch output directly out to a gzip disk file
        mode = 'w' if first_batch else 'a'
        proba_df.to_csv(
            output_filename, 
            mode=mode, 
            index=False, 
            header=first_batch, 
            compression='gzip',
        )
        first_batch = False

        del batch_x, proba_df, proba
        gc.collect()

    print(f"Inference Completed! Saved out to: {output_filename}")


if __name__ == "__main__":
    # --- GLOBAL SYSTEM CONFIGURATIONS ---
    BATCH_SIZE = 100000
    THRESHOLD = 1 / 217
    
    # Model artifact tracking pathways
    LABEL_ENCODER_PATH = "xgboost_super_label_encoder.joblib"
    REALMLP_MODEL_PATH = "realmlp_super_final_model.joblib"  # Save checkpoint from Cell 11 training
    
    # Input datasets matching the MEPS 2022 environment block
    MY_2022_DATA = "super_data_2022.csv"
    OUTPUT_FILE = "realmlp_super_proba_2022.csv.gz"

    # Start deployment pipeline loop
    run_meps_2022_realmlp_pipeline(
        data_path=MY_2022_DATA,
        output_filename=OUTPUT_FILE,
        encoder_path=LABEL_ENCODER_PATH,
        model_path=REALMLP_MODEL_PATH
    )