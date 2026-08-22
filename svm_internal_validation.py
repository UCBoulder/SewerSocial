# This script applies the exact preprocessing/imputation schema from the 2022 notebook cell
# and runs inference for SVM base model.
# Compatible with Python 3.12.7.

import pandas as pd
import numpy as np
import sklearn
import joblib
import gc
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report
)
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from permetrics import ClassificationMetric
import warnings
warnings.filterwarnings('ignore')

# Check for proper OneHotEncoder argument compatibility
from packaging import version
OHE_SPARSE_KWARG = 'sparse_output' if version.parse(sklearn.__version__) >= version.parse('1.2') else 'sparse'

# Define custom class to allow successful joblib deserialization of the preprocessor
class SelectiveRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, drug_encoded_val, sentinel_value=-1.0):
        self.drug_encoded_val = drug_encoded_val
        self.sentinel_value = sentinel_value
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        if y is not None:
            mask = (y != self.drug_encoded_val)
            if mask.any():
                self.scaler.fit(X[mask])
            else:
                self.scaler.fit(X)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=self.sentinel_value)
        return X_scaled


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


def run_meps_2022_svm_pipeline(data_path, output_filename, preprocessor_path, encoder_path, model_path):
    print(f"\n--- Loading Components ---")
    le = joblib.load(encoder_path)
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    feature_cols = ['Age', 'Sex', 'Family_income', 'Insurance_coverage', 'Race_ethnicity', 
                    'Quantity', 'Form', 'Strength', 'Day_Supply']

    # 1. Load 2022 super dataset matching original notebook block
    print(f"Reading dataset: {data_path}")
    data_2022 = pd.read_csv(data_path, sep=None, engine='python', encoding='utf-8-sig')
    
    if 'Unnamed: 0' in data_2022.columns:
        data_2022 = data_2022.drop(columns=['Unnamed: 0'])

    # 2. Same sentinel filling as notebook training configuration
    mask = data_2022['Drug'] == 'no prescriptions'
    data_2022.loc[mask, 'Form'] = data_2022.loc[mask, 'Form'].fillna('-1')

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

    # 5. Apply preprocessor pipeline using training metrics (transform only, no refitting)
    X_2022_processed = preprocessor.transform(X_2022)

    print(f"Preprocessed features array shape: {X_2022_processed.shape}")
    print("Processing predictions in memory-safe batches...")

    first_batch = True

    # 7. Batched Graph Querying & Optimized CSR Matrix Prediction Steps
    for i in range(0, len(X_2022_processed), BATCH_SIZE):
        batch_x = X_2022_processed[i : i + BATCH_SIZE]
        
        proba = model.predict_proba(batch_x)
        
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


if __name__ == "__main__":
    # --- GLOBAL SYSTEM CONFIGURATIONS ---
    BATCH_SIZE = 100000
    THRESHOLD = 1 / 217
    
    # Model configuration mappings
    PREPROCESSOR_PATH = "knn_svm_preprocessor.joblib"
    LABEL_ENCODER_PATH = "xgboost_super_label_encoder.joblib"
    SVM_MODEL_PATH = "svm_super_final_model.joblib"
    
    # Input files matching the MEPS 2022 validation block environment
    MY_2022_DATA = "super_data_2022.csv"
    OUTPUT_FILE = "svm_super_proba_2022.csv.gz"

    # Start target run
    run_meps_2022_svm_pipeline(
        data_path=MY_2022_DATA,
        output_filename=OUTPUT_FILE,
        preprocessor_path=PREPROCESSOR_PATH,
        encoder_path=LABEL_ENCODER_PATH,
        model_path=SVM_MODEL_PATH
    )