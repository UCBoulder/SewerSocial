# This script runs the saved ANN base model on the 2022 validation dataset.

# import necessary libraries
import joblib
import numpy as np
import pandas as pd
import os
import sklearn
import gc
import scipy.sparse as sp
from numba import njit
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle
from sklearn.neighbors import sort_graph_by_row_values
from packaging import version

# Check for proper OneHotEncoder argument compatibility
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

# Define Numba Hassanat distance metric
@njit
def hassanat_distance(a, b):
    total = 0.0
    for i in range(a.shape[0]):
        x = a[i]
        y = b[i]
        min_val = min(x, y)
        max_val = max(x, y)
        if min_val >= 0:
            total += 1.0 - (1.0 + min_val) / (1.0 + max_val)
        else:
            shift = abs(min_val)
            total += 1.0 - 1.0 / (1.0 + max_val + shift)
    return total


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

def run_meps_2022_knn_pipeline(data_path, output_filename, preprocessor_path, encoder_path, index_graph_path, clf_path):
    le = joblib.load(encoder_path)
    preprocessor = joblib.load(preprocessor_path)
    clf_step = joblib.load(clf_path)
    
    global_knn_index = joblib.load(index_graph_path)

    feature_cols = ['Age', 'Sex', 'Family_income', 'Insurance_coverage', 'Race_ethnicity', 
                    'Quantity', 'Form', 'Strength', 'Day_Supply']

    # Load 2022 super dataset matching original notebook block
    data_2022 = pd.read_csv(data_path, sep=None, engine='python', encoding='utf-8-sig')
    
    if 'Unnamed: 0' in data_2022.columns:
        data_2022 = data_2022.drop(columns=['Unnamed: 0'])

    # Same sentinel filling as notebook training configuration
    mask = data_2022['Drug'] == 'no prescriptions'
    data_2022.loc[mask, 'Form'] = data_2022.loc[mask, 'Form'].fillna('-1')

    # Drop any drugs in 2022 dataset not mapped during model training
    unseen = set(data_2022['Drug'].unique()) - set(le.classes_)
    if unseen:
        print(f"Unseen drugs in 2022 (will be dropped): {unseen}")
    data_2022 = data_2022[data_2022['Drug'].isin(le.classes_)].reset_index(drop=True)
    print(f"Rows after initial class filter: {len(data_2022):,}")

    # Sequential hierarchical median imputation steps from notebook
    print("Applying hierarchical missing value imputations...")
    age_medians = joblib.load('knn_super_age_medians.joblib')
    strength_medians = joblib.load('knn_super_strength_medians.joblib')
    day_supply_medians = joblib.load('knn_super_day_supply_medians.joblib')

     # Guard against a stale knn_super_age_medians.joblib that predates the
    # nearest-year fallback logic. Without this check, a missing key would only
    # surface as a KeyError deep inside apply_age_medians, which is harder to
    # diagnose than a clear message here.
    required_age_keys = {
        'income_bin_edges', 'hh_medians', 'grp_medians',
        'yr_medians', 'global_median', 'available_years',
    }
    missing_keys = required_age_keys - set(age_medians.keys())
    assert not missing_keys, (
        f"knn_super_age_medians.joblib is missing required key(s): {missing_keys}. "
        "This usually means the file is stale — re-run fit_age_medians(super_df) "
        "in the training script and re-save knn_super_age_medians.joblib before "
        "running this inference script."
    )

    data_2022 = apply_age_medians(data_2022, age_medians)
    data_2022 = apply_strength_medians(data_2022, strength_medians)
    data_2022 = apply_day_supply_medians(data_2022, day_supply_medians)

    # Isolate targets and identifiers before transforming features
    y_2022_encoded = le.transform(data_2022['Drug'])
    obs_ids = data_2022['Observation_ID'].values if 'Observation_ID' in data_2022.columns else None

    X_2022 = data_2022[feature_cols].copy()

    # 5. Apply preprocessor pipeline using training metrics (transform only, no refitting)
    X_2022_processed = preprocessor.transform(X_2022)

    print(f"Preprocessed features array shape: {X_2022_processed.shape}")
    print("Processing predictions in memory-safe batches...")

    first_batch = True
    n_train = global_knn_index._raw_data.shape[0]

    # 7. Batched Graph Querying & Optimized CSR Matrix Prediction Steps
    for i in range(0, len(X_2022_processed), BATCH_SIZE):
        batch_x = X_2022_processed[i : i + BATCH_SIZE]
        
        # Query the PyNNDescent pre-built graph framework
        indices, distances = global_knn_index.query(batch_x, k=5)

        n_test_batch = batch_x.shape[0]
        rows = np.repeat(np.arange(n_test_batch), 5)
        
        # Structure precomputed sparse distance graph required by KNeighborsClassifier step
        dist_matrix = sp.csr_matrix(
            (distances.ravel(), (rows, indices.ravel())),
            shape=(n_test_batch, n_train)
        )

        # Re-sort to remove internal scikit-learn structure layout warnings
        dist_matrix = sort_graph_by_row_values(dist_matrix, warn_when_not_sorted=False)

        # Run multi-class class probability prediction maps
        proba = clf_step.predict_proba(dist_matrix)
        
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

        del batch_x, proba_df, dist_matrix
        gc.collect()

    print(f"Inference Completed! Saved out to: {output_filename}")


if __name__ == "__main__":
    # --- GLOBAL SYSTEM CONFIGURATIONS ---
    BATCH_SIZE = 100000
    THRESHOLD = 1 / 217
    
    # Model configuration mappings
    PREPROCESSOR_PATH = "knn_svm_preprocessor.joblib"
    LABEL_ENCODER_PATH = "xgboost_super_label_encoder.joblib"
    INDEX_GRAPH_PATH = "knn_final_prebuilt_index.joblib"
    KNN_CLF_PATH = "knn_super_clf_only.joblib"
    
    # Input files matching the MEPS 2022 validation block environment
    MY_2022_DATA = "super_data_2022.csv"
    OUTPUT_FILE = "knn_super_proba_2022.csv.gz"

    # Start target run
    run_meps_2022_knn_pipeline(
        data_path=MY_2022_DATA,
        output_filename=OUTPUT_FILE,
        preprocessor_path=PREPROCESSOR_PATH,
        encoder_path=LABEL_ENCODER_PATH,
        index_graph_path=INDEX_GRAPH_PATH,
        clf_path=KNN_CLF_PATH
    )