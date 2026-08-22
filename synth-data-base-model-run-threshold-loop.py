# -*- coding: utf-8 -*-
"""synth-data-inference-city.py
Runs inference for one city-level dataset at a time.
Change ACTIVE_CITY to switch between cities, then re-run.
Compatible with Python 3.12.7.
"""

import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from packaging import version
import gc
from numba import njit
from pynndescent import NNDescent
import scipy.sparse as sp
from sklearn.neighbors import sort_graph_by_row_values

OHE_SPARSE_KWARG = 'sparse_output' if version.parse(sklearn.__version__) >= version.parse('1.2') else 'sparse'

# =============================================================================
# 1. ACTIVE CITY — change this to switch between runs
# Options: "clark_county_nv" | "urbana_champaign_il" | "sandwich_ma"
# =============================================================================
#ACTIVE_CITY = "clark_county_nv"
#ACTIVE_CITY = "urbana_champaign_il"
ACTIVE_CITY = "sandwich_ma"

CITY_DATASETS = {
    "clark_county_nv":     "clark_county_nv_demo_rx.csv",
    "urbana_champaign_il": "urbana_champaign_il_demo_rx.csv",
    "sandwich_ma":         "sandwich_ma_demo_rx.csv",
}

MODELS_CONFIG = {
    "xgboost": {"model_file": "xgboost_super_final_model.ubj"},
    "knn":     {"model_file": "knn_super_clf_only.joblib"},
    "svm":     {"model_file": "svm_super_final_model.joblib"},
    "tabicl":  {"model_file": "tabicl_trained_model.pkl"},
    "realmlp": {"model_file": "realmlp_super_final_model.joblib"},
}

# =============================================================================
# 2. GLOBAL PARAMETERS
# =============================================================================
BATCH_SIZE        = 100000
THRESHOLD         = 1 / 217
PREPROCESSOR_PATH = "knn_svm_preprocessor.joblib"
encoder_path      = "xgboost_super_label_encoder.joblib"

# =============================================================================
# 3. CUSTOM CLASSES AND FUNCTIONS
# Must be defined before loading any joblib artifacts that reference them.
# =============================================================================
class SelectiveRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, drug_encoded_val, sentinel_value=-1.0):
        self.drug_encoded_val = drug_encoded_val
        self.sentinel_value   = sentinel_value
        self.scaler           = RobustScaler()

    def fit(self, X, y=None):
        if y is not None:
            mask = (y != self.drug_encoded_val)
            self.scaler.fit(X[mask] if mask.any() else X)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return np.nan_to_num(X_scaled, nan=self.sentinel_value)


@njit
def hassanat_distance(a, b):
    total = 0.0
    for i in range(a.shape[0]):
        x, y    = a[i], b[i]
        min_val = min(x, y)
        max_val = max(x, y)
        if min_val >= 0:
            total += 1.0 - (1.0 + min_val) / (1.0 + max_val)
        else:
            total += 1.0 - 1.0 / (1.0 + max_val + abs(min_val))
    return total

# Numba warmup
hassanat_distance(np.array([1.0, 2.0], dtype=np.float32),
                  np.array([2.0, 1.0], dtype=np.float32))

# =============================================================================
# 4. PREPROCESSING FUNCTIONS
# =============================================================================
TRAINING_FORMS = [
    '-1', 'DENTAL', 'INHALATION',
    'INTRA-ARTERIAL/INTRAMUSCULAR/INTRATHECAL/INTRAVENOUS',
    'INTRA-ARTERIAL/INTRATHECAL/INTRAVENOUS/INTRAMUSCULAR',
    'INTRAMUSCULAR',
    'INTRAMUSCULAR/INTRA-ARTICULAR/SOFT TISSUE/INTRALESIONAL',
    'INTRAMUSCULAR/INTRAVENOUS/INTRA-ARTERIAL',
    'INTRAVENOUS', 'INTRAVENOUS/INTRAMUSCULAR',
    'INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTERIAL',
    'INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTERIAL/INTRATHECAL',
    'INTRAVENOUS/INTRAMUSCULAR/INTRA-ARTICULAR/SOFT TISSUE/INTRALESIONAL',
    'INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS',
    'INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS/INTRA-ARTERIAL',
    'INTRAVENOUS/INTRAMUSCULAR/SUBCUTANEOUS/INTRATHECAL',
    'MUCOSA', 'NASAL', 'OPHTHALMIC', 'ORAL', 'ORAL MUCOSA',
    'OTIC', 'RECTAL', 'SUBCUTANEOUS', 'SUBLINGUAL',
    'TOPICAL', 'TRANSDERMAL', 'VAGINAL'
]


def _apply_form_categories(data):
    data['Form'] = data['Form'].astype(str)
    data['Form'] = pd.Categorical(data['Form'], categories=TRAINING_FORMS)
    return data


def preprocess_for_xgboost(data, categorical_cols, numeric_rx_cols):
    data.replace("NONE", np.nan, inplace=True)
    fill_values = {col: -1 for col in numeric_rx_cols}
    fill_values["Form"] = "-1"
    data.fillna(value=fill_values, inplace=True)
    for col in numeric_rx_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(-1).astype('float32')
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    data = _apply_form_categories(data)
    return xgb.DMatrix(data, enable_categorical=True)


def preprocess_for_realmlp(data, categorical_cols, numeric_rx_cols):
    data.replace("NONE", np.nan, inplace=True)
    fill_values = {col: -1 for col in numeric_rx_cols}
    fill_values["Form"] = "-1"
    data.fillna(value=fill_values, inplace=True)
    for col in numeric_rx_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(-1).astype('float32')
    for col in categorical_cols:
        data[col] = data[col].astype('category')
        data[col] = data[col].cat.reorder_categories(sorted(data[col].unique()), ordered=False)
    return _apply_form_categories(data)


def preprocess_for_tabicl(data, categorical_cols, numeric_rx_cols):
    data.replace("NONE", np.nan, inplace=True)
    fill_values = {col: -1 for col in numeric_rx_cols}
    fill_values["Form"] = "-1"
    data.fillna(value=fill_values, inplace=True)
    for col in numeric_rx_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(-1).astype('float32')
    for col in categorical_cols:
        data[col] = data[col].astype('category')
        data[col] = data[col].cat.reorder_categories(sorted(data[col].unique()), ordered=False)
    for col in data.select_dtypes(include=['float64']).columns:
        data[col] = data[col].astype('float32')
    for col in data.select_dtypes(include=['int64']).columns:
        data[col] = data[col].astype('int32')
    return _apply_form_categories(data)


def preprocess_for_knn_svm(data, numeric_rx_cols, categorical_cols, preprocessor):
    data.replace("NONE", np.nan, inplace=True)
    data['Form'] = data['Form'].fillna('-1').astype(str).str.strip()
    data['Form'] = pd.Categorical(data['Form'], categories=TRAINING_FORMS)
    data['Form'] = data['Form'].fillna('-1')
    for col in [c for c in categorical_cols if c != 'Form']:
        data[col] = data[col].fillna("Unknown").astype(str)
    for col in numeric_rx_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('float32')
    return preprocessor.transform(data)


# =============================================================================
# 5. LOAD AND PREDICT
# =============================================================================
def load_and_predict(file_path, raw_data):
    file_name    = os.path.basename(file_path)
    ext          = os.path.splitext(file_name)[1].lower()
    feature_cols = ['Age', 'Sex', 'Family_income', 'Insurance_coverage', 'Race_ethnicity',
                    'Quantity', 'Form', 'Strength', 'Day_Supply']
    cat_cols     = ['Sex', 'Insurance_coverage', 'Race_ethnicity', 'Form']
    num_rx_cols  = ['Quantity', 'Strength', 'Day_Supply']

    ids = raw_data['Observation_ID'].values if 'Observation_ID' in raw_data.columns else None
    data_for_processing = raw_data[feature_cols].copy()

    if file_name == "xgboost_super_final_model.ubj":
        processed_data = preprocess_for_xgboost(data_for_processing, cat_cols, num_rx_cols)
    elif file_name == "realmlp_super_final_model.joblib":
        processed_data = preprocess_for_realmlp(data_for_processing, cat_cols, num_rx_cols)
    elif file_name == "tabicl_trained_model.pkl":
        processed_data = preprocess_for_tabicl(data_for_processing, cat_cols, num_rx_cols)
    elif file_name in ["knn_super_clf_only.joblib", "svm_super_final_model.joblib"]:
        fitted_preprocessor = joblib.load(PREPROCESSOR_PATH)
        processed_data = preprocess_for_knn_svm(
            data_for_processing, num_rx_cols, cat_cols, fitted_preprocessor
        )
    else:
        processed_data = data_for_processing
        print(f"No custom preprocessing for {file_name}, using raw data.")

    try:
        if ext in ['.joblib', '.pkl']:
            if "knn" in file_path.lower():
                clf_step = joblib.load("knn_super_clf_only.joblib")
                indices, distances = GLOBAL_KNN_INDEX.query(processed_data, k=5)
                n_test  = processed_data.shape[0]
                n_train = GLOBAL_KNN_INDEX._raw_data.shape[0]
                rows    = np.repeat(np.arange(n_test), 5)
                dist_matrix = sp.csr_matrix(
                    (distances.ravel(), (rows, indices.ravel())),
                    shape=(n_test, n_train)
                )
                dist_matrix = sort_graph_by_row_values(dist_matrix, warn_when_not_sorted=False)
                proba = clf_step.predict_proba(dist_matrix)
            else:
                model = joblib.load(file_path)
                if hasattr(model, "feature_names_in_"):
                    model.feature_names_in_ = np.array(processed_data.columns.tolist())
                proba = model.predict_proba(processed_data)

        elif ext == '.ubj':
            model = xgb.Booster()
            model.load_model(file_path)
            proba = model.predict(processed_data)
        else:
            print(f"Unsupported file type: {ext}")
            return None

        proba_df = pd.DataFrame(proba, columns=le.classes_)
        if ids is not None:
            proba_df.insert(0, 'Observation_ID', ids)
        return proba_df

    except Exception as e:
        print(f"Error during prediction for {file_name}: {e}")
        return None


# =============================================================================
# 6. RUN INFERENCE FOR THE ACTIVE CITY
# =============================================================================
def run_inference(model_key, data_path, threshold_map_path=None):
    """Run batched inference for one model on the active city dataset."""
    model_file      = MODELS_CONFIG[model_key]['model_file']
    city_prefix     = os.path.basename(data_path).replace("_demo_rx.csv", "")
    output_filename = f"{city_prefix}_{model_key}_threshold_results.csv.gz"

    print(f"\n  --- {model_key.upper()} on {city_prefix} ---")

    data = pd.read_csv(
        data_path, sep=None, engine='python', encoding='utf-8-sig',
        dtype={'Form': str, 'Sex': str, 'Insurance_coverage': str, 'Race_ethnicity': str}
    )

    t_vector = None
    if model_key == "xgboost" and threshold_map_path:
        t_map    = joblib.load(threshold_map_path)
        t_vector = np.array([t_map.get(drug, THRESHOLD) for drug in le.classes_])

    first_batch = True
    for i in range(0, len(data), BATCH_SIZE):
        batch_df      = data.iloc[i: i + BATCH_SIZE].copy()
        batch_results = load_and_predict(file_path=model_file, raw_data=batch_df)

        if batch_results is not None:
            prob_cols = [c for c in batch_results.columns if c != 'Observation_ID']
            if model_key == "xgboost" and t_vector is not None:
                probs    = batch_results[prob_cols].values
                adjusted = probs / t_vector[np.newaxis, :]
                max_vals = adjusted.max(axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1.0
                batch_results[prob_cols] = adjusted / max_vals

            mode = 'w' if first_batch else 'a'
            batch_results.to_csv(
                output_filename, mode=mode, index=False,
                header=first_batch, compression='gzip'
            )
            first_batch = False

        del batch_df, batch_results
        gc.collect()

    print(f"  Saved: {output_filename}")


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    if ACTIVE_CITY not in CITY_DATASETS:
        raise ValueError(f"ACTIVE_CITY '{ACTIVE_CITY}' not recognized. "
                         f"Choose from: {list(CITY_DATASETS.keys())}")

    data_path = CITY_DATASETS[ACTIVE_CITY]

    print("Loading shared artifacts...")
    le = joblib.load(encoder_path)
    try:
        no_presc_idx = list(le.classes_).index('no prescriptions')
    except ValueError:
        no_presc_idx = 0

    print("Loading pre-built PyNNDescent index graph...")
    GLOBAL_KNN_INDEX = joblib.load("knn_final_prebuilt_index.joblib")
    print("All shared artifacts loaded.\n")

    print(f"\n{'='*65}")
    print(f"CITY-LEVEL INFERENCE: {ACTIVE_CITY}")
    print(f"{'='*65}")

    for model_key in MODELS_CONFIG:
        run_inference(
            model_key=model_key,
            data_path=data_path,
            threshold_map_path="xgboost_threshold_map.joblib"
        )

    print("\n" + "="*65)
    print(f"ALL MODELS COMPLETE FOR: {ACTIVE_CITY}")
    print("="*65)
    print("\nOutputs:")
    for model_key in MODELS_CONFIG:
        print(f"  {ACTIVE_CITY}_{model_key}_threshold_results.csv.gz")
