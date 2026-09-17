# Applies the ensemble model for loaded external base model output.

# import necessary libraries
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL PATHS AND CONSTANTS
# =============================================================================
DATA_DIR         = './'
OUTPUT_DIR       = './ensemble_output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GLOBAL_THRESHOLD     = 1 / 217
WW_VOLUME_PER_CAPITA = 310.404  # L/day per capita

# city level pop_ww_volume values
pop_ww_volume_clark = WW_VOLUME_PER_CAPITA*2398871
pop_ww_volume_urbana = WW_VOLUME_PER_CAPITA*238842
pop_ww_volume_sandwich = WW_VOLUME_PER_CAPITA*2890

# =============================================================================
# DATASET CONFIGURATION
# All 13 datasets defined here. Update name, model_files, pop_size as needed.
# =============================================================================
DATASET_CONFIGS = [
     # --- SEWERSHED-LEVEL SUBSETS ---
    {
        "name": "clark_county_nv",
        "demo_rx": "clark_county_nv_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "clark_county_nv_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 2398871,  
        "pop_ww_volume": pop_ww_volume_clark,
    },
    {
        "name": "urbana_champaign_il",
        "demo_rx": "urbana_champaign_il_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "urbana_champaign_il_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "urbana_champaign_il_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "urbana_champaign_il_knn_threshold_results.csv.gz",
            "SVM_Super":     "urbana_champaign_il_svm_threshold_results.csv.gz",
            "TabICL":        "urbana_champaign_il_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 238842,  
        "pop_ww_volume": pop_ww_volume_urbana,
    },
    {
        "name": "sandwich_ma",
        "demo_rx": "sandwich_ma_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "sandwich_ma_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "sandwich_ma_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "sandwich_ma_knn_threshold_results.csv.gz",
            "SVM_Super":     "sandwich_ma_svm_threshold_results.csv.gz",
            "TabICL":        "sandwich_ma_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 2890,  
        "pop_ww_volume": pop_ww_volume_sandwich,
    },

    # --- FACILITY-LEVEL SUBSETS ---
    {
        "name": "clark_county_nv_campus",
        "demo_rx": "clark_county_nv_demo_rx.csv",  
        "model_files": {
            "XGBoost_Super": "clark_county_nv_campus_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_campus_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_campus_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_campus_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_campus_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 3750, 
        "pop_ww_volume": 1164015,
    },
    {
        "name": "clark_county_nv_f1",
        "demo_rx": "clark_county_nv_demo_rx.csv",  
        "model_files": {
            "XGBoost_Super": "clark_county_nv_f1_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_f1_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_f1_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_f1_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_f1_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 872009,  
        "pop_ww_volume": 378541000,
    },
    {
        "name": "clark_county_nv_f2",
        "demo_rx": "clark_county_nv_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "clark_county_nv_f2_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_f2_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_f2_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_f2_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_f2_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 86330,  
        "pop_ww_volume": 18927050,
    },
    {
        "name": "clark_county_nv_f3",
        "demo_rx": "clark_county_nv_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "clark_county_nv_f3_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_f3_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_f3_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_f3_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_f3_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 757418, 
        "pop_ww_volume": 158987220,
    },
    {
        "name": "clark_county_nv_f4",
        "demo_rx": "clark_county_nv_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "clark_county_nv_f4_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_f4_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_f4_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_f4_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_f4_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 248509,  
        "pop_ww_volume": 83279020,
    },
    {
        "name": "clark_county_nv_f4a",
        "demo_rx": "clark_county_nv_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "clark_county_nv_f4a_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_f4a_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_f4a_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_f4a_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_f4a_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 133977, 
        "pop_ww_volume": 60566560,
    },
    {
        "name": "clark_county_nv_f4b",
        "demo_rx": "clark_county_nv_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "clark_county_nv_f4b_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_f4b_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_f4b_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_f4b_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_f4b_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 114532,  
        "pop_ww_volume": 22712460,
    },
    {
        "name": "clark_county_nv_f5",
        "demo_rx": "clark_county_nv_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "clark_county_nv_f5_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_f5_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_f5_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_f5_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_f5_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 255008,  
        "pop_ww_volume": 75708200,
    },
    {
        "name": "clark_county_nv_f6",
        "demo_rx": "clark_county_nv_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "clark_county_nv_f6_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "clark_county_nv_f6_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "clark_county_nv_f6_knn_threshold_results.csv.gz",
            "SVM_Super":     "clark_county_nv_f6_svm_threshold_results.csv.gz",
            "TabICL":        "clark_county_nv_f6_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 16399,  
        "pop_ww_volume": 3028328,
    },
    {
        "name": "urbana_champaign_il_f1",
        "demo_rx": "urbana_champaign_il_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "urbana_champaign_il_f1_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "urbana_champaign_il_f1_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "urbana_champaign_il_f1_knn_threshold_results.csv.gz",
            "SVM_Super":     "urbana_champaign_il_f1_svm_threshold_results.csv.gz",
            "TabICL":        "urbana_champaign_il_f1_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 109634,  
        "pop_ww_volume": 34030835.9,
    },
    {
        "name": "sandwich_ma_f1",
        "demo_rx": "sandwich_ma_demo_rx.csv",
        "model_files": {
            "XGBoost_Super": "sandwich_ma_f1_xgboost_threshold_results.csv.gz",
            "RealMLP_Super": "sandwich_ma_f1_realmlp_threshold_results.csv.gz",
            "KNN_Super":     "sandwich_ma_f1_knn_threshold_results.csv.gz",
            "SVM_Super":     "sandwich_ma_f1_svm_threshold_results.csv.gz",
            "TabICL":        "sandwich_ma_f1_tabicl_threshold_results.csv.gz",
        },
        "pop_size": 976, 
        "pop_ww_volume": 302832,
    },
]

# =============================================================================
# 3. LOAD SHARED ARTIFACTS ONCE
# =============================================================================
print("=" * 65)
print("LOADING SHARED ARTIFACTS")
print("=" * 65)

le               = joblib.load(os.path.join(DATA_DIR, 'xgboost_super_label_encoder.joblib'))
drug_classes     = list(le.classes_)
no_presc_col_idx = drug_classes.index('no prescriptions')
print(f"LabelEncoder loaded. Drug classes: {len(drug_classes)}")

specialist_map = joblib.load(os.path.join(DATA_DIR, 'specialist_map_ensemble.joblib'))
print(f"Specialist map loaded: {len(specialist_map)} expert assignments.")

comparison_per_drug = joblib.load(os.path.join(DATA_DIR, 'comparison_per_drug.joblib'))
print(f"Per-drug comparison table loaded: {len(comparison_per_drug)} drugs.")

pharmuse = pd.read_csv(os.path.join(DATA_DIR, "pharmuse.csv"))
pharmuse["Excretion_percentage"] = pd.to_numeric(pharmuse["Excretion_percentage"], errors="coerce") / 100
pharmuse["Molar_Mass"]           = pd.to_numeric(pharmuse["Molar_Mass"], errors="coerce").astype(float)
excretion_factors = dict(
    zip(
        zip(pharmuse["Pharmaceutical"].str.lower().str.strip(),
            pharmuse["Administration_Route"].str.lower().str.strip()),
        pharmuse["Excretion_percentage"]
    )
)
if ("no prescriptions", "none") not in excretion_factors:
    excretion_factors[("no prescriptions", "none")] = 0.0
molar_mass_lookup = pharmuse[["Pharmaceutical", "Molar_Mass"]].drop_duplicates()
print("PharmUse excretion factors and molar masses loaded.")

# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def load_proba_dict(model_files):
    """Load probability CSVs into a dict of DataFrames keyed by model name."""
    proba_dict = {}
    for model_name, fname in model_files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIPPING: {model_name} ({fname} not found)")
            continue
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')
        df['Observation_ID'] = df['Observation_ID'].astype(str)
        df = df.set_index('Observation_ID')[drug_classes]
        proba_dict[model_name] = df
        print(f"  LOADED: {model_name} — shape {df.shape}")
    return proba_dict

def build_wta_matrix(proba_dict, common_obs):
    """Assemble the WTA probability matrix routing each drug to its best model."""
    wta_proba_df = pd.DataFrame(0.0, index=common_obs, columns=drug_classes)
    for _, row in comparison_per_drug.iterrows():
        drug       = row['Drug']
        best_model = row['Best_Model']
        if best_model in proba_dict:
            wta_proba_df[drug] = proba_dict[best_model][drug]
    return wta_proba_df.to_numpy()

def calculate_row_wastewater_mass(row):
    """Compute wastewater mass contribution for a single registry row."""
    drug = str(row['Predicted_Drug']).lower().strip()
    form = str(row['Form']).lower().strip()
    if drug == 'no prescriptions' or pd.isna(drug):
        return 0.0
    strength = row['Strength']
    days     = row['Day_Supply']
    quantity = row['Quantity']
    if pd.isna(days) or pd.isna(quantity) or pd.isna(strength):
        return 0.0
    if days == 0 or quantity == 0:
        return 0.0
    daily_freq = quantity / days
    cf         = excretion_factors.get((drug, form), 1.0)
    return strength * daily_freq * cf

# =============================================================================
# 5. MAIN ENSEMBLE FUNCTION
# =============================================================================

def run_ensemble(config):
    """Run the full ensemble pipeline for one dataset config."""
    name     = config["name"]
    pop_size = config["pop_size"]
    pop_ww_volume = config["pop_ww_volume"]

    print(f"\n{'='*65}")
    print(f"PROCESSING: {name}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # STEP A: Load probability CSVs
    # ------------------------------------------------------------------
    print("Loading probability CSVs...")
    proba_dict = load_proba_dict(config["model_files"])

    if len(proba_dict) < 2:
        print(f"  ERROR: Need at least 2 models, found {len(proba_dict)}. Skipping.")
        return

    # ------------------------------------------------------------------
    # STEP B: Align on common Observation_IDs with numeric sort
    # ------------------------------------------------------------------
    obs_ids    = [set(df.index) for df in proba_dict.values()]
    common_obs = sorted(obs_ids[0].intersection(*obs_ids[1:]), key=lambda x: int(x))
    print(f"Common observations: {len(common_obs):,}")

    for model_name in proba_dict:
        proba_dict[model_name] = proba_dict[model_name].loc[common_obs]

    # ------------------------------------------------------------------
    # STEP C: Build WTA matrix and resolve predictions via post-argmax
    #         threshold gate — threshold applied after argmax, not before
    # ------------------------------------------------------------------
    wta_proba_matrix = build_wta_matrix(proba_dict, common_obs)

    row_maxima     = wta_proba_matrix.max(axis=1)
    y_pred_indices = wta_proba_matrix.argmax(axis=1)
    for i in range(len(y_pred_indices)):
        if row_maxima[i] <= GLOBAL_THRESHOLD:
            y_pred_indices[i] = no_presc_col_idx

    final_predicted_drugs  = [drug_classes[idx] for idx in y_pred_indices]
    predictions_tracker_df = pd.DataFrame({
        'Observation_ID': common_obs,
        'Predicted_Drug': final_predicted_drugs
    })

    # ------------------------------------------------------------------
    # STEP D: Merge predictions with cohort features
    # For facility subsets, demo_rx is the parent city file — the inner
    # merge naturally restricts to only the facility's Observation_IDs.
    # ------------------------------------------------------------------
    demo_rx_path       = os.path.join(DATA_DIR, config["demo_rx"])
    external_cohort_df = pd.read_csv(demo_rx_path)
    external_cohort_df['Observation_ID'] = external_cohort_df['Observation_ID'].astype(str)

    final_population_registry = pd.merge(
        external_cohort_df, predictions_tracker_df, on='Observation_ID', how='inner'
    )
    print(f"Merged registry rows: {len(final_population_registry):,}")

    if len(final_population_registry) == 0:
        print("  ERROR: Merge resulted in 0 rows. Check Observation_ID alignment.")
        return

    # ------------------------------------------------------------------
    # STEP E: Data integrity audit
    # ------------------------------------------------------------------
    print("\n--- DATA INTEGRITY AUDIT ---")
    print(f"predictions_tracker_df shape: {predictions_tracker_df.shape}")
    print(f"Top 5 predicted drugs:\n{predictions_tracker_df['Predicted_Drug'].value_counts().head(5)}")
    for col in ['Strength', 'Day_Supply', 'Quantity']:
        none_count = (external_cohort_df[col].astype(str).str.upper().str.strip() == 'NONE').sum()
        print(f"  '{col}' literal NONE count: {none_count}")
    real_drug_sample = final_population_registry[
        final_population_registry['Predicted_Drug'] != 'no prescriptions'
    ].head(3)
    if not real_drug_sample.empty:
        print("Sample real drug rows:")
        print(real_drug_sample[['Observation_ID', 'Predicted_Drug', 'Strength', 'Day_Supply', 'Quantity']].to_string())
    print("--- END AUDIT ---\n")

    # ------------------------------------------------------------------
    # STEP F: Wastewater mass calculation
    # ------------------------------------------------------------------
    for col in ['Strength', 'Day_Supply', 'Quantity']:
        final_population_registry[col] = (
            final_population_registry[col]
            .astype(str).str.strip()
            .replace(['NONE', 'None', 'none', ''], np.nan)
        )
        final_population_registry[col] = pd.to_numeric(
            final_population_registry[col], errors='coerce'
        )

    final_population_registry['Form'] = (
        final_population_registry['Form'].astype(str).str.lower().str.strip()
    )
    final_population_registry['Predicted_Drug'] = (
        final_population_registry['Predicted_Drug'].astype(str).str.lower().str.strip()
    )

    final_population_registry['Wastewater_Mass_mcg'] = (
        final_population_registry.apply(calculate_row_wastewater_mass, axis=1)
    )

    # ------------------------------------------------------------------
    # STEP G: Aggregate wastewater metrics
    # ------------------------------------------------------------------
    ww_volume_per_capita = pop_ww_volume / pop_size

    wastewater_summary = (
        final_population_registry
        .groupby("Predicted_Drug")["Wastewater_Mass_mcg"]
        .sum()
        .reset_index()
        .rename(columns={"Wastewater_Mass_mcg": "Aggregate_Mass_Load_mcg"})
    )

    wastewater_summary["Wastewater_Concentration_mcg_per_L"] = (
        wastewater_summary["Aggregate_Mass_Load_mcg"] / pop_ww_volume
    ) / 4

    wastewater_summary["Population_Normalized_Mass_Load_mcg"] = (
        wastewater_summary["Wastewater_Concentration_mcg_per_L"] * ww_volume_per_capita
    )

    # Merge Molar Mass data directly into the master summary
    wastewater_summary = pd.merge(
        wastewater_summary,
        molar_mass_lookup,
        left_on="Predicted_Drug",
        right_on="Pharmaceutical",
        how="left",
    )

    # Clean up redundant merge columns, fill missing masses with NaN (or 1.0 if safety fallback is needed)
    wastewater_summary = wastewater_summary.drop(columns=["Pharmaceutical"])

    # Calculate Molar Concentration (Any missing Molar Mass rows naturally become NaN)
    wastewater_summary["Molar_Wastewater_Concentration_mol_per_L"] = (
        wastewater_summary["Wastewater_Concentration_mcg_per_L"]
        / (1000000 * wastewater_summary["Molar_Mass"])
    )

    # ------------------------------------------------------------------
    # STEP H: Export outputs
    # ------------------------------------------------------------------
    indiv_path = os.path.join(OUTPUT_DIR, f"{name}_indiv_predictions_and_mass.csv")
    final_population_registry.to_csv(indiv_path, index=False)
    print(f"Individual registry saved: {indiv_path}")

    summary_path = os.path.join(OUTPUT_DIR, f"{name}_aggregate_wastewater_summary.csv")
    wastewater_summary.to_csv(summary_path, index=False)
    print(f"Aggregate summary saved:   {summary_path}")

    print(f"\nTop 10 drugs by mass load for {name}:")
    print(wastewater_summary.head(10).to_string(index=False))
    print(f"\n✅ COMPLETE: {name}")


# =============================================================================
# 6. RUN ALL 13 DATASETS
# =============================================================================
if __name__ == "__main__":
    for config in DATASET_CONFIGS:
        run_ensemble(config)

    print("\n" + "=" * 65)
    print("ALL 13 DATASETS PROCESSED SUCCESSFULLY")
    print("=" * 65)