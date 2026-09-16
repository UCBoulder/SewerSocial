# Construct the ensemble mapping from historical recall and apply it to the 2022 validation dataset.

# Install required libraries.
import os
import ast
import warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report
)
from permetrics import ClassificationMetric
import sklearn

warnings.filterwarnings('ignore')
print('pandas version:      ', pd.__version__)
print('numpy version:       ', np.__version__)
print('scikit-learn version:', sklearn.__version__)

# All files are in the same folder as this notebook.
DATA_DIR = './'
print(f'\nData directory: {DATA_DIR}')

# Define which model probability files to load.
MODEL_FILES = [
    ('XGBoost_Super', 'xgboost_super_proba_2022.csv/xgboost_super_proba_2022.csv'),
    ('RealMLP_Super', 'realmlp_super_proba_2022.csv/realmlp_super_proba_2022.csv'),
    ('KNN_Super',     'knn_super_proba_2022.csv/knn_super_proba_2022.csv'),
    ('SVM_Super',     'svm_super_proba_2022.csv/svm_super_proba_2022.csv'),
    ('TabICL',        'tabicl_super_proba_2022.csv/tabicl_super_proba_2022.csv'),
]

# Check which files are present before loading.
print('Checking for model probability files...')
available = []
missing   = []
for model_name, fname in MODEL_FILES:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        available.append((model_name, fname))
        print(f'  FOUND:   {fname}')
    else:
        missing.append((model_name, fname))
        print(f'  MISSING: {fname}')

print(f'\nAvailable: {len(available)} / {len(MODEL_FILES)} models')
assert len(available) >= 2, f'ERROR: Need at least 2 model proba files, found {len(available)}.'

# Load the reference LabelEncoder.
le = joblib.load(f'{DATA_DIR}xgboost_super_label_encoder.joblib')
drug_classes = list(le.classes_)

print('Reference LabelEncoder loaded.')
print(f'Number of drug classes: {len(drug_classes)}')
assert len(drug_classes) == 217, f'ERROR: Expected 217 drug classes, found {len(drug_classes)}.'

# Global Threshold Constant
THRESHOLD = 1 / 217

# Load threshold mapping for XGBoost tuning
THRESHOLD_MAP_PATH = "xgboost_threshold_map.joblib"
if os.path.exists(THRESHOLD_MAP_PATH):
    t_map = joblib.load(THRESHOLD_MAP_PATH)
    t_vector = np.array([t_map.get(drug, THRESHOLD) for drug in drug_classes])
    print("XGBoost specialized threshold vector generated successfully.")
else:
    raise FileNotFoundError(f"Required threshold map not found at {THRESHOLD_MAP_PATH}")

# Load all available model probability files and align them
proba_dict = {}  

for model_name, fname in available:
    path = os.path.join(DATA_DIR, fname)
    df   = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')

    df['Observation_ID'] = df['Observation_ID'].astype(str)
    assert 'Observation_ID' in df.columns, f'ERROR: {fname} missing Observation_ID column.'

    # Set index and align columns to match reference encoder order
    df = df.set_index('Observation_ID')[drug_classes]

    proba_dict[model_name] = df
    print(f'Loaded & filtered {model_name}: shape {df.shape}')

# Find common observation IDs across all datasets
obs_ids = [set(df.index) for df in proba_dict.values()]
common_obs = obs_ids[0].intersection(*obs_ids[1:])
common_obs_sorted = sorted(common_obs, key=lambda x: int(x))
print(f'\nCommon Observation_IDs across all models: {len(common_obs_sorted):,}')

# Align all probability metrics to common observations
for model_name in proba_dict:
    proba_dict[model_name] = proba_dict[model_name].loc[common_obs_sorted]

# APPLY XGBOOST SPECIALIZED THRESHOLD TUNING
if 'XGBoost_Super' in proba_dict:
    print("\nApplying specialized threshold tuning to XGBoost probabilities...")
    xgb_probs = proba_dict['XGBoost_Super'].values
    adjusted_probs = xgb_probs / t_vector[np.newaxis, :]
    max_vals = adjusted_probs.max(axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    final_probs = adjusted_probs / max_vals

    proba_dict['XGBoost_Super'] = pd.DataFrame(
        final_probs, index=common_obs_sorted, columns=drug_classes
    )
    print("XGBoost probabilities tuned and updated.")

# CREATE EXPERT MAPPINGS VIA HISTORICAL RECALLS
BASE_DRUG_RECALL_FILES = [
    ('XGBoost_Super', 'xgboost_super_per_drug_recall.csv'),
    ('RealMLP_Super', 'realmlp_super_per_drug_recall.csv'),
    ('KNN_Super',     'knn_super_per_drug_recall.csv'),
    ('SVM_Super',     'svm_super_per_drug_recall.csv'),
    ('TabICL',        'tabicl_super_per_drug_recall.csv'),
]

comparison_per_drug = pd.DataFrame({'Drug': drug_classes})
for model_name, fname in BASE_DRUG_RECALL_FILES:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, sep=None, engine="python", encoding='utf-8-sig')
        recall_col = [c for c in df.columns if 'Recall' in c][0]
        df_sub = df[['Drug', recall_col]].rename(columns={recall_col: f'Recall_{model_name}'})
        comparison_per_drug = comparison_per_drug.merge(df_sub, on='Drug', how='left')
    else:
        print(f'WARNING: Historical file {fname} not found.')

recall_cols = [c for c in comparison_per_drug.columns if c.startswith('Recall_')]
comparison_per_drug[recall_cols] = comparison_per_drug[recall_cols].fillna(0)

# Raw idxmax/max are kept as diagnostic columns only
comparison_per_drug['Raw_Best_Model']  = comparison_per_drug[recall_cols].idxmax(axis=1).str.replace('Recall_', '')
comparison_per_drug['Best_Recall']     = comparison_per_drug[recall_cols].max(axis=1)

# Build active specialist map filtering based on recall.
RECALL_THRESHOLD = 0.75
specialist_map = {}
for _, row in comparison_per_drug.iterrows():
    if row['Best_Recall'] >= RECALL_THRESHOLD:
        specialist_map[row['Drug']] = row['Raw_Best_Model']

print(f"Ensemble configuration complete: {len(specialist_map)} specialist drugs mapped.")

# APPLY THE SPECIALIST MAP FILTER BEFORE BUILDING PREDICTIONS/METRICS
comparison_per_drug['Best_Model'] = comparison_per_drug['Drug'].map(specialist_map).fillna('None/Excluded')

# CONSTRUCT ENSEMBLE PROBABILITY MATRIX
wta_proba_df = pd.DataFrame(0.0, index=common_obs_sorted, columns=drug_classes)

for _, row in comparison_per_drug.iterrows():
    drug = row['Drug']
    best_model = row['Best_Model']
    if best_model in proba_dict:
        wta_proba_df[drug] = proba_dict[best_model][drug]

# Convert probability matrix to numpy for calculation steps
wta_proba_matrix = wta_proba_df.to_numpy()
no_presc_col_idx = drug_classes.index('no prescriptions')

# Hard predictions — Pick highest probability column (Argmax), with threshold
row_maxima = wta_proba_matrix.max(axis=1)
y_pred_ensemble = wta_proba_matrix.argmax(axis=1)
for i in range(len(y_pred_ensemble)):
    if row_maxima[i] <= THRESHOLD:
        y_pred_ensemble[i] = no_presc_col_idx

# Save pristine non-zero probability matrix for reference records
proba_export_df = wta_proba_df.copy()
proba_export_df.insert(0, 'Observation_ID', common_obs_sorted)
proba_export_df.to_csv('ensemble_proba_2022.csv', index=False)
print("Saved probability distributions to 'ensemble_proba_2022.csv'")

# EVALUATION AND PERFORMANCE METRICS
data_2022 = pd.read_csv(f'{DATA_DIR}super_data_2022.csv')
if 'Unnamed: 0' in data_2022.columns:
    data_2022 = data_2022.drop(columns=['Unnamed: 0'])

print('\n2022 ground truth data shape:', data_2022.shape)

# Align ground truth labels with identical common observation IDs
data_2022['Observation_ID'] = data_2022['Observation_ID'].astype(int).astype(str)
data_2022 = data_2022[data_2022['Observation_ID'].isin(common_obs_sorted)].copy()
data_2022 = data_2022.set_index('Observation_ID').loc[common_obs_sorted].reset_index()

# Drop any drugs not seen during standard training processes
data_2022 = data_2022[data_2022['Drug'].isin(le.classes_)].reset_index(drop=True)
y_2022_encoded = le.transform(data_2022['Drug'])

# Re-align final ensemble array indices
obs_id_to_idx   = {obs_id: i for i, obs_id in enumerate(common_obs_sorted)}
filtered_idx    = [obs_id_to_idx[oid] for oid in data_2022['Observation_ID']]
y_pred_filtered = y_pred_ensemble[filtered_idx]

# Calculate classification performance values
acc_ens   = accuracy_score(y_2022_encoded, y_pred_filtered)
kappa_ens = cohen_kappa_score(y_2022_encoded, y_pred_filtered)
mcc_ens   = matthews_corrcoef(y_2022_encoded, y_pred_filtered)

evaluator_ens     = ClassificationMetric(y_2022_encoded, y_pred_filtered)
macro_prec_ens    = evaluator_ens.precision_score(average='macro')
micro_prec_ens    = evaluator_ens.precision_score(average='micro')
macro_recall_ens  = evaluator_ens.recall_score(average='macro')
micro_recall_ens  = evaluator_ens.recall_score(average='micro')
macro_f1_ens      = evaluator_ens.f1_score(average='macro')
micro_f1_ens      = evaluator_ens.f1_score(average='micro')
macro_f2_ens      = evaluator_ens.fbeta_score(beta=2, average='macro')
micro_f2_ens      = evaluator_ens.fbeta_score(beta=2, average='micro')

# Parse granular performance details on individual drug items
report_ens = classification_report(
    y_2022_encoded, y_pred_filtered,
    labels=np.arange(len(le.classes_)),
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)

drug_recall_ens = pd.DataFrame([
    {
        'Drug':         drug,
        'Recall_2022':  report_ens[drug]['recall'],
        'Precision_2022': report_ens[drug]['precision'],
        'F1_2022':        report_ens[drug]['f1-score'],
        'Support_2022':   report_ens[drug]['support']
    }
    for drug in le.classes_ if drug in report_ens
])

# Extract drugs with 2022 recall >= 0.75 for ensemble
ens_drugs_ge_075 = drug_recall_ens[drug_recall_ens['Recall_2022'] >= 0.75]['Drug'].tolist()

# Save ensemble validation metrics table summary
ensemble_summary = pd.DataFrame([{
    'model':               'Ensemble_WinnerTakesAll',
    'models_included':     str(list(proba_dict.keys())),
    'dataset':             'MEPS_2022_internal_validation',
    'accuracy':            acc_ens,
    'cohen_kappa':         kappa_ens,
    'mcc':                 mcc_ens,
    'macro_precision':     macro_prec_ens,
    'micro_precision':     micro_prec_ens,
    'macro_recall':        macro_recall_ens,
    'micro_recall':        micro_recall_ens,
    'macro_f1':            macro_f1_ens,
    'micro_f1':            micro_f1_ens,
    'macro_f2':            macro_f2_ens,
    'micro_f2':            micro_f2_ens,
    'drugs_recall_ge_075': str(ens_drugs_ge_075)
}])
ensemble_summary.to_csv('ensemble_validation_summary.csv', index=False)

drug_recall_ens['Recall_Category'] = pd.cut(
    drug_recall_ens['Recall_2022'],
    bins=[-0.001, 0.75, 0.90, 1.001],
    labels=['Poor', 'Acceptable', 'Best'],
    right=False
)
drug_recall_ens.sort_values('Recall_2022', ascending=False).to_csv('ensemble_per_drug_recall.csv', index=False)

# Load master validation summaries (all corresponding to 2022 data)
BASE_MODEL_SUMMARIES = [
    'xgboost_super_validation_summary.csv',
    'realmlp_super_validation_summary.csv',
    'knn_super_validation_summary.csv',
    'svm_super_validation_summary.csv',
    'tabicl_super_validation_summary.csv',
]

all_summaries = []
for fname in BASE_MODEL_SUMMARIES:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        all_summaries.append(pd.read_csv(path))

all_summaries.append(ensemble_summary)

comparison_df = pd.concat(all_summaries, ignore_index=True)
comparison_df.to_csv('ensemble_vs_base_models.csv', index=False)

# Start visualization
fig = plt.figure(figsize=(20, 13))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.5)

ax1 = fig.add_subplot(gs[0, 0:2])   
ax2 = fig.add_subplot(gs[0, 2:4])   
ax3 = fig.add_subplot(gs[1, 1:3]) 

# ---------------------------------------------------------------------
# PLOT 3a : Recall Distribution & The 75% Cutoff Threshold (2022 Data)
# ---------------------------------------------------------------------
sns.histplot(
    data=drug_recall_ens, x='Recall_2022', bins=20, kde=False,
    color='#2b5c8f', alpha=0.75, ax=ax1
)

ax1.axvline(x=0.75, color='#d9534f', linestyle='--', linewidth=2.5)
ax1.axvspan(0.75, 1.0, alpha=0.12, color='#1b9e77', hatch='//')

ax1.set_xlabel('Macro Recall', fontsize=18, labelpad=10)
ax1.set_ylabel('Number of APIs', fontsize=18, labelpad=10)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.yaxis.grid(True, linestyle=':', alpha=0.6, color='#b0bec5')
ax1.set_axisbelow(True)
sns.despine(ax=ax1)

existing_ticks = [t for t in ax1.get_xticks() if t < 0.7 or t > 0.8]
ax1.set_xticks(sorted(existing_ticks + [0.75]))
ax1.set_xlim(-0.02, 1.02)

# ---------------------------------------------------------------------
# PLOT 3c : Ensemble vs. Base Models (2022 Macro Metrics Comparison)
# ---------------------------------------------------------------------
metrics_to_plot = ['accuracy', 'macro_recall', 'macro_precision', 'mcc']
df_melted = comparison_df.melt(
    id_vars='model', value_vars=metrics_to_plot,
    var_name='Metric', value_name='Score'
)

df_melted['Metric'] = df_melted['Metric'].str.replace('_', ' ').str.title()
df_melted['Metric'] = df_melted['Metric'].replace({'Mcc': 'MCC'})
df_melted['model'] = df_melted['model'].str.replace('_Super', '').str.replace('_WinnerTakesAll', ' (Ensemble)')
model_order = df_melted['model'].unique().tolist()

sns.barplot(
    data=df_melted, x='Metric', y='Score', hue='model', hue_order=model_order,
    palette='crest', alpha=0.9, ax=ax2, legend=False
)

ax2.set_xlabel('', labelpad=55)
ax2.set_ylabel('Metric Value', fontsize=18, labelpad=10)
ax2.set_ylim(0, 1.12)
ax2.tick_params(axis='y', which='major', labelsize=18)
ax2.yaxis.grid(True, linestyle=':', alpha=0.6, color='#b0bec5')
ax2.set_axisbelow(True)
sns.despine(ax=ax2)

metric_positions = ax2.get_xticks()
metric_labels = [t.get_text() for t in ax2.get_xticklabels()]

ax2.tick_params(axis='x', which='major', bottom=False, labelbottom=False)

for pos, label in zip(metric_positions, metric_labels):
    ax2.text(
        pos, 1.06, label,
        ha='center', va='bottom', fontsize=18, fontweight='bold', color='#222222',
        transform=ax2.get_xaxis_transform()
    )

for container, model_name in zip(ax2.containers, model_order):
    for bar in container:
        x = bar.get_x() + bar.get_width() / 2.
        ax2.text(
            x, -0.03, model_name,
            ha='center', va='top', rotation=90, fontsize=15, color='#444444',
            transform=ax2.get_xaxis_transform()
        )

# ---------------------------------------------------------------------
# PLOT 3b: Number of Drugs Reaching Recall >= 0.75 (Descending, Uniform Color)
# ---------------------------------------------------------------------
def parse_recall_count(val):
    """Parses integer, float, list, or string representation of list/count."""
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float, np.number)):
        return int(val)
    if isinstance(val, (list, tuple, set)):
        return len(val)
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            try:
                parsed = ast.literal_eval(val_str)
                return len(parsed)
            except Exception:
                pass
        if ',' in val_str:
            return len([x for x in val_str.split(',') if x.strip()])
        try:
            return int(float(val_str))
        except ValueError:
            return 0
    return 0

threshold_counts = {}
for summary in all_summaries:
    if 'model' in summary.columns and 'drugs_recall_ge_075' in summary.columns:
        model_name = summary['model'].iloc[0]
        clean_name = str(model_name).replace('_Super', '').replace('_WinnerTakesAll', ' (Ensemble)')
        if 'Ensemble' in clean_name:
            clean_name = 'Ensemble'
        
        raw_val = summary['drugs_recall_ge_075'].iloc[0]
        threshold_counts[clean_name] = parse_recall_count(raw_val)

df_counts = pd.DataFrame(list(threshold_counts.items()), columns=['Model', 'Passing_Count'])

# Strictly sort in descending order
df_counts = df_counts.sort_values(by='Passing_Count', ascending=False).reset_index(drop=True)

sns.barplot(
    data=df_counts, x='Model', y='Passing_Count',
    order=df_counts['Model'], color='#2b5c8f',
    alpha=0.85, ax=ax3
)

ax3.axhline(y=217, color='#2b5c8f', linestyle='--', linewidth=2.5)

ax3.text(
    0.02, 217, 'Total Number of APIs = 217',
    transform=ax3.get_yaxis_transform(),
    ha='left', va='bottom', color='#2b5c8f', fontsize=13, weight='semibold'
)

ax3.set_xlabel('', labelpad=10)
ax3.set_ylabel('Number of APIs', fontsize=18, labelpad=10)
ax3.set_ylim(0, len(drug_classes) * 1.05)
ax3.tick_params(axis='both', which='major', labelsize=18)
ax3.tick_params(axis='x', labelrotation=15)
for label in ax3.get_xticklabels():
    label.set_ha('right')

sns.despine(ax=ax3)
ax3.yaxis.grid(True, linestyle=':', alpha=0.6, color='#b0bec5')
ax3.set_axisbelow(True)

plt.tight_layout()
plt.savefig('internal_val_combined_grid.png', dpi=300, bbox_inches='tight')
plt.savefig('internal_val_combined_grid.svg', format='svg', bbox_inches='tight')
plt.close()

print("Saved: 'internal_val_combined_grid.svg'")