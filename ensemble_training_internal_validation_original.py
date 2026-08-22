# -*- coding: utf-8 -*-
"""ensemble_pharmshed_final.ipynb
PharmShed Ensemble Model with WTA Mapping, Threshold Filters, and Evaluation Metrics

FIX APPLIED: The specialist_map (historical recall >= 0.75 gate) is now applied
BEFORE the WTA probability matrix is built and BEFORE any 2022 validation metrics
are computed. Previously, the unfiltered idxmax() 'Best_Model' was used to build
predictions/metrics, and the specialist_map filter was only applied afterward when
pickling comparison_per_drug for script 2. That meant script 1's reported 2022
metrics reflected an ensemble that could route ANY of the 217 drugs to their best
model, while script 2 (using the filtered pickle) could only ever predict drugs
that passed the historical recall threshold. Now both use the identical rule.
"""

# Install required libraries.
#!pip install scikit-learn permetrics joblib

# Import all required libraries.
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report
)
from permetrics import ClassificationMetric
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

import sklearn
print('pandas version:      ', pd.__version__)
print('numpy version:       ', np.__version__)
print('scikit-learn version:', sklearn.__version__)

# All files are in the same folder as this notebook.
DATA_DIR = './'
print(f'\nData directory: {DATA_DIR}')

# Define which model probability files to load.
MODEL_FILES = [
    ('XGBoost_Super', 'xgboost_super_proba_2022.csv'),
    ('RealMLP_Super', 'realmlp_super_proba_2022.csv'),
    ('KNN_Super',     'knn_super_proba_2022.csv'),
    ('SVM_Super',     'svm_super_proba_2022.csv'),
    ('TabICL',        'tabicl_super_proba_2022.csv'),
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
proba_dict = {}  # model_name -> DataFrame

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

# --- APPLY XGBOOST SPECIALIZED THRESHOLD TUNING ---
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

# --- CREATE EXPERT MAPPINGS VIA HISTORICAL RECALLS ---
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

# Raw idxmax/max are kept as diagnostic columns only (NOT used for routing anymore).
comparison_per_drug['Raw_Best_Model']  = comparison_per_drug[recall_cols].idxmax(axis=1).str.replace('Recall_', '')
comparison_per_drug['Best_Recall']     = comparison_per_drug[recall_cols].max(axis=1)

# Build active specialist map filtering based on reliability threshold.
# THIS is the single routing rule used everywhere (2022 validation AND external data).
RECALL_THRESHOLD = 0.75
specialist_map = {}
for _, row in comparison_per_drug.iterrows():
    if row['Best_Recall'] >= RECALL_THRESHOLD:
        specialist_map[row['Drug']] = row['Raw_Best_Model']

print(f"Ensemble configuration complete: {len(specialist_map)} specialist drugs mapped.")

# --- APPLY THE SPECIALIST MAP FILTER *BEFORE* BUILDING PREDICTIONS/METRICS ---
# This is the fix: Best_Model now reflects the same routing rule that will later
# be pickled and reused in script 2, instead of the unfiltered idxmax.
comparison_per_drug['Best_Model'] = comparison_per_drug['Drug'].map(specialist_map).fillna('None/Excluded')

# --- CONSTRUCT ENSEMBLE PROBABILITY MATRIX ---
# Drugs mapped to 'None/Excluded' will never match a key in proba_dict, so they
# correctly stay at an all-zero probability column here -- consistent with how
# script 2's build_wta_matrix() behaves on the external datasets.
wta_proba_df = pd.DataFrame(0.0, index=common_obs_sorted, columns=drug_classes)

for _, row in comparison_per_drug.iterrows():
    drug = row['Drug']
    best_model = row['Best_Model']
    if best_model in proba_dict:
        wta_proba_df[drug] = proba_dict[best_model][drug]

# Convert probability matrix to numpy for calculation steps
wta_proba_matrix = wta_proba_df.to_numpy()
no_presc_col_idx = drug_classes.index('no prescriptions')

# Hard predictions — Pick highest probability column (Argmax), with threshold to filter out predictions with confidence < 1/217
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

# --- EVALUATION AND PERFORMANCE METRICS ---
# Load ground truth labels for 2022 validation
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

# Re-align final ensemble array indices with the newly filtered validation rows
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

print('\n' + '='*50)
print('ENSEMBLE VALIDATION — WINNER-TAKES-ALL 2022 METRICS')
print('(restricted to specialist_map-eligible drugs, consistent with external data)')
print('='*50)
print(f'Accuracy:           {acc_ens:.4f}')
print(f'Cohen Kappa:        {kappa_ens:.4f}')
print(f'MCC:                {mcc_ens:.4f}')
print(f'Macro Precision:    {macro_prec_ens:.4f}')
print(f'Micro Precision:    {micro_prec_ens:.4f}')
print(f'Macro Recall:       {macro_recall_ens:.4f}')
print(f'Micro Recall:       {micro_recall_ens:.4f}')
print(f'Macro F2:           {macro_f2_ens:.4f}')
print(f'Micro F2:           {micro_f2_ens:.4f}')

# Save ensemble validation metrics table summary
ensemble_summary = pd.DataFrame([{
    'model':            'Ensemble_WinnerTakesAll',
    'models_included':  str(list(proba_dict.keys())),
    'dataset':          'MEPS_2022_internal_validation',
    'accuracy':         acc_ens,
    'cohen_kappa':      kappa_ens,
    'mcc':              mcc_ens,
    'macro_precision':  macro_prec_ens,
    'micro_precision':  micro_prec_ens,
    'macro_recall':     macro_recall_ens,
    'micro_recall':     micro_recall_ens,
    'macro_f1':         macro_f1_ens,
    'micro_f1':         micro_f1_ens,
    'macro_f2':         macro_f2_ens,
    'micro_f2':         micro_f2_ens,
}])
ensemble_summary.to_csv('ensemble_validation_summary.csv', index=False)

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
        'Drug':           drug,
        'Recall_2022':    report_ens[drug]['recall'],
        'Precision_2022': report_ens[drug]['precision'],
        'F1_2022':        report_ens[drug]['f1-score'],
        'Support_2022':   report_ens[drug]['support']
    }
    for drug in le.classes_ if drug in report_ens
])

drugs_recalled_ens = (drug_recall_ens['Recall_2022'] > 0).sum()
print(f'\nTotal drug elements successfully recalled (recall > 0): {drugs_recalled_ens} / {len(drug_recall_ens)}')

# Categorize metrics data for final reporting documentation
drug_recall_ens['Recall_Category'] = pd.cut(
    drug_recall_ens['Recall_2022'],
    bins=[-0.001, 0.75, 0.90, 1.001],
    labels=['Poor', 'Acceptable', 'Best'],
    right=False
)
drug_recall_ens.sort_values('Recall_2022', ascending=False).to_csv('ensemble_per_drug_recall.csv', index=False)

# --- UNFILTERED (RAW) WTA PASS — VISUALIZATION ONLY ---
# The official ensemble above only ever predicts specialist_map-eligible drugs,
# so any excluded drug's Recall_2022 is forced to 0 -- that's correct for the
# real model, but it hides what that drug's recall actually was before the
# specialist_map gate zeroed its column out. To recover that "actual" per-drug
# recall for plotting, we build a second WTA matrix that routes every drug to
# its Raw_Best_Model (no >= 0.75 gate) and score it against the same 2022
# ground truth. This pass is NEVER used for the final model, the reported
# validation metrics, the exported CSVs/summary, or the pickled routing rule --
# it exists solely to populate the Support vs. Recall scatterplot below.
wta_proba_df_raw = pd.DataFrame(0.0, index=common_obs_sorted, columns=drug_classes)
for _, row in comparison_per_drug.iterrows():
    drug = row['Drug']
    raw_best_model = row['Raw_Best_Model']
    if raw_best_model in proba_dict:
        wta_proba_df_raw[drug] = proba_dict[raw_best_model][drug]

wta_proba_matrix_raw = wta_proba_df_raw.to_numpy()
row_maxima_raw = wta_proba_matrix_raw.max(axis=1)
y_pred_ensemble_raw = wta_proba_matrix_raw.argmax(axis=1)
for i in range(len(y_pred_ensemble_raw)):
    if row_maxima_raw[i] <= THRESHOLD:
        y_pred_ensemble_raw[i] = no_presc_col_idx

y_pred_filtered_raw = y_pred_ensemble_raw[filtered_idx]

report_ens_raw = classification_report(
    y_2022_encoded, y_pred_filtered_raw,
    labels=np.arange(len(le.classes_)),
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)

drug_recall_raw = pd.DataFrame([
    {'Drug': drug, 'Recall_2022_Raw': report_ens_raw[drug]['recall']}
    for drug in le.classes_ if drug in report_ens_raw
])

# Merge the pre-filter recall in as an extra column for visualization only --
# drug_recall_ens['Recall_2022'] (the official, post-filter number) is untouched.
drug_recall_ens = drug_recall_ens.merge(drug_recall_raw, on='Drug', how='left')

# Compile comprehensive master summary metrics files
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

all_summaries.append(ensemble_summary[[
    'model', 'dataset', 'accuracy', 'cohen_kappa', 'mcc',
    'macro_precision', 'micro_precision', 'macro_recall',
    'micro_recall', 'macro_f2', 'micro_f2'
]])

comparison_df = pd.concat(all_summaries, ignore_index=True)
comparison_df.to_csv('ensemble_vs_base_models.csv', index=False)

# Complete master drug column framework mapping comparisons
# MERGE instead of overwriting so we keep the base model recall columns!
ensemble_recall_sub = drug_recall_ens[['Drug', 'Recall_2022']].rename(columns={'Recall_2022': 'Recall_Ensemble'})
comparison_per_drug = comparison_per_drug.merge(ensemble_recall_sub, on='Drug', how='left')

# NOTE: Best_Model was already set from specialist_map above, BEFORE predictions
# were generated, so no second overwrite is needed here anymore. Leaving a no-op
# reassignment removed to avoid ambiguity about when the filter was applied.
comparison_per_drug.to_csv('ensemble_per_drug_comparison.csv', index=False)

# Save the specialist lookup map dictionary
joblib.dump(specialist_map, os.path.join(DATA_DIR, 'specialist_map_ensemble.joblib'))

print('\n' + '='*55)
print('COMPUTATION COMPLETE — ALL 2022 METRICS LOGGED AND EXPORTED')
print('='*55)

print("--- DATA INTEGRITY DIAGNOSTIC ---")
# Check 1: Did the ground truth alignment collapse the dataset?
print(f"Original ground truth rows: {len(data_2022)}")
print(f"Filtered ground truth rows: {len(y_2022_encoded)}")

# Check 2: Are the predictions mostly uniform?
preds_df = pd.Series(y_pred_filtered).value_counts().head(5)
print("\nTop 5 predicted class indices:\n", preds_df)

# Check 3: Check for NaN corruption in probability dict
for model, df in proba_dict.items():
    print(f"{model} total non-zero values: {np.count_nonzero(df.values)}")

# Check 4 (NEW): Confirm the routing rule used for prediction matches specialist_map exactly
n_excluded = (comparison_per_drug['Best_Model'] == 'None/Excluded').sum()
print(f"\nDrugs excluded from WTA routing (consistent with script 2): {n_excluded} / {len(drug_classes)}")
print(f"Drugs eligible via specialist_map: {len(specialist_map)} / {len(drug_classes)}")
assert n_excluded == len(drug_classes) - len(specialist_map), \
    "Mismatch between excluded-drug count and specialist_map size — routing rule inconsistency!"

joblib.dump(comparison_per_drug, os.path.join(DATA_DIR, 'comparison_per_drug.joblib'))

fig = plt.figure(figsize=(20, 13))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.5)

ax1 = fig.add_subplot(gs[0, 0:2])   # top-left
ax2 = fig.add_subplot(gs[0, 2:4])   # top-right
ax3 = fig.add_subplot(gs[1, 1:3])   # bottom, centered (half width, columns 1-3 of 4)

# ---------------------------------------------------------------------
# PLOT 1 (top-left): Recall Distribution & The 75% Cutoff Threshold
# ---------------------------------------------------------------------
sns.histplot(
    data=drug_recall_ens, x='Recall_2022', bins=20, kde=False,
    color='#2b5c8f', alpha=0.75, ax=ax1
)

ax1.axvline(
    x=0.75, color='#d9534f', linestyle='--', linewidth=2.5
)

ax1.axvspan(
    0.75, 1.0, alpha=0.12, color='#1b9e77', hatch='//'
)

ax1.set_xlabel('Macro Recall', fontsize=18, labelpad=10)
ax1.set_ylabel('Number of APIs', fontsize=18, labelpad=10)
ax1.set_xlim(-0.02, 1.02)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.yaxis.grid(True, linestyle=':', alpha=0.6, color='#b0bec5')
ax1.set_axisbelow(True)
sns.despine(ax=ax1)

existing_ticks = [t for t in ax1.get_xticks() if t < 0.7 or t > 0.8]
ax1.set_xticks(sorted(existing_ticks + [0.75]))
ax1.set_xlim(-0.02, 1.02)

# ---------------------------------------------------------------------
# PLOT 2 (top-right): Ensemble vs. Base Models (Macro Metrics Comparison)
# ---------------------------------------------------------------------
metrics_to_plot = ['accuracy', 'macro_recall', 'macro_precision', 'mcc']
df_melted = comparison_df.melt(
    id_vars='model', value_vars=metrics_to_plot,
    var_name='Metric', value_name='Score'
)

df_melted['Metric'] = df_melted['Metric'].str.replace('_', ' ').str.title()
df_melted['Metric'] = df_melted['Metric'].replace({'Mcc': 'MCC'})
df_melted['model'] = df_melted['model'].str.replace('_Super', '').str.replace('_WinnerTakesAll', ' (Ensemble)')

# Explicit hue order so container index reliably maps to model name below
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

# Grab metric tick positions/labels BEFORE hiding them
metric_positions = ax2.get_xticks()
metric_labels = [t.get_text() for t in ax2.get_xticklabels()]

# Hide the default x tick labels (metric names) so they don't collide
# with the model names placed underneath the bars
ax2.tick_params(axis='x', which='major', bottom=False, labelbottom=False)

# Metric name above each group of bars
for pos, label in zip(metric_positions, metric_labels):
    ax2.text(
        pos, 1.06, label,
        ha='center', va='bottom', fontsize=18, fontweight='bold', color='#222222',
        transform=ax2.get_xaxis_transform()
    )

# Model configuration under each bar, matched by explicit hue_order
for container, model_name in zip(ax2.containers, model_order):
    for bar in container:
        x = bar.get_x() + bar.get_width() / 2.
        ax2.text(
            x, -0.03, model_name,
            ha='center', va='top', rotation=90, fontsize=15, color='#444444',
            transform=ax2.get_xaxis_transform()
        )

# ---------------------------------------------------------------------
# PLOT 3 (bottom, centered): Number of Drugs Reaching Acceptable Recall (>= 0.75)
# ---------------------------------------------------------------------
model_recall_cols = [c for c in comparison_per_drug.columns if 'Recall_' in c]

threshold_counts = {}
for col in model_recall_cols:
    clean_name = col.replace('Recall_', '').replace('_Super', '')
    if clean_name == 'Ensemble':
        clean_name = 'Ensemble'

    passing_count = (comparison_per_drug[col] >= 0.75).sum()
    threshold_counts[clean_name] = passing_count

df_counts = pd.DataFrame(list(threshold_counts.items()), columns=['Model', 'Passing_Count'])
df_counts = df_counts.sort_values(by='Passing_Count', ascending=False).reset_index(drop=True)

colors = ['#1b9e77' if model == 'Ensemble' else '#2b5c8f' for model in df_counts['Model']]

sns.barplot(
    data=df_counts, x='Model', y='Passing_Count',
    palette=colors, alpha=0.85, ax=ax3
)

ax3.axhline(
    y=217, color='#2b5c8f', linestyle='--', linewidth=2.5
)

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

""" # ---------------------------------------------------------------------
# Support Tier Distribution vs. Recall Threshold Pass/Fail
# ---------------------------------------------------------------------
support_bins = [1, 10, 100, 1000, np.inf]
support_labels = ['Very Low (1-9)', 'Low (10-99)', 'Medium (100-999)', 'High (1000+)']

drug_recall_ens['Support_Tier'] = pd.cut(
    drug_recall_ens['Support_2022'],
    bins=support_bins,
    labels=support_labels,
    right=False
) """

""" drug_recall_ens['Passed_Threshold'] = drug_recall_ens['Recall_2022'] >= 0.75

tier_breakdown = drug_recall_ens.groupby(['Support_Tier', 'Passed_Threshold'], observed=True).size().unstack(fill_value=0)
tier_breakdown.columns = ['Failed (<0.75)', 'Passed (>=0.75)']
tier_breakdown['Total'] = tier_breakdown.sum(axis=1)
tier_breakdown['Pass_Rate'] = (tier_breakdown['Passed (>=0.75)'] / tier_breakdown['Total'] * 100).round(1)

print(tier_breakdown) """


""" # ---------------------------------------------------------------------
# PLOT 3 (bottom-left): Support vs. Recall (Tracking Rare-Drug Vulnerabilities)
# ---------------------------------------------------------------------
from matplotlib.lines import Line2D

from matplotlib.lines import Line2D

# Bring in historical inclusion status AND historical recall value
drug_recall_ens = drug_recall_ens.merge(
    comparison_per_drug[['Drug', 'Best_Model', 'Best_Recall']], on='Drug', how='left'
)

# Recolor based on HISTORICAL recall, not 2022 recall
drug_recall_ens['Recall_Category'] = pd.cut(
    drug_recall_ens['Best_Recall'],
    bins=[-0.001, 0.75, 0.90, 1.001],
    labels=['Poor', 'Acceptable', 'Best'],
    right=False
)

def inclusion_status(row):
    hist_included = row['Best_Model'] != 'None/Excluded'      # passed historical 0.75 gate
    val_included  = row['Recall_2022_Raw'] >= 0.75             # would pass gate on 2022 data

    if hist_included == val_included:
        return 'Match'
    elif hist_included and not val_included:
        return 'Historical Only (Missed in 2022)'
    else:
        return 'Validation Only (Missed Historically)'

drug_recall_ens['Inclusion_Status'] = drug_recall_ens.apply(inclusion_status, axis=1)

inclusion_markers = {
    'Match': 'o',
    'Historical Only (Missed in 2022)': 'X',
    'Validation Only (Missed Historically)': '^',
}

sns.scatterplot(
    data=drug_recall_ens, x='Support_2022', y='Recall_2022_Raw',
    hue='Recall_Category', style='Inclusion_Status',
    palette={'Poor': '#d9534f', 'Acceptable': '#f0ad4e', 'Best': '#5cb85c'},
    markers=inclusion_markers, style_order=list(inclusion_markers.keys()),
    size='Support_2022', sizes=(40, 350), alpha=0.65, ax=ax3, legend=False
)

ax3.set_xscale('log')

top10 = drug_recall_ens.nlargest(10, 'Support_2022')
bottom10 = drug_recall_ens.nsmallest(10, 'Support_2022')
labeled_subset = pd.concat([top10, bottom10])

try:
    from adjustText import adjust_text
    texts = []
    for _, row in labeled_subset.iterrows():
        texts.append(
            ax3.text(
                row['Support_2022'], row['Recall_2022_Raw'], row['Drug'],
                fontsize=13, color='#222222', zorder=5
            )
        )
    adjust_text(
        texts, ax=ax3,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6),
        expand_points=(1.2, 1.2)
    )
except ImportError:
    print("NOTE: 'adjustText' not installed — labels may overlap. "
          "Install with: pip install adjustText")
    for _, row in labeled_subset.iterrows():
        ax3.annotate(
            row['Drug'],
            (row['Support_2022'], row['Recall_2022_Raw']),
            fontsize=13, color='#222222',
            xytext=(4, 4), textcoords='offset points'
        )

ax3.axhline(
    y=0.75, color='#d9534f', linestyle='--', linewidth=1.5
)

xmin, xmax = ax3.get_xlim()
ax3.text(
    xmax * 0.5, 0.775, 'Recall Cutoff = 0.75',
    color='#d9534f', fontsize=13, weight='semibold', va='bottom', ha='center'
)

ax3.set_xlabel('Support (Log Scale)', fontsize=18, labelpad=10)
ax3.set_ylabel('Macro Recall', fontsize=18, labelpad=10)
ax3.tick_params(axis='both', which='major', labelsize=18)
ax3.set_ylim(-0.03, 1.05)
ax3.grid(True, which="both", linestyle=':', alpha=0.5, color='#b0bec5')
ax3.set_axisbelow(True)
sns.despine(ax=ax3)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d9534f', markersize=10, label='Poor Recall'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#f0ad4e', markersize=10, label='Acceptable Recall'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#5cb85c', markersize=10, label='Best Recall'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#666666', markersize=10, label='Historical Only'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#666666', markersize=10, label='Validation Only'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markersize=6, label='Support=2000'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markersize=18, label='Support=10000'),
]

ax3.legend(
    handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.18),
    ncol=4, frameon=False, fontsize=11, handletextpad=0.4, columnspacing=1.4
) """