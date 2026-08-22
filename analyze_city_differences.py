# -*- coding: utf-8 -*-
"""
analyze_city_differences.py

Designed for large environmental datasets (millions of rows). 
Compares pharmaceutical presence and mass loads across 3 cities.
Calculates statistical significance, effect sizes, and fold changes 
for BOTH the full dataset and a downsampled subset.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# CONFIG — Edit these to match your real data
# --------------------------------------------------------------------------
CITY_DATA_PATHS = {
    'Clark_County_NV': 'clark_county_nv_indiv_predictions_and_mass.csv',
    'Urbana_Champaign_IL': 'urbana_champaign_il_indiv_predictions_and_mass.csv',
    'Sandwich_MA': 'sandwich_ma_indiv_predictions_and_mass.csv'
}

CITY_POPULATIONS = {
    'Clark_County_NV': 2398871,
    'Urbana_Champaign_IL': 238842,
    'Sandwich_MA': 2890
}

# Display names used in all figures/legends (keys must match CITY_DATA_PATHS keys)
CITY_DISPLAY_NAMES = {
    'Clark_County_NV': 'Clark County, NV',
    'Urbana_Champaign_IL': 'Urbana-Champaign, IL',
    'Sandwich_MA': 'Sandwich, MA'
}

OUTPUT_XLSX = 'city_differences_analysis.xlsx'
DOWNSAMPLE_N = 10000          # Number of prescriptions to sample per city
MIN_OBS_FOR_STATS = 10        # Minimum observations to run statistical tests

# --------------------------------------------------------------------------
# 1. LOAD AND PREPARE FULL DATASET
# --------------------------------------------------------------------------
print("1. Loading and preparing full datasets...")
combined_dfs = []
total_records_by_city = {}

for city_name, path in CITY_DATA_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file for {city_name} at: {path}")
    
    df_city = pd.read_csv(path)
    df_city['City'] = city_name
    df_city = df_city.rename(columns={'Predicted_Drug': 'Drug'})
    df_city['Drug'] = df_city['Drug'].astype(str).str.lower().str.strip()
    df_city = df_city.rename(columns={'Wastewater_Mass_mcg': 'Mass_Micrograms'})
    
    total_records_by_city[city_name] = len(df_city)
    combined_dfs.append(df_city)

df_all = pd.concat(combined_dfs, ignore_index=True)
print(f"   Loaded {len(df_all):,} total records.")
for city, count in total_records_by_city.items():
    print(f"   - {city}: {count:,} prescriptions")

# --------------------------------------------------------------------------
# Helper Functions for Effect Sizes and Fold Changes
# --------------------------------------------------------------------------
def get_cramers_v(chi2, n, shape):
    """Calculates Cramér's V for Chi-Square tests"""
    min_dim = min(shape) - 1
    if min_dim > 0 and n > 0:
        return np.sqrt(chi2 / (n * min_dim))
    return np.nan

def get_eta_squared(h_stat, n_total, k=3):
    """Calculates Eta-Squared for Kruskal-Wallis"""
    if n_total > k:
        eta_sq = (h_stat - k + 1) / (n_total - k)
        return max(0, eta_sq)
    return np.nan

def safe_fold_change(num, denom):
    """Safely calculates fold change, handling divisions by zero"""
    if denom == 0:
        return np.nan if num == 0 else np.inf
    return num / denom

# --------------------------------------------------------------------------
# 3. MASS LOAD & FOLD CHANGE CALCULATIONS
# --------------------------------------------------------------------------
print("\n3. Calculating population-normalized mass loads and fold changes...")

full_agg = df_all.groupby(['City', 'Drug'])['Mass_Micrograms'].sum().reset_index()
full_agg['Population'] = full_agg['City'].map(CITY_POPULATIONS)
full_agg['Norm_Load'] = full_agg['Mass_Micrograms'] / full_agg['Population']
mass_comparison = full_agg.pivot(index='Drug', columns='City', values='Norm_Load').fillna(0)

# Pairwise fold changes
mass_comparison['FC_A_B'] = mass_comparison.apply(lambda r: safe_fold_change(r['Clark_County_NV'], r['Urbana_Champaign_IL']), axis=1)
mass_comparison['FC_B_C'] = mass_comparison.apply(lambda r: safe_fold_change(r['Urbana_Champaign_IL'], r['Sandwich_MA']), axis=1)
mass_comparison['FC_A_C'] = mass_comparison.apply(lambda r: safe_fold_change(r['Clark_County_NV'], r['Sandwich_MA']), axis=1)

# --------------------------------------------------------------------------
# 4. COMPREHENSIVE STATISTICAL PIPELINE
# --------------------------------------------------------------------------
print("\n4. Running statistical pipeline...")

categorical_results = []
distribution_results = []
all_drugs = df_all['Drug'].unique()

total_full_rx = df_all['City'].value_counts()

for drug in all_drugs:
    df_drug_full = df_all[df_all['Drug'] == drug]

    # ------------------ CATEGORICAL TESTS (PRESENCE) ------------------
    full_counts = df_drug_full['City'].value_counts()
    full_contingency = np.array([[full_counts.get(c, 0), total_full_rx[c] - full_counts.get(c, 0)] for c in CITY_DATA_PATHS.keys()])

    f_chi2, f_p, f_cv = np.nan, np.nan, np.nan
    if full_contingency[:, 0].sum() >= MIN_OBS_FOR_STATS:
        f_chi2, f_p, _, _ = stats.chi2_contingency(full_contingency)
        f_cv = get_cramers_v(f_chi2, full_contingency.sum(), full_contingency.shape)

    categorical_results.append({
        'Drug': drug,
        'Chi2': f_chi2, 'p_value': f_p, 'CramersV': f_cv, 'Sig_p<0.05': f_p < 0.05 if not np.isnan(f_p) else False
    })

    # ------------------ DISTRIBUTION TESTS (MASS) ------------------
    full_groups = [df_drug_full[df_drug_full['City'] == c]['Mass_Micrograms'].dropna().values for c in CITY_DATA_PATHS.keys()]
    valid_full = [g for g in full_groups if len(g) >= MIN_OBS_FOR_STATS]

    f_kw, f_kw_p, f_eta = np.nan, np.nan, np.nan
    if len(valid_full) == 3:
        f_kw, f_kw_p = stats.kruskal(*valid_full)
        f_eta = get_eta_squared(f_kw, sum(len(g) for g in valid_full))

    distribution_results.append({
        'Drug': drug,
        'KW_Stat': f_kw, 'p_value': f_kw_p, 'EtaSquared': f_eta, 'Sig_p<0.05': f_kw_p < 0.05 if not np.isnan(f_kw_p) else False
    })

df_categorical = pd.DataFrame(categorical_results)
df_distribution = pd.DataFrame(distribution_results)

# --------------------------------------------------------------------------
# 5. WRITE WORKBOOK
# --------------------------------------------------------------------------
print(f"\n5. Exporting complete results to {OUTPUT_XLSX}")
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    mass_comparison.to_excel(writer, sheet_name='Mass_Loads_and_Fold_Changes')
    df_categorical.to_excel(writer, sheet_name='Presence_ChiSq_Effect_Sizes', index=False)
    df_distribution.to_excel(writer, sheet_name='Distribution_KW_Effect_Sizes', index=False)

print("\nAnalysis Complete! File generated successfully.")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# --------------------------------------------------------------------------
# APPLY DISPLAY NAMES TO CITY COLUMNS (used by all figures below)
# --------------------------------------------------------------------------
df_all_plot = df_all.copy()
df_all_plot['City'] = df_all_plot['City'].map(CITY_DISPLAY_NAMES)

# Order used consistently across all 3 panels (legend order + x-axis order)
city_order = [CITY_DISPLAY_NAMES[k] for k in CITY_DATA_PATHS.keys()]

# --------------------------------------------------------------------------
# PREP DATA FOR FIGURE 1 (Prescription rates by city, full dataset)
# --------------------------------------------------------------------------
print("\nPreparing Figure 1 data: Prescription rates for Chi-Square hits...")

city_totals = df_all_plot['City'].value_counts()
drug_counts = df_all_plot.groupby(['City', 'Drug']).size().reset_index(name='Rx_Count')
drug_counts['City_Total'] = drug_counts['City'].map(city_totals)
drug_counts['Prescription_Rate_Pct'] = (drug_counts['Rx_Count'] / drug_counts['City_Total']) * 100

apis_to_plot_fig1 = [
    'amoxicillin',
    'cefdinir',
    'chlorhexidine',
    'dorzolamide',
    'fluticasone',
    'imiquimod',
    'lactulose',
    'levofloxacin',
    'mupirocin',
    'nitroglycerin',
    'ondansetron',
    'oxymetazoline',
    'triamcinolone',
    'no prescriptions'
]

apis_to_plot_fig2 = [
    'amoxicillin',
    'cefdinir',
    'chlorhexidine',
    'dorzolamide',
    'fluticasone',
    'imiquimod',
    'lactulose',
    'levofloxacin',
    'mupirocin',
    'nitroglycerin',
    'ondansetron',
    'oxymetazoline',
    'triamcinolone'
]


df_fig1 = drug_counts[drug_counts['Drug'].isin(apis_to_plot_fig2)]

# --------------------------------------------------------------------------
# PREP DATA FOR FIGURE 2 (Mass load fold-change relative to Clark County)
# --------------------------------------------------------------------------
print("Preparing Figure 2 data: Mass load fold-change vs. Clark County...")

fc_long = pd.DataFrame({
    'Drug': mass_comparison.index.tolist() * 2,
    'City': (['Urbana-Champaign, IL / Clark County, NV'] * len(mass_comparison)) +
            (['Sandwich, MA / Clark County, NV'] * len(mass_comparison)),
    'Fold_Change': (
        mass_comparison['Urbana_Champaign_IL'] / mass_comparison['Clark_County_NV']
    ).tolist() + (
        mass_comparison['Sandwich_MA'] / mass_comparison['Clark_County_NV']
    ).tolist()
})
fc_long = fc_long.replace([np.inf, -np.inf], np.nan).dropna(subset=['Fold_Change'])

df_fig2 = fc_long[fc_long['Drug'].isin(apis_to_plot_fig2)]

# # --------------------------------------------------------------------------
# # PREP DATA FOR PANEL 3 (No-prescription baseline rates by city)
# # --------------------------------------------------------------------------
# # The source data marks a record as having no active prescription with the
# # literal string 'no prescriptions' in the Predicted_Drug/Drug column (the
# # Quantity/Form/Strength/Day_Supply fields are also blank on those rows).
# # After the existing .str.lower().str.strip() upstream, this value survives
# # unchanged, so we match on it directly.
# print("Preparing baseline no-prescription rate data...")

# NO_RX_LABEL = 'no prescriptions'

# no_rx_data = []
# for city_key in CITY_DATA_PATHS.keys():
#     city_display = CITY_DISPLAY_NAMES[city_key]
#     df_city_all = df_all[df_all['City'] == city_key]
#     total_records = len(df_city_all)
#     no_rx_count = (df_city_all['Drug'] == NO_RX_LABEL).sum() if total_records > 0 else 0
#     no_rx_pct = (no_rx_count / total_records * 100) if total_records > 0 else 0
#     no_rx_data.append({
#         'City': city_display,
#         'No_Prescription_Rate_Pct': no_rx_pct,
#         'Count': no_rx_count,
#         'Total_Records': total_records
#     })

# df_no_rx_plot = pd.DataFrame(no_rx_data)

# --------------------------------------------------------------------------
# COMBINED FIGURE: fig1 + fig2 on top row, no-rx plot centered underneath
# --------------------------------------------------------------------------
print("\nGenerating combined figure...")

sns.set_theme(style="ticks")

fig = plt.figure(figsize=(16, 12))
# 4-column grid: top row splits 2/2 across fig1 & fig2; bottom row's plot
# spans the middle 2 columns so it's centered under the pair above it.
gs = gridspec.GridSpec(
    2, 4,
    height_ratios=[1, 1],
    hspace=0.55,
    wspace=0.6
)

ax1 = fig.add_subplot(gs[0, 0:2])   # top-left: Figure 1
ax2 = fig.add_subplot(gs[0, 2:4])   # top-right: Figure 2
#ax3 = fig.add_subplot(gs[1, 1:3])   # bottom-center: No-Rx plot

# --- Panel 1: Prescription rates ---
sns.barplot(
    data=df_fig1,
    x='Prescription_Rate_Pct',
    y='Drug',
    hue='City',
    hue_order=city_order,
    palette='muted',
    edgecolor='black',
    linewidth=1,
    ax=ax1
)
#ax1.set_xscale('log')
ax1.set_xlabel('Predicted Prescription Rate (%)', fontsize=18, fontweight='bold', labelpad=10)
ax1.set_ylabel('API', fontsize=18, fontweight='bold', labelpad=10)
ax1.set_yticklabels(
    [t.get_text().capitalize() for t in ax1.get_yticklabels()],
    fontsize=13
)
ax1.legend(title='City', title_fontsize='18', fontsize=13, loc='lower right', frameon=True)
sns.despine(ax=ax1, trim=True)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- Panel 2: Mass load fold-change vs. Clark County ---
fc_city_order = [
    'Urbana-Champaign, IL / Clark County, NV',
    'Sandwich, MA / Clark County, NV'
]

sns.barplot(
    data=df_fig2,
    x='Fold_Change',
    y='Drug',
    hue='City',
    hue_order=fc_city_order,
    palette='muted',
    edgecolor='black',
    linewidth=1,
    ax=ax2
)
ax2.axvline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.6, zorder=0)
ax2.set_ylabel('API', fontsize=18, fontweight='bold', labelpad=10)
ax2.set_xlabel('Fold Change in Mass Load', fontsize=18, fontweight='bold', labelpad=10)
ax2.set_yticklabels(
    [t.get_text().capitalize() for t in ax2.get_yticklabels()],
    fontsize=13
)
ax2.legend(title='City', title_fontsize='18', fontsize=13, loc='lower right', frameon=True)
sns.despine(ax=ax2, trim=True)
ax2.grid(axis='x', linestyle='--', alpha=0.5)

# # --- Panel 3: No-prescription baseline rates ---
# sns.barplot(
#     data=df_no_rx_plot,
#     x='City',
#     y='No_Prescription_Rate_Pct',
#     order=city_order,
#     palette='muted',
#     edgecolor='black',
#     linewidth=1.2,
#     width=0.5,
#     ax=ax3
# )
# ax3.set_xlabel('Municipality', fontsize=11, fontweight='bold', labelpad=10)
# ax3.set_ylabel('No Active Prescription (%)', fontsize=11, fontweight='bold', labelpad=10)
# ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=10)
# max_val = df_no_rx_plot['No_Prescription_Rate_Pct'].max()
# ax3.set_ylim(0, max_val * 1.15 if max_val > 0 else 1)
# sns.despine(ax=ax3, trim=True)
# ax3.grid(axis='y', linestyle='--', alpha=0.5)

# --------------------------------------------------------------------------
# SAVE
# --------------------------------------------------------------------------
output_combined_png = 'combined_figures.png'
output_combined_svg = 'combined_figures.svg'
plt.savefig(output_combined_png, dpi=300, bbox_inches='tight')
plt.savefig(output_combined_svg, format='svg', bbox_inches='tight')
plt.close()