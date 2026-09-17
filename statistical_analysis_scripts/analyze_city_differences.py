"""
Conducts chi-square and Kruskal-Wallis tests to determine if there are statistically significant
differences in prescription rate or mass load of APIs between the communities of Sandwich, MA,
Urbana-Champaign, IL, and Clark County, NV. Also calculates fold change in mass load between these
communities.

"""

# import necessary libraries
import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# Configure the external dataset paths and info (edit as needed)
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

# Display names used in all figures/legends
CITY_DISPLAY_NAMES = {
    'Clark_County_NV': 'Clark County, NV',
    'Urbana_Champaign_IL': 'Urbana-Champaign, IL',
    'Sandwich_MA': 'Sandwich, MA'
}

OUTPUT_XLSX = 'city_differences_analysis.xlsx'
MIN_OBS_FOR_STATS = 10        # Minimum observations to run statistical tests

# --------------------------------------------------------------------------
# Load and prepare data per community
# --------------------------------------------------------------------------
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
# Define functions to find Cramer's V effect size, eta-squared effect size,
# and fold change.
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
# Calculates population-normalized mass loads and fold changes.
# --------------------------------------------------------------------------
full_agg = df_all.groupby(['City', 'Drug'])['Mass_Micrograms'].sum().reset_index()
full_agg['Population'] = full_agg['City'].map(CITY_POPULATIONS)
full_agg['Norm_Load'] = full_agg['Mass_Micrograms'] / full_agg['Population']
mass_comparison = full_agg.pivot(index='Drug', columns='City', values='Norm_Load').fillna(0)

# Pairwise fold changes by community
mass_comparison['FC_A_B'] = mass_comparison.apply(lambda r: safe_fold_change(r['Clark_County_NV'], r['Urbana_Champaign_IL']), axis=1)
mass_comparison['FC_B_C'] = mass_comparison.apply(lambda r: safe_fold_change(r['Urbana_Champaign_IL'], r['Sandwich_MA']), axis=1)
mass_comparison['FC_A_C'] = mass_comparison.apply(lambda r: safe_fold_change(r['Clark_County_NV'], r['Sandwich_MA']), axis=1)

# --------------------------------------------------------------------------
# Run the chi-square and Kruskal-Wallis tests
# --------------------------------------------------------------------------
categorical_results = []
distribution_results = []
all_drugs = df_all['Drug'].unique()

total_full_rx = df_all['City'].value_counts()

for drug in all_drugs:
    df_drug_full = df_all[df_all['Drug'] == drug]

    # ------------------ Chi-squared test (pharmaceutical presence) ------------------
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

    # ------------------ Kruskal-Wallis test (pharmaceutical mass load) ------------------
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
# Write statistical analysis results to Excel workbook
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
# Apply correct display names to figures for each community
# --------------------------------------------------------------------------
df_all_plot = df_all.copy()
df_all_plot['City'] = df_all_plot['City'].map(CITY_DISPLAY_NAMES)

city_order = [CITY_DISPLAY_NAMES[k] for k in CITY_DATA_PATHS.keys()]

# --------------------------------------------------------------------------
# Prep data for Fig 1 (Prescription rates for pharmaceuticals with fold
# change >= 2)
# --------------------------------------------------------------------------
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
    'no prescriptions' # add this line to get "no prescriptions" bars (manually added into Fig1 to preserve scaling)
]

df_fig1 = drug_counts[drug_counts['Drug'].isin(apis_to_plot_fig1)]
df_fig1.to_csv("df_fig1.csv")

# --------------------------------------------------------------------------
# Prep data for Fig 2 (Mass load fold-change relative to Clark County for 
# pharmaceuticals with fold change >= 2)
# --------------------------------------------------------------------------
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
fc_long = fc_long[fc_long['Fold_Change'] > 0]  # log undefined for 0/negative
fc_long['Log2_Fold_Change'] = np.log2(fc_long['Fold_Change'])

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

df_fig2 = fc_long[fc_long['Drug'].isin(apis_to_plot_fig2)]
df_fig2.to_csv("df_fig2.csv")

# --------------------------------------------------------------------------
# Combining Fig 1 and 2
# --------------------------------------------------------------------------
sns.set_theme(style="ticks")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(
    2, 4,
    height_ratios=[1, 1],
    hspace=0.55,
    wspace=0.6
)

ax1 = fig.add_subplot(gs[0, 0:2])  
ax2 = fig.add_subplot(gs[0, 2:4])   

# --- Fig 1: Prescription rates ---
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
ax1.set_xlabel('Predicted Prescription Rate (%)', fontsize=18, fontweight='bold', labelpad=10)
ax1.set_ylabel('API', fontsize=18, fontweight='bold', labelpad=10)
ax1.set_yticklabels(
    [t.get_text().capitalize() for t in ax1.get_yticklabels()],
    fontsize=13
)
ax1.legend(title='City', title_fontsize='18', fontsize=13, loc='lower right', frameon=True)
sns.despine(ax=ax1, trim=True)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# ---Fig 2: Mass load fold-change vs. Clark County ---
fc_city_order = [
    'Urbana-Champaign, IL / Clark County, NV',
    'Sandwich, MA / Clark County, NV'
]

sns.barplot(
    data=df_fig2,
    x='Log2_Fold_Change',
    y='Drug',
    hue='City',
    hue_order=fc_city_order,
    palette='muted',
    edgecolor='black',
    linewidth=1,
    ax=ax2
)
ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.6, zorder=0)
ax2.set_ylabel('API', fontsize=18, fontweight='bold', labelpad=10)
ax2.set_xlabel('Log$_2$ Fold Change in Mass Load', fontsize=18, fontweight='bold', labelpad=10)
ax2.set_yticklabels(
    [t.get_text().capitalize() for t in ax2.get_yticklabels()],
    fontsize=13
)
ax2.legend(title='City', title_fontsize='18', fontsize=13, loc='lower right', frameon=True)
sns.despine(ax=ax2, trim=True)
ax2.grid(axis='x', linestyle='--', alpha=0.5)

# --------------------------------------------------------------------------
# Save PNG and SVG versions of the figures
# --------------------------------------------------------------------------
output_combined_png = 'combined_figures.png'
output_combined_svg = 'combined_figures.svg'
plt.savefig(output_combined_png, dpi=300, bbox_inches='tight')
plt.savefig(output_combined_svg, format='svg', bbox_inches='tight')
plt.close()