"""
Conducts chi-square and Mann-Whitney U tests to determine if there are statistically significant
differences in features between APIs included in the model and excluded from the model. Also checks
if there are statistically significant differences in features for those on a prescription versus in
the "no prescriptions" class. Finally, calculates fold changes in numeric features between APIs and
performs a logistic regression to compute odds ratios of having different features. Analysis conducted
on 2022 validation data based on included drug list from specialist mapping (see
ensemble_training_internal_validation_original.py).

"""

# import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# Configure data paths, label included APIs, and indicate other labels.
# --------------------------------------------------------------------------
DATA_PATH = 'super_data_2022.csv'
FULL_DRUG_LIST_PATH = 'common_years_drug_list.xlsx' 

INCLUDED_DRUGS = ["albuterol", "alendronate", "amoxicillin", "apixaban", "aspirin", "azelastine", "azithromycin", "benzonatate", "bimatoprost", "brimonidine", "budesonide", "carvedilol", "cefdinir", "chlorhexidine", "ciclopirox", "clavulanate", "clindamycin", "clobetasol", "clonidine", "clopidogrel", "colchicine", "cyclosporine", "dorzolamide", "fenofibrate", "fluconazole", "fluorouracil", "fluticasone", "gabapentin", "guanfacine", "hydrochlorothiazide", "hydrocortisone", "ibuprofen", "imiquimod", "ipratropium", "ketoconazole", "lactulose", "latanoprost", "levofloxacin", "lidocaine", "linaclotide", "liraglutide", "meloxicam", "metformin", "methotrexate", "methylprednisolone", "mometasone", "moxifloxacin", "mupirocin", "nitroglycerin", "no prescription required", "norethindrone", "ofloxacin", "olopatadine", "ondansetron", "oxymetazoline", "permethrin", "prednisolone", "rizatriptan", "senna", "silver sulfadiazine", "sumatriptan", "tamsulosin", "terbinafine", "timolol", "tiotropium", "tramadol", "triamcinolone"]

NO_RX_LABEL = 'no prescriptions'
OUTPUT_XLSX = 'drug_pattern_analysis.xlsx'
AGE_BINS = [0, 18, 35, 50, 65, 200]
AGE_LABELS = ['0-17', '18-34', '35-49', '50-64', '65+']
INCOME_BINS = [-1, 25000, 50000, 100000, np.inf]
INCOME_LABELS = ['<25k', '25-50k', '50-100k', '100k+']
TOP_N_FOR_PLOTS = 20

# --------------------------------------------------------------------------
# Functions to calculate Cramer's V and eta-squared effect sizes, fold
# changes, and summary statistics on features.
# --------------------------------------------------------------------------
def calculate_cramers_v(data, group_col, var_col):
    """Computes Cramer's V for categorical variables."""
    contingency = pd.crosstab(data[group_col], data[var_col])
    contingency = contingency.loc[contingency.sum(axis=1) > 0]
    if contingency.empty or contingency.values.sum() == 0:
        return np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
    return p, cramers_v

def calculate_eta_squared(data, group_col, var_col):
    """Computes Eta-Squared (n^2) effect size via a One-Way ANOVA breakdown."""
    clean_df = data[[group_col, var_col]].dropna()
    groups = [df_sub[var_col].values for _, df_sub in clean_df.groupby(group_col) if len(df_sub) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    f_stat, p_val = stats.f_oneway(*groups)
    df_between = len(groups) - 1
    df_within = sum(len(g) for g in groups) - len(groups)
    denominator = (f_stat * df_between) + df_within
    eta_squared = (f_stat * df_between) / denominator if denominator != 0 else np.nan
    return p_val, eta_squared

def calculate_fold_change(data, group_col, var_col, pos_group, neg_group):
    """Calculates the ratio of group means (Fold Change) safely."""
    mean_pos = data.loc[data[group_col] == pos_group, var_col].mean()
    mean_neg = data.loc[data[group_col] == neg_group, var_col].mean()
    if pd.isna(mean_neg) or mean_neg == 0:
        return np.nan
    return mean_pos / mean_neg

def group_summary(data, group_col, col, is_categorical, decimals=1):
    if is_categorical:
        tbl = pd.crosstab(data[group_col], data[col], normalize='index') * 100
        return tbl.round(decimals)
    else:
        return data.groupby(group_col)[col].agg(['count', 'mean', 'median', 'std']).round(decimals)

# --------------------------------------------------------------------------
# Load the 2022 data.
# --------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

df['Drug'] = df['Drug'].astype(str).str.lower().str.strip()
df['Form'] = df['Form'].astype(str).str.upper().str.strip()

print(f'Loaded {len(df):,} rows, {df["Drug"].nunique()} unique drugs.')

full_drug_list = pd.read_excel(FULL_DRUG_LIST_PATH, header=None)[0].astype(str).str.lower().str.strip().tolist()
full_drug_list = [d for d in full_drug_list if d != NO_RX_LABEL]

clean_included_drugs = [str(d).lower().strip() for d in INCLUDED_DRUGS]
included_set = set(d for d in clean_included_drugs if d != NO_RX_LABEL)
not_included_set = set(full_drug_list) - included_set

print(f'Full drug list config: {len(full_drug_list)} drugs')
print(f'  Included group:     {len(included_set)} drugs')
print(f'  Not-included group: {len(not_included_set)} drugs')

df_all = df[df['Drug'].isin(included_set | not_included_set)].copy()
df_all['Group'] = np.where(df_all['Drug'].isin(included_set), 'Included', 'Not Included')

print(f'Rows in comparison scope: {len(df_all):,} (Included: {(df_all["Group"]=="Included").sum():,}, Not Included: {(df_all["Group"]=="Not Included").sum():,})')

# --------------------------------------------------------------------------
# Run significance tests, fold change calculations, and summary statsitics
# on demographic features.
# --------------------------------------------------------------------------
person_all = df_all.sort_values('Observation_ID').drop_duplicates(subset='Person_ID', keep='first').copy()
people_ever_included = set(df_all.loc[df_all['Group'] == 'Included', 'Person_ID'])
person_all['Ever_Included'] = np.where(person_all['Person_ID'].isin(people_ever_included), 'Ever Included', 'Never Included')

print(f'Person-level demographic context: {len(person_all):,} unique people')

categorical_demos = ['Sex', 'Insurance_coverage', 'Race_ethnicity']
continuous_demos = ['Age', 'Family_income']

demo_test_results = []

# Cramer's V
for demo in categorical_demos:
    p_full, v_full = calculate_cramers_v(person_all, 'Ever_Included', demo)

    demo_test_results.append({
        'Variable': demo, 'Metric_Type': "Cramer's V",
        'P_Value': p_full, 'Effect_Size': v_full,
        'Fold_Change': np.nan
    })

# Eta-squared and fold change
for demo in continuous_demos:
    p_full, eta_full = calculate_eta_squared(person_all, 'Ever_Included', demo)
    fc = calculate_fold_change(person_all, 'Ever_Included', demo, 'Ever Included', 'Never Included')

    demo_test_results.append({
        'Variable': demo, 'Metric_Type': 'Eta-Squared',
        'P_Value': p_full, 'Effect_Size': eta_full,
        'Fold_Change': fc
    })

demo_test_df = pd.DataFrame(demo_test_results)

demo_summary_sheets = {
    'Demo_Sex_by_Group': group_summary(person_all, 'Ever_Included', 'Sex', True),
    'Demo_Insurance_by_Group': group_summary(person_all, 'Ever_Included', 'Insurance_coverage', True),
    'Demo_Race_by_Group': group_summary(person_all, 'Ever_Included', 'Race_ethnicity', True),
    'Demo_Age_by_Group': group_summary(person_all, 'Ever_Included', 'Age', False),
    'Demo_Income_by_Group': group_summary(person_all, 'Ever_Included', 'Family_income', False),
}

# --------------------------------------------------------------------------
# Logistic regression for demographic variables
# --------------------------------------------------------------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf

person_all['Ever_Included_Flag'] = np.where(person_all['Ever_Included'] == 'Ever Included', 1, 0)
reg_data_inc = person_all[['Ever_Included_Flag', 'Age', 'Sex', 'Race_ethnicity', 'Insurance_coverage', 'Family_income']].dropna()
reg_data_inc['Family_income_10k'] = reg_data_inc['Family_income'] / 10000

logit_model_inc = smf.logit(
    'Ever_Included_Flag ~ Age + C(Sex) + C(Race_ethnicity, Treatment(reference="Non-Hispanic White")) '
    '+ C(Insurance_coverage, Treatment(reference="Private")) + Family_income_10k',
    data=reg_data_inc
).fit(disp=0)

logit_summary_inc = pd.DataFrame({
    'Coefficient': logit_model_inc.params,
    'Std_Error': logit_model_inc.bse,
    'p_value': logit_model_inc.pvalues,
    'Odds_Ratio': np.exp(logit_model_inc.params),
    'OR_CI_Lower': np.exp(logit_model_inc.conf_int()[0]),
    'OR_CI_Upper': np.exp(logit_model_inc.conf_int()[1]),
}).reset_index().rename(columns={'index': 'Predictor'})
logit_summary_inc['Significant_p<0.05'] = logit_summary_inc['p_value'] < 0.05

# --------------------------------------------------------------------------
# Run significance tests, fold change calculations, and summary statsitics
# on prescription features.
# --------------------------------------------------------------------------
prescribing_test_results = []
prescribing_continuous = ['Quantity', 'Strength', 'Day_Supply']

# Eta-squared and fold change
for var in prescribing_continuous:
    p_full, eta_full = calculate_eta_squared(df_all, 'Group', var)
    fc = calculate_fold_change(df_all, 'Group', var, 'Included', 'Not Included')

    prescribing_test_results.append({
        'Variable': var, 'Metric_Type': 'Eta-Squared',
        'P_Value': p_full, 'Effect_Size': eta_full,
        'Fold_Change': fc
    })

# Cramer's V
p_full, v_full = calculate_cramers_v(df_all, 'Group', 'Form')
prescribing_test_results.append({
    'Variable': 'Form', 'Metric_Type': "Cramer's V",
    'P_Value': p_full, 'Effect_Size': v_full,
    'Fold_Change': np.nan
})

prescribing_test_df = pd.DataFrame(prescribing_test_results)

prescribing_summary_sheets = {
    'Quantity_by_Group': group_summary(df_all, 'Group', 'Quantity', False),
    'Strength_by_Group': group_summary(df_all, 'Group', 'Strength', False),
    'Day_Supply_by_Group': group_summary(df_all, 'Group', 'Day_Supply', False),
    'Form_by_Group': group_summary(df_all, 'Group', 'Form', True, decimals=5),
    'Form_by_Group_Counts': pd.crosstab(df_all['Group'], df_all['Form']),
}

# --------------------------------------------------------------------------
# Run significance tests, fold change calculations, and summary statsitics
# on demographic features for "no prescriptions" class.
# --------------------------------------------------------------------------
df_norx_compare = df[df['Drug'].isin(included_set | not_included_set | {NO_RX_LABEL})].copy()
df_norx_compare['Rx_Status'] = np.where(df_norx_compare['Drug'] == NO_RX_LABEL, 'No Prescription', 'Has Prescription')

person_level = df_norx_compare.sort_values('Observation_ID').drop_duplicates(subset='Person_ID', keep='first').copy()
people_with_rx = set(df_norx_compare.loc[df_norx_compare['Rx_Status'] == 'Has Prescription', 'Person_ID'])
person_level['No_Rx_Flag'] = np.where(person_level['Person_ID'].isin(people_with_rx), 0, 1)
person_level['Rx_Status'] = np.where(person_level['No_Rx_Flag'] == 1, 'No Prescription', 'Has Prescription')

# Cramer's V
norx_test_results = []
for demo in categorical_demos:
    p_full, v_full = calculate_cramers_v(person_level, 'Rx_Status', demo)
    norx_test_results.append({
        'Variable': demo, 'Metric_Type': "Cramer's V",
        'P_Value': p_full, 'Effect_Size': v_full,
        'Fold_Change': np.nan
    })

# Eta-squared and effect size
for demo in continuous_demos:
    p_full, eta_full = calculate_eta_squared(person_level, 'Rx_Status', demo)
    fc = calculate_fold_change(person_level, 'Rx_Status', demo, 'No Prescription', 'Has Prescription')
    norx_test_results.append({
        'Variable': demo, 'Metric_Type': 'Eta-Squared',
        'P_Value': p_full, 'Effect_Size': eta_full,
        'Fold_Change': fc
    })

norx_test_df = pd.DataFrame(norx_test_results)

norx_demo_summary_sheets = {
    'NoRx_Sex': group_summary(person_level, 'Rx_Status', 'Sex', True),
    'NoRx_Insurance': group_summary(person_level, 'Rx_Status', 'Insurance_coverage', True),
    'NoRx_Race': group_summary(person_level, 'Rx_Status', 'Race_ethnicity', True),
    'NoRx_Age': group_summary(person_level, 'Rx_Status', 'Age', False),
    'NoRx_Income': group_summary(person_level, 'Rx_Status', 'Family_income', False),
}

# --------------------------------------------------------------------------
# Logistic regression for "no prescriptions" analysis
# --------------------------------------------------------------------------
reg_data = person_level[['No_Rx_Flag', 'Age', 'Sex', 'Race_ethnicity', 'Insurance_coverage', 'Family_income']].dropna()
reg_data['Family_income_10k'] = reg_data['Family_income'] / 10000

logit_model = smf.logit(
    'No_Rx_Flag ~ Age + C(Sex) + C(Race_ethnicity, Treatment(reference="Non-Hispanic White")) '
    '+ C(Insurance_coverage, Treatment(reference="Private")) + Family_income_10k',
    data=reg_data
).fit(disp=0)

logit_summary = pd.DataFrame({
    'Coefficient': logit_model.params,
    'Std_Error': logit_model.bse,
    'p_value': logit_model.pvalues,
    'Odds_Ratio': np.exp(logit_model.params),
    'OR_CI_Lower': np.exp(logit_model.conf_int()[0]),
    'OR_CI_Upper': np.exp(logit_model.conf_int()[1]),
}).reset_index().rename(columns={'index': 'Predictor'})
logit_summary['Significant_p<0.05'] = logit_summary['p_value'] < 0.05

# --------------------------------------------------------------------------
# Get summary stats per drug
# --------------------------------------------------------------------------
per_drug_summary = df_all.groupby(['Group', 'Drug']).agg(
    N=('Drug', 'size'),
    Age_Mean=('Age', 'mean'),
    Income_Mean=('Family_income', 'mean'),
    Quantity_Mean=('Quantity', 'mean'),
    Strength_Mean=('Strength', 'mean'),
    Day_Supply_Mean=('Day_Supply', 'mean'),
).round(1).reset_index().sort_values(['Group', 'N'], ascending=[True, False])

# --------------------------------------------------------------------------
# Save as Excel multi-sheet workbook
# --------------------------------------------------------------------------
def clean_sheet_name(name):
    for char in ['\\', '/', '?', '*', ':', '[', ']']:
        name = name.replace(char, '-')
    return name[:31]

with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    demo_test_df.to_excel(writer, sheet_name=clean_sheet_name('Demo_Comparison_Tests'), index=False)
    for sheet_name, tbl in demo_summary_sheets.items():
        tbl.to_excel(writer, sheet_name=clean_sheet_name(sheet_name))

    prescribing_test_df.to_excel(writer, sheet_name=clean_sheet_name('Prescribing_Comparison_Tests'), index=False)
    for sheet_name, tbl in prescribing_summary_sheets.items():
        tbl.to_excel(writer, sheet_name=clean_sheet_name(sheet_name))

    logit_summary_inc.to_excel(writer, sheet_name=clean_sheet_name('EverIncluded_Logistic_Reg'), index=False)
    norx_test_df.to_excel(writer, sheet_name=clean_sheet_name('NoRx_vs_HasRx_Tests'), index=False)

    for sheet_name, tbl in norx_demo_summary_sheets.items():
        tbl.to_excel(writer, sheet_name=clean_sheet_name(sheet_name))

    logit_summary.to_excel(writer, sheet_name=clean_sheet_name('NoRx_Logistic_Regression'), index=False)
    per_drug_summary.to_excel(writer, sheet_name=clean_sheet_name('Per_Drug_Detail'), index=False)