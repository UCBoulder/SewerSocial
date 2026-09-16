## Training with cross-validation for the TabICL base model. The TabICL model predicts the target class (Drug) from the features (age, race/ethnicity, sex, family income, insurance coverage, prescription strength, prescription day supply, prescription quantity, and prescription form).

# NOTE: Set your cache variable based on your system path below in Line 204.

# prep to time code
import time
from datetime import timedelta

# parallelize code
import os
import torch

# let OS manage threads and don't let PyTorch conflict with GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# check GPU available
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True 

# Install necessary libraries
import pandas as pd
import numpy as np
import joblib
from tabicl import TabICLClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report
)
from permetrics import ClassificationMetric
from sklearn.utils import shuffle
import gc

start_total = time.perf_counter() 

print("Loading and preprocessing data...")
start_prep = time.perf_counter()

# Load dataset
data = pd.read_csv(
    "super_integrated_data.csv", 
    sep=None, 
    engine='python', 
    encoding='utf-8-sig'
)

# shrink dataset size
data = data.sample(n=100000, random_state=42)

# Downcast data to preserve memory
for col in data.select_dtypes(include=['float64']).columns:
    data[col] = data[col].astype('float32')
for col in data.select_dtypes(include=['int64']).columns:
    data[col] = data[col].astype('int32')

# Fill missing values (for those rows where people had "no prescriptions" as the target variable), done separately for numeric vs. categorical data
numeric_prescription_cols = ['Quantity', 'Strength', 'Day_Supply']
mask = data['Drug'] == 'no prescriptions'
data.loc[mask, 'Form'] = data.loc[mask, 'Form'].fillna('-1')

# Define feature types
numeric_features = ['Age', 'Family_income', 'Quantity', 'Strength', 'Day_Supply']
categorical_features = ['Sex', 'Insurance_coverage', 'Race_ethnicity', 'Form']
target_variable = 'Drug'

# Cast feature types
for col in categorical_features:
    data[col] = data[col].astype('category')

# Encode target variable 
le = LabelEncoder()
data[target_variable] = le.fit_transform(data[target_variable].astype(str))

# Imputation functions for Age, Strength, and Day_Supply.
# Applied within each fold to prevent data leakage.
# Consistent with all other base models.

def fit_age_medians(df):
    df = df.copy()

    income_bin_edges = {}
    income_bracket = pd.Series(index=df.index, dtype='float64')
    for year, group in df.groupby('Year'):
        try:
            bins, edges = pd.qcut(
                group['Family_income'], 4, labels=False,
                duplicates='drop', retbins=True
            )
            income_bin_edges[year] = edges
            income_bracket.loc[group.index] = bins + 1
        except ValueError:
            income_bin_edges[year] = None
    df['income_bracket'] = income_bracket

    hh_medians    = df.groupby(['Year', 'Household_ID'])['Age'].median()
    grp_medians   = df.groupby(['Year', 'income_bracket', 'Insurance_coverage'])['Age'].median()
    yr_medians    = df.groupby('Year')['Age'].median()
    global_median = df['Age'].median()

    return {
        'income_bin_edges': income_bin_edges,
        'hh_medians': hh_medians,
        'grp_medians': grp_medians,
        'yr_medians': yr_medians,
        'global_median': global_median,
        'available_years': sorted(yr_medians.index.unique().tolist()),
    }

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

    # Map every row's Year to its nearest training year for all lookups below.
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

def fit_strength_medians(df):
    mask = df['Drug'] != 'no prescriptions'
    yr_drug_medians = df.loc[mask].groupby(['Year', 'Drug'])['Strength'].median()
    drug_medians    = df.loc[mask].groupby('Drug')['Strength'].median()
    global_median   = df.loc[mask, 'Strength'].median()
    return {
        'yr_drug_medians': yr_drug_medians,
        'drug_medians': drug_medians,
        'global_median': global_median,
    }

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

def fit_day_supply_medians(df):
    mask = df['Drug'] != 'no prescriptions'
    yr_drug_medians = df.loc[mask].groupby(['Year', 'Drug'])['Day_Supply'].median()
    drug_medians    = df.loc[mask].groupby('Drug')['Day_Supply'].median()
    global_median   = df.loc[mask, 'Day_Supply'].median()
    return {
        'yr_drug_medians': yr_drug_medians,
        'drug_medians': drug_medians,
        'global_median': global_median,
    }

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

end_prep = time.perf_counter()
print(f"Data Prep Time: {timedelta(seconds=end_prep - start_prep)}")

print("Starting final model training...")
start_train = time.perf_counter()

# Updated path for your Windows machine (CHANGE THIS FOR YOUR MACHINE)
cache_dir = r"C:\Users\Owner\vanessa\tabicl_cache"
# test if directory exists
os.makedirs(cache_dir, exist_ok=True)

# Setup cross validation
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) 
fold_results          = []
per_drug_recall_folds = []

# Loop through folds
for fold, (train_idx, val_idx) in enumerate(sgkf.split(data, y=data[target_variable], groups=data['Person_ID'])):
    # Split the raw data
    train_fold = data.iloc[train_idx].copy()
    val_fold = data.iloc[val_idx].copy()

    # Impute Age, Strength, Day_Supply within each fold.
    # Imputation is applied on train and val separately to prevent data leakage.
    age_medians = fit_age_medians(train_fold)
    train_fold = apply_age_medians(train_fold, age_medians)
    val_fold   = apply_age_medians(val_fold, age_medians)

    strength_medians = fit_strength_medians(train_fold)
    train_fold = apply_strength_medians(train_fold, strength_medians)
    val_fold   = apply_strength_medians(val_fold, strength_medians)

    day_supply_medians = fit_day_supply_medians(train_fold)
    train_fold = apply_day_supply_medians(train_fold, day_supply_medians)
    val_fold   = apply_day_supply_medians(val_fold, day_supply_medians)

    # set sentinel value for "no prescriptions" rows
    train_fold.fillna(-1, inplace=True)
    val_fold.fillna(-1, inplace=True)

    # Feature selection
    features = numeric_features + categorical_features
    X_train, y_train = train_fold[features], train_fold[target_variable] 
    X_val, y_val = val_fold[features], val_fold[target_variable]

    # ensure there are not any unpredictable target classes
    train_classes = set(y_train)
    val_classes = set(y_val)
    
    missing_in_train = val_classes - train_classes
    missing_in_val = train_classes - val_classes

    print(f"\nFold {fold+1} Diagnostics")
    print(f"Drugs in Test but NOT in Train (Unpredictable): {len(missing_in_train)}")
    print(f"Drugs in Train but NOT in Test: {len(missing_in_val)}")
    
    if missing_in_train:
        unpredictable_drugs = le.inverse_transform(list(missing_in_train))
        print(f"Example unpredictable drugs in this fold: {list(unpredictable_drugs)[:5]}")

    # Fit model
    clf = TabICLClassifier(
        device="cuda",              
        use_amp="auto",            
        n_jobs=4,                 
        support_many_classes=True, 
        offload_mode="disk",        
        disk_offload_dir=cache_dir, 
        batch_size=4,               
        n_estimators=8,             
        kv_cache=False,             
        softmax_temperature=0.9,    
        average_logits=True,       
        use_fa3="auto"
    )
    
    clf.fit(X_train, y_train)
    print(f'Fold {fold} training complete.')
    y_pred = clf.predict(X_val)

    # Assess performance metrics

    # from sklearn
    acc   = accuracy_score(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)
    mcc   = matthews_corrcoef(y_val, y_pred)

     # from permetrics (using macro and micro averaging)
    # Macro: average per class equally — treats rare and common drugs equally.
    # Micro: aggregate all counts — dominated by the most common drugs.
    evaluator = ClassificationMetric(y_val.values, y_pred)

    macro_precision = evaluator.precision_score(average='macro')
    micro_precision = evaluator.precision_score(average='micro')
    macro_recall    = evaluator.recall_score(average='macro')
    micro_recall    = evaluator.recall_score(average='micro')
    macro_f1        = evaluator.f1_score(average='macro')
    micro_f1        = evaluator.f1_score(average='micro')

    # F2 score weights recall twice as much as precision.
    # Higher recall is more important here because missing a drug class
    # means underestimating its wastewater load.
    macro_f2 = evaluator.fbeta_score(beta=2, average='macro')
    micro_f2 = evaluator.fbeta_score(beta=2, average='micro')

    fold_results.append({
        'fold': fold,
        'accuracy': acc,
        'cohen_kappa': kappa,
        'mcc': mcc,
        'macro_precision': macro_precision,
        'micro_precision': micro_precision,
        'macro_recall': macro_recall,
        'micro_recall': micro_recall,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_f2': macro_f2,
        'micro_f2': micro_f2,
    })

    # Per-drug recall for this fold.
    # Used for ensemble model selection — the ensemble picks the best model per drug.
    report = classification_report(
        y_val, y_pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    drug_recalls         = {drug: report[drug]['recall'] for drug in le.classes_ if drug in report}
    drug_recalls['fold'] = fold
    per_drug_recall_folds.append(drug_recalls)

    print(f'Fold {fold} results:')
    print(f'  Accuracy:      {acc:.4f}')
    print(f'  Cohen Kappa:   {kappa:.4f}')
    print(f'  MCC:           {mcc:.4f}')
    print(f'  Macro Recall:  {macro_recall:.4f}')
    print(f'  Micro Recall:  {micro_recall:.4f}')
    print(f'  Macro F2:      {macro_f2:.4f}')

    del clf, X_train, y_train, train_fold, val_fold
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

print("ALL FOLDS COMPLETE")

# Summarize cross-validation results across all 5 folds.
# Report mean and standard deviation for each metric.
# Standard deviation shows how stable the model is across different data splits.
results_df = pd.DataFrame(fold_results)

print('CV RESULTS — MEAN +/- STD ACROSS 5 FOLDS')
print('='*55)
metric_cols = [c for c in results_df.columns if c != 'fold']
for col in metric_cols:
    mean = results_df[col].mean()
    std  = results_df[col].std()
    print(f'  {col:<25}: {mean:.4f} +/- {std:.4f}')

results_df.to_csv('tabicl_super_cv_results.csv', index=False)
print('\nCV results saved to tabicl_super_cv_results.csv')

# Compute average per-drug recall across all 5 folds.
# This is the key output for ensemble model selection.
# The ensemble picks whichever base model has the highest recall for each drug.
per_drug_df = pd.DataFrame(per_drug_recall_folds)
drug_cols   = [c for c in per_drug_df.columns if c != 'fold']

mean_drug_recall = per_drug_df[drug_cols].mean().reset_index()
mean_drug_recall.columns = ['Drug', 'Mean_Recall_TabICL']
mean_drug_recall = mean_drug_recall.sort_values('Mean_Recall_TabICL', ascending=False)

print('Top 20 drugs by mean recall:')
print(mean_drug_recall.head(20).to_string(index=False))

print('\nBottom 20 drugs by mean recall:')
print(mean_drug_recall.tail(20).to_string(index=False))

mean_drug_recall.to_csv('tabicl_super_per_drug_recall.csv', index=False)
print('\nPer-drug recall saved to tabicl_super_per_drug_recall.csv')

# Train final model refitted on full data

# impute Age, Strength, Day_Supply
age_medians_final = fit_age_medians(data)
strength_medians_final = fit_strength_medians(data)
day_supply_medians_final = fit_day_supply_medians(data)

joblib.dump(age_medians_final, 'tabicl_super_age_medians.joblib')
joblib.dump(strength_medians_final, 'tabicl_super_strength_medians.joblib')
joblib.dump(day_supply_medians_final, 'tabicl_super_day_supply_medians.joblib')

data_final = apply_age_medians(data.copy(), age_medians_final)
fdata_final = apply_strength_medians(data_final, strength_medians_final)
data_final = apply_day_supply_medians(data_final, day_supply_medians_final) 

# set sentinel value for "no prescriptions" rows
data_final.fillna(-1, inplace=True)

# set final features and target variable
X_final = data_final[features]
y_final = data_final[target_variable]

# shuffle to separate refills
X_final, y_final = shuffle(X_final, y_final, random_state=42)
X_final = X_final.reset_index(drop=True)

# final fit of the model
final_clf = TabICLClassifier(
    device="cuda",              
    use_amp="auto",             
    n_jobs=4,               
    support_many_classes=True, 
    offload_mode="disk",        
    disk_offload_dir=cache_dir, 
    batch_size=4,             
    n_estimators=8,           
    kv_cache=False,          
    softmax_temperature=0.9,   
    average_logits=True         
)

final_clf.fit(X_final, y_final)

# save model for outputting
print("Saving final trained model...")
final_clf.save(
    "tabicl_trained_model.pkl",
    save_model_weights=True,   
    save_training_data=True,    
    save_kv_cache=False        
)
print("Model saved: weights and training context included.")

end_train = time.perf_counter()
train_duration = end_train - start_train
print(f"Final Model Fit Time: {timedelta(seconds=train_duration)}")