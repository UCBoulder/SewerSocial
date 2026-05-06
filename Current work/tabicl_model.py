## The following script implements the TabIClv2 model in Python 3.13.12. The TabIClv2 model is provided with a multi-class classification problem with 9 features 
# (comprising demographic and prescription features) and with 218 target classes (217 drugs and "no prescription" for those without any drugs prescribed).

# Install necessary libraries
import pandas as pd
import numpy as np
from tabicl import TabICLClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report
)
from permetrics import ClassificationMetric
from sklearn.utils import shuffle

# Load dataset
data = pd.read_csv("data/super_integrated_data.csv")

# Fill missing values (for those rows where people had "no prescriptions" as the target variable), done separately for numeric vs. categorical data
numeric_prescription_cols = ['Quantity', 'Strength', 'Day_Supply']
data[numeric_prescription_cols] = data[numeric_prescription_cols].fillna(-1)
data['Form'] = data['Form'].fillna('-1')

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

# function to impute for Age variable, taking median by grouping by income and insurance with priority,
# then grouping by household, then taking median for that year. All medians are taken within years
def impute_age(df):

    df = df.copy()

    # Create income brackets
    df['income_bracket'] = (
        df.groupby('Year')['Family_income']
        .transform(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop') + 1)
    )

    # Calculate medians 
    # A. Household Median
    hh_meds = df.groupby(['Year', 'Household_ID'])['Age'].transform('median')
    
    # B. Income/Insurance/Year Median
    grp_meds = df.groupby(['Year', 'income_bracket', 'Insurance_coverage'])['Age'].transform('median')

    # C. Global Year Median
    yr_meds = df.groupby('Year')['Age'].transform('median')
    
    # D. Absolute Global Median (Final fallback)
    global_med = df['Age'].median()

    # Impute according to the hierarchy defined above
    df['Age'] = df['Age'].fillna(hh_meds)   # Priority 1
    df['Age'] = df['Age'].fillna(grp_meds)  # Priority 2
    df['Age'] = df['Age'].fillna(yr_meds)   # Priority 3
    df['Age'] = df['Age'].fillna(global_med)# Final Fallback

   # Clean up temporary columns
    df.drop(columns=['income_bracket'], inplace=True)
    
    return df

# function to impute for Strength variable with median within each year
def impute_strength(df):

    df = df.copy()

    # Calculate median strength, grouping by year and drug
    median_strength = df.groupby(['Year', 'Drug'])['Strength'].transform('median')

    # Complete the imputation
    df['Strength'] = df['Strength'].fillna(median_strength)
    
    return df

# function to impute for Day_Supply variable with median within each year
def impute_day_supply(df):

    df = df.copy()

    # Calculate median day supply, grouping by year and drug
    median_day_supply = df.groupby(['Year', 'Drug'])['Day_Supply'].transform('median')

    # Complete the imputation
    df['Day_Supply'] = df['Day_Supply'].fillna(median_day_supply)
    
    return df

# Setup cross validation
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_results          = []
per_drug_recall_folds = []
all_predictions = [] # keeps track of each observation ID

# Loop through folds
for fold, (train_idx, val_idx) in enumerate(sgkf.split(data, y=data[target_variable], groups=data['Person_ID'])):
    # Split the raw data
    train_fold = data.iloc[train_idx].copy()
    val_fold = data.iloc[val_idx].copy()

    # Impute Age within each fold
    train_fold = impute_age(train_fold)
    val_fold = impute_age(val_fold)

    # Impute Strength within each fold
    train_fold = impute_strength(train_fold)
    val_fold = impute_strength(val_fold)

    # Feature selection
    features = numeric_features + categorical_features
    X_train, y_train = train_fold[features + ['Observation_ID']], train_fold[target_variable]
    X_val, y_val = val_fold[features + ['Observation_ID']], val_fold[target_variable]

    # --- TARGET DISTRIBUTION CHECK ---
    train_classes = set(y_train)
    val_classes = set(y_val)
    
    missing_in_train = val_classes - train_classes
    missing_in_val = train_classes - val_classes

    print(f"\n--- Fold {fold+1} Diagnostics ---")
    print(f"Drugs in Test but NOT in Train (Unpredictable): {len(missing_in_train)}")
    print(f"Drugs in Train but NOT in Test: {len(missing_in_val)}")
    
    if missing_in_train:
        # Convert the encoded numbers back to original drug names for readability
        unpredictable_drugs = le.inverse_transform(list(missing_in_train))
        print(f"Example unpredictable drugs in this fold: {list(unpredictable_drugs)[:5]}")
    # ---------------------------------

    # Fit model
    clf = TabICLClassifier(categorical_columns=categorical_features)
    clf.fit(X_train[features], y_train)
    print(f'Fold {fold} training complete.')
    y_pred = clf.predict(X_val[features])

    # Calculate row-level correctness (1 if correct, 0 if wrong)
    is_correct = (y_val == y_pred).astype(int)

    # Get the full classification report as a dictionary for this fold
    # We use target_names=le.classes_ so we can map by drug name directly
    report_dict = classification_report(
        y_val, y_pred, 
        target_names=le.classes_, 
        output_dict=True, 
        zero_division=0
    )

    # Create the observation-level DataFrame
    fold_preds_df = pd.DataFrame({
        'Observation_ID': X_val['Observation_ID'],
        'Actual_Drug': le.inverse_transform(y_val),
        'Predicted_Drug': le.inverse_transform(y_pred),
        'Is_Correct': is_correct,
        'Fold': fold
    })

    # Map class-level performance back to the individual observation
    # This shows the Recall/F1 for the SPECIFIC drug in that row for this fold
    fold_preds_df['Class_Recall_In_Fold'] = fold_preds_df['Actual_Drug'].map(
        lambda x: report_dict[x]['recall'] if x in report_dict else 0
    )
    fold_preds_df['Class_F1_In_Fold'] = fold_preds_df['Actual_Drug'].map(
        lambda x: report_dict[x]['f1-score'] if x in report_dict else 0
    )

    all_predictions.append(fold_preds_df)

    # Assess performance metrics

    # from sklearn
    acc   = accuracy_score(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)
    mcc   = matthews_corrcoef(y_val, y_pred)

     # from permetrics (using macro and micro averaging)
    # Macro: average per class equally — treats rare and common drugs equally.
    # Micro: aggregate all counts — dominated by the most common drugs.
    evaluator = ClassificationMetric(y_val, y_pred)

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

print("ALL FOLDS COMPLETE")

# Export observation-level predictions to back-calculate for each observation ID
full_predictions_df = pd.concat(all_predictions, axis=0)
full_predictions_df.to_csv('tabicl_observation_predictions.csv', index=False)
print('\nObservation-level predictions saved to tabicl_observation_predictions.csv')

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
mean_drug_recall.columns = ['Drug', 'Mean_Recall']
mean_drug_recall = mean_drug_recall.sort_values('Mean_Recall', ascending=False)

print('Top 20 drugs by mean recall:')
print(mean_drug_recall.head(20).to_string(index=False))

print('\nBottom 20 drugs by mean recall:')
print(mean_drug_recall.tail(20).to_string(index=False))

mean_drug_recall.to_csv('tabicl_super_per_drug_recall.csv', index=False)
print('\nPer-drug recall saved to tabicl_super_per_drug_recall.csv')

# Train final model refitted on full data

# impute Age, Strength, Day_Supply
data_final = impute_day_supply(impute_strength(impute_age(data)))

# set final features and target variable
X_final = data_final[features]
y_final = data_final[target_variable]

# shuffle to separate refills
X_final, y_final = shuffle(X_final, y_final, random_state=42)
X_final = X_final.reset_index(drop=True)

# final fit of the model
final_clf = TabICLClassifier(categorical_columns=categorical_features)
final_clf.fit(X_final, y_final)

# Internal validation

# load 2022 data for internal validation dataset
data_2022 = pd.read_csv('super_data_2022.csv')
if 'Unnamed: 0' in data_2022.columns:
    data_2022 = data_2022.drop(columns=['Unnamed: 0'])

# fill missing prescription info for "no prescriptions" to -1
numeric_prescription_cols = ['Quantity', 'Strength', 'Day_Supply']
data_2022[numeric_prescription_cols] = data_2022[numeric_prescription_cols].fillna(-1)
data_2022['Form'] = data_2022['Form'].fillna('-1')

print('2022 data shape:', data_2022.shape)

# Drop any drugs in 2022 not seen during training
unseen = set(data_2022['Drug'].unique()) - set(le.classes_)
print(f'Unseen drugs in 2022 (will be dropped): {unseen}')
data_2022 = data_2022[data_2022['Drug'].isin(le.classes_)].reset_index(drop=True)
print(f'2022 rows after filtering: {len(data_2022):,}')

# Cast feature types
for col in categorical_features:
    data_2022[col] = data_2022[col].astype('category')

# Impute day_supply and age
data_2022 = impute_day_supply(impute_age(data_2022))

# set features and set categorical variables
features = numeric_features + categorical_features
X_2022 = data_2022[features].copy()
for col in categorical_features:
    X_2022[col] = X_2022[col].astype('category')

# encode target variable and predict with final model
y_2022_encoded = le.transform(data_2022['Drug'])
y_pred_2022    = final_clf.predict(X_2022)

# Compute validation metrics

# from sklearn
acc_2022   = accuracy_score(y_2022_encoded, y_pred_2022)
kappa_2022 = cohen_kappa_score(y_2022_encoded, y_pred_2022)
mcc_2022   = matthews_corrcoef(y_2022_encoded, y_pred_2022)

# from permetrics
evaluator_2022    = ClassificationMetric(y_2022_encoded, y_pred_2022)
macro_prec_2022   = evaluator_2022.precision_score(average='macro')
micro_prec_2022   = evaluator_2022.precision_score(average='micro')
macro_recall_2022 = evaluator_2022.recall_score(average='macro')
micro_recall_2022 = evaluator_2022.recall_score(average='micro')
macro_f2_2022     = evaluator_2022.fbeta_score(beta=2, average='macro')
micro_f2_2022     = evaluator_2022.fbeta_score(beta=2, average='micro')

print('\nINTERNAL VALIDATION — MEPS 2022 Results')
print(f'Accuracy:          {acc_2022:.4f}')
print(f'Cohen Kappa:       {kappa_2022:.4f}')
print(f'MCC:               {mcc_2022:.4f}')
print(f'Macro Precision:   {macro_prec_2022:.4f}')
print(f'Micro Precision:   {micro_prec_2022:.4f}')
print(f'Macro Recall:      {macro_recall_2022:.4f}')
print(f'Micro Recall:      {micro_recall_2022:.4f}')
print(f'Macro F2:          {macro_f2_2022:.4f}')
print(f'Micro F2:          {micro_f2_2022:.4f}')

# Save per-drug metrics and overall validation summary for 2022.
report_2022 = classification_report(
    y_2022_encoded, y_pred_2022,
    labels=np.arange(len(le.classes_)),
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)

drug_recall_2022 = pd.DataFrame([
    {
        'Drug': drug,
        'Recall_2022': report_2022[drug]['recall'],
        'Precision_2022': report_2022[drug]['precision'],
        'F1_2022': report_2022[drug]['f1-score'],
        'Support_2022': report_2022[drug]['support']
    }
    for drug in le.classes_ if drug in report_2022
]).sort_values('Recall_2022', ascending=False)

print('Top 15 drugs by recall on 2022 data:')
print(drug_recall_2022.head(15).to_string(index=False))
print('\nBottom 15 drugs by recall on 2022 data:')
print(drug_recall_2022.tail(15).to_string(index=False))

drug_recall_2022.to_csv('tabicl_super_2022_per_drug_metrics.csv', index=False)
print('\nPer-drug 2022 metrics saved to tabicl_super_2022_per_drug_metrics.csv')

validation_summary = pd.DataFrame([{
    'model': 'TabICL',
    'dataset': 'MEPS_2022_internal_validation',
    'accuracy': acc_2022,
    'cohen_kappa': kappa_2022,
    'mcc': mcc_2022,
    'macro_precision': macro_prec_2022,
    'micro_precision': micro_prec_2022,
    'macro_recall': macro_recall_2022,
    'micro_recall': micro_recall_2022,
    'macro_f2': macro_f2_2022,
    'micro_f2': micro_f2_2022,
}])
validation_summary.to_csv('tabicl_super_validation_summary.csv', index=False)
print('Validation summary saved to tabicl_super_validation_summary.csv')

# Observation level predictions

# Create the row-level Correctness indicator
is_correct_2022 = (y_2022_encoded == y_pred_2022).astype(int)

# Build the observation-level DataFrame
# We pull Observation_ID from data_2022 (which was filtered and reset)
obs_2022_df = pd.DataFrame({
    'Observation_ID': data_2022['Observation_ID'],
    'Actual_Drug': le.inverse_transform(y_2022_encoded),
    'Predicted_Drug': le.inverse_transform(y_pred_2022),
    'Is_Correct': is_correct_2022
})

# Map the aggregate Class Metrics back to each specific row
# Using the report_2022 dictionary you already created
obs_2022_df['Class_Recall_Total_2022'] = obs_2022_df['Actual_Drug'].map(
    lambda x: report_2022[x]['recall'] if x in report_2022 else 0
)
obs_2022_df['Class_F1_Total_2022'] = obs_2022_df['Actual_Drug'].map(
    lambda x: report_2022[x]['f1-score'] if x in report_2022 else 0
)

# Export the file
obs_2022_df.to_csv('tabicl_2022_observation_predictions.csv', index=False)

print(f'\nDetailed 2022 observation-level predictions saved to: tabicl_2022_observation_predictions.csv')
print(f'Final count of 2022 observations: {len(obs_2022_df):,}')
