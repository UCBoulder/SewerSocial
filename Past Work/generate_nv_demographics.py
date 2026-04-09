import pandas as pd
import numpy as np

# Note on 15-24: This group (14.2%) is split between Child (15-17) and Adult (18-24).
clark_county_dist = {
    # Child Sub-Buckets
    '0-1':  0.010, 
    '1-4':  0.042, 
    '5-14': 0.121, 
    '15-17': 0.043, # (Calculated share of the 15-24 bracket)

    # Adult Sub-Buckets
    '18-24': 0.099, # (Calculated share of the 15-24 bracket)
    '25-34': 0.143, 
    '35-44': 0.134, 
    '45-54': 0.135, 
    '55-64': 0.120,

    # Senior Sub-Buckets
    '65-74': 0.091, 
    '75-84': 0.047, 
    '85+':   0.014
}

def get_granular_age(broad_bucket):
    """
    Given a broad bucket (e.g., '18-64'), this picks a specific age 
    based on the real-world Clark County distribution curve.
    """
    if broad_bucket == '0-17':
        # Probability weights extracted from your screenshot for kids
        sub_buckets = ['0-1', '1-4', '5-14', '15-17']
        probs = [clark_county_dist['0-1'], clark_county_dist['1-4'], 
                 clark_county_dist['5-14'], clark_county_dist['15-17']]
        
    elif broad_bucket == '18-64':
        # Probability weights for adults (Shows the "Bulge" in the 20s/30s)
        sub_buckets = ['18-24', '25-34', '35-44', '45-54', '55-64']
        probs = [clark_county_dist['18-24'], clark_county_dist['25-34'], 
                 clark_county_dist['35-44'], clark_county_dist['45-54'], 
                 clark_county_dist['55-64']]
        
    elif broad_bucket == '65+':
        # Probability weights for seniors (Tapers off with age)
        sub_buckets = ['65-74', '75-84', '85+']
        probs = [clark_county_dist['65-74'], clark_county_dist['75-84'], 
                 clark_county_dist['85+']]
    
    probs = np.array(probs)
    probs /= probs.sum()
    
    # 1. Pick the sub-bucket (e.g., "25-34")
    choice = np.random.choice(sub_buckets, p=probs)
    
    # 2. Pick exact age uniformly within that small bucket
    if choice == '85+':
        return np.random.randint(85, 99)
    elif choice == '0-1':
        return 0
    else:
        # Extract range "25-34" -> 25, 34
        start, end = map(int, choice.split('-'))
        return np.random.randint(start, end + 1)

def generate_city_data_granular(city_name, n_samples, race_probs, age_probs):
    data = []
    
    for _ in range(n_samples):
        # Gender (50/50)
        gender = np.random.choice(['Male', 'Female'], p=[0.5, 0.5])
        
        # Race (City Specific)
        race = np.random.choice(list(race_probs.keys()), p=list(race_probs.values()))
        
        # Age Generation (Hierarchical)
        # Step 1: Pick Broad Group (Child/Adult/Senior) using City Data (Henderson vs Las Vegas)
        broad_group = np.random.choice(list(age_probs.keys()), p=list(age_probs.values()))
        
        # Step 2: Pick Specific Age using Clark County "Shape"
        age = get_granular_age(broad_group)
        
        # Assign Label
        if age < 18: group = "Under 18"
        elif age < 65: group = "Adult (18-64)"
        else: group = "Senior (65+)"
        
        data.append([city_name, gender, race, age, group])
        
    return pd.DataFrame(data, columns=['City', 'Gender', 'Race_Ethnicity', 'Age', 'Age_Group'])

np.random.seed(42)

# City Parameters (Census 2024 Estimates)
lv_race = {'White (Non-Hispanic)': 0.408, 'Hispanic': 0.341, 'Black': 0.119, 'Asian': 0.069, 'Other': 0.063}
lv_age_broad = {'0-17': 0.228, '18-64': 0.616, '65+': 0.156}

hen_race = {'White (Non-Hispanic)': 0.589, 'Hispanic': 0.182, 'Black': 0.062, 'Asian': 0.095, 'Other': 0.072}
hen_age_broad = {'0-17': 0.214, '18-64': 0.582, '65+': 0.204}


df_lv = generate_city_data_granular("Las Vegas", 1000, lv_race, lv_age_broad)
df_hen = generate_city_data_granular("Henderson", 1000, hen_race, hen_age_broad)

final_df = pd.concat([df_lv, df_hen])
final_df.to_csv("nv_synthetic_granular.csv", index=False)

print(final_df.groupby('City')['Age'].describe())
