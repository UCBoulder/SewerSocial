# Randomly sample from community-level probability output to generate predictions for
# facilities of the correct population size.

# import necessary libraries
import pandas as pd
import os
import gc

# =============================================================================
# 1. FACILITY CONFIGURATION
# name:     file structure for output file
# parent:   must match the city name prefix used in the city-level output files
# facility: label used in the output filename
# sample_n: number of rows to sample (same seed as original inference scripts)
# =============================================================================
FACILITY_DATASETS = [
    {"name": "clark_county_nv_campus", "parent": "clark_county_nv",     "facility": "campus", "sample_n": 3750},
    {"name": "clark_county_nv_f1",     "parent": "clark_county_nv",     "facility": "f1",     "sample_n": 872009},
    {"name": "clark_county_nv_f2",     "parent": "clark_county_nv",     "facility": "f2",     "sample_n": 86330},
    {"name": "clark_county_nv_f3",     "parent": "clark_county_nv",     "facility": "f3",     "sample_n": 757418},
    {"name": "clark_county_nv_f4",     "parent": "clark_county_nv",     "facility": "f4",     "sample_n": 248509},
    {"name": "clark_county_nv_f4a",    "parent": "clark_county_nv",     "facility": "f4a",    "sample_n": 133977},
    {"name": "clark_county_nv_f4b",    "parent": "clark_county_nv",     "facility": "f4b",    "sample_n": 114532},
    {"name": "clark_county_nv_f5",     "parent": "clark_county_nv",     "facility": "f5",     "sample_n": 255008},
    {"name": "clark_county_nv_f6",     "parent": "clark_county_nv",     "facility": "f6",     "sample_n": 16399},
    {"name": "urbana_champaign_il_f1", "parent": "urbana_champaign_il", "facility": "f1",     "sample_n": 109634},
    {"name": "sandwich_ma_f1",         "parent": "sandwich_ma",         "facility": "f1",     "sample_n": 976},
]

MODELS = ["xgboost", "knn", "svm", "tabicl", "realmlp"]

# =============================================================================
# 2. SLICE FACILITIES FROM CITY FILES
# Reads each city file once per model and writes all its facility subsets,
# then deletes the city file from memory before moving to the next model.
# =============================================================================
def generate_all_facility_subsets():
    # Group facilities by parent city so each city file is only read once per model
    cities = {}
    for f in FACILITY_DATASETS:
        cities.setdefault(f["parent"], []).append(f)

    for model_key in MODELS:
        print(f"\n{'='*65}")
        print(f"MODEL: {model_key.upper()}")
        print(f"{'='*65}")

        for city_name, facilities in cities.items():
            city_file = f"{city_name}_{model_key}_threshold_results.csv.gz"

            if not os.path.exists(city_file):
                print(f"  WARNING: {city_file} not found — skipping {city_name} facilities for {model_key}.")
                continue

            print(f"\n  Reading {city_file}...")
            city_proba = pd.read_csv(city_file, compression='gzip')
            city_proba['Observation_ID'] = city_proba['Observation_ID'].astype(str)
            print(f"  Loaded {len(city_proba):,} rows.")

            for facility in facilities:
                actual_n       = min(facility["sample_n"], len(city_proba))
                facility_proba = city_proba.sample(n=actual_n, random_state=42).reset_index(drop=True)
                out_file       = f"{facility['name']}_{model_key}_threshold_results.csv.gz"
                facility_proba.to_csv(out_file, index=False, compression='gzip')
                print(f"    {facility['name']}: {len(facility_proba):,} rows → {out_file}")

            del city_proba
            gc.collect()

    print("\n" + "="*65)
    print("ALL FACILITY SUBSETS COMPLETE")
    print("="*65)
    print("\nFacility-level outputs:")
    for facility in FACILITY_DATASETS:
        for model_key in MODELS:
            print(f"  {facility['name']}_{model_key}_threshold_results.csv.gz")


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    generate_all_facility_subsets()
