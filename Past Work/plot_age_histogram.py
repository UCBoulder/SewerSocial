import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
filename = "nv_synthetic_granular.csv"
try:
    df = pd.read_csv(filename)
    print(f"Data loaded successfully from {filename}")
except FileNotFoundError:
    print(f"Error: Could not find '{filename}'. Make sure you ran 'generate_nv_demographics_granular.py' first!")
    exit()
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.histplot(data=df, x="Age", hue="City", multiple="dodge", bins=20, kde=True, palette="viridis")
plt.title("Age Distribution: Las Vegas vs. Henderson (Granular Model)", fontsize=14)
plt.xlabel("Age (Years)", fontsize=12)
plt.ylabel("Count of Residents", fontsize=12)
output_file = "age_distribution_granular.png"
plt.savefig(output_file, dpi=300)
plt.show()
