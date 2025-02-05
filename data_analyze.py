import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
from scipy import stats
from scipy.stats import ks_2samp

# 1.1
t1_path = './Data/t1_user_active_min.csv'

df_t1 = pd.read_csv(t1_path)
df_t1.info()
df_t1.head()
print(df_t1['active_mins'].describe())
df_t1["dt"] = pd.to_datetime(df_t1["dt"])
print(f"Date time range from: {df_t1['dt'].min()} to {df_t1['dt'].max()}")

# 1.2
t2_path = './Data/t2_user_variant.csv'

df_t2 = pd.read_csv(t2_path)
df_t2.info()
df_t2.head()

print(df_t2.nunique())

# 1.3
t3_path = './Data/t3_user_active_min_pre.csv'

df_t3 = pd.read_csv(t3_path)
df_t3.info()
df_t3.head()
print(df_t3['active_mins'].describe())
print(df_t3.nunique())
df_t3["dt"] = pd.to_datetime(df_t3["dt"])
print(f"Date time range from: {df_t3['dt'].min()} to {df_t3['dt'].max()}")

# 1.4
t4_path = "./Data/t4_user_attributes.csv"
df_t4 = pd.read_csv(t4_path)
df_t4.info()
df_t4.head()
print(df_t4.nunique())
print(df_t4['gender'].unique())
print(df_t4['user_type'].unique())

# 1.5 it is txt file. 

# Part 2
df_merged = df_t1.merge(df_t2[['uid', 'variant_number']], on="uid", how="left")
df_aggregated = df_merged.groupby(['uid', 'variant_number'])['active_mins'].sum().reset_index()
df_aggregated.rename(columns={'active_mins': 'total_active_mins'}, inplace=True)
df_aggregated.to_csv('./EDA/t1_t2_aggregated.csv', index=False)
group_stats = df_aggregated.groupby('variant_number')['total_active_mins'].describe()
print(group_stats)
control_group = df_aggregated[df_aggregated['variant_number'] == 0]['total_active_mins']
treatment_group = df_aggregated[df_aggregated['variant_number'] == 1]['total_active_mins']
print(f"Control Group (variant_number = 0): {len(control_group)} users")
print(f"Treatment Group (variant_number = 1): {len(treatment_group)} users")
print("Control Group Summary:")
print(control_group.describe())
print("Treatment Group Summary:")
print(treatment_group.describe())

# Part 3
t_stat, p_value = ttest_ind(control_group, treatment_group, equal_var=False)

print("\nT-test Results:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Part 4
## Use histogram and Q-Q plot to see normality
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(control_group, kde=True, bins=30, color='blue', ax=axes[0, 0])
axes[0, 0].set_title("Histogram of Active Minutes (Control Group)")
axes[0, 0].set_xlabel("Total Active Minutes")
axes[0, 0].set_ylabel("Frequency")

sns.histplot(treatment_group, kde=True, bins=30, color='orange', ax=axes[0, 1])
axes[0, 1].set_title("Histogram of Active Minutes (Treatment Group)")
axes[0, 1].set_xlabel("Total Active Minutes")
axes[0, 1].set_ylabel("Frequency")

stats.probplot(control_group, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot of Control Group")

stats.probplot(treatment_group, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot of Treatment Group")

plt.tight_layout()
plt.savefig('./images/part4_histograms_qq_plots.png', dpi=300)

## Use ks test for checking normaility
ks_control = ks_2samp(control_group, np.random.normal(np.mean(control_group), np.std(control_group), len(control_group)))
ks_treatment = ks_2samp(treatment_group, np.random.normal(np.mean(treatment_group), np.std(treatment_group), len(treatment_group)))
print(f"KS Test for control group: {ks_control} for treatment group: {ks_treatment}")

## Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=control_group, showfliers=True)
plt.ylabel("Total Active Minutes")
plt.title("Control Group Box Plot of Active Minutes")
plt.savefig('./images/part4_box_plot_control.png')

plt.figure(figsize=(10, 6))
sns.boxplot(data=treatment_group, showfliers=True)
plt.ylabel("Total Active Minutes")
plt.title("Treatment Group Box Plot of Active Minutes")
plt.savefig('./images/part4_box_plot_treatment.png')

## Remove outliers. 
df_t1 = pd.read_csv("./Data/t1_user_active_min.csv")
print(f"t1 min: {df_t1['active_mins'].min()}, t1 max: {df_t1['active_mins'].max()}")
df_t2 = pd.read_csv("./Data/t2_user_variant.csv")

df_t1_cleaned = df_t1[df_t1["active_mins"] <= 1440]

df_merged_cleaned = df_t1_cleaned.merge(df_t2[['uid', 'variant_number']], on="uid", how="left")

df_aggregated_cleaned = df_merged_cleaned.groupby(['uid', 'variant_number'])['active_mins'].sum().reset_index()
df_aggregated_cleaned.rename(columns={'active_mins': 'total_active_mins'}, inplace=True)
df_aggregated_cleaned.to_csv('./EDA/t1_t2_aggregated_cleaned.csv', index=False)

control_group_cleaned = df_aggregated_cleaned[df_aggregated_cleaned['variant_number'] == 0]['total_active_mins']
treatment_group_cleaned = df_aggregated_cleaned[df_aggregated_cleaned['variant_number'] == 1]['total_active_mins']

cleaned_group_stats = pd.DataFrame({
    "Group": ["Control", "Treatment"],
    "Mean Active Minutes": [control_group_cleaned.mean(), treatment_group_cleaned.mean()],
    "Median Active Minutes": [control_group_cleaned.median(), treatment_group_cleaned.median()],
    "Standard Deviation": [control_group_cleaned.std(), treatment_group_cleaned.std()],
    "Count": [len(control_group_cleaned), len(treatment_group_cleaned)]
})

t_stat_cleaned, p_value_cleaned = ttest_ind(control_group_cleaned, treatment_group_cleaned, equal_var=False)

print("Cleaned Group Stats::")
print(cleaned_group_stats)

print("\nCleaned T-test Results:")
print(f"T-statistic: {t_stat_cleaned}, P-value: {p_value_cleaned}")