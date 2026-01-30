import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
data_folder = "harth"   # relative path to dataset folder
results_folder = "data_stats/harth"        # where CSVs & plots are saved
label_col = "label"               # HARTH label column
subject_col = "subject"           # added subject identifier column
plot_top_k_subjects = None        # set to an int to plot only top-K subjects

os.makedirs(results_folder, exist_ok=True)

# -----------------------------
# 1. Load CSVs & combine them
# -----------------------------
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in '{data_folder}'.")

df_list = []
for file in csv_files:
    tmp = pd.read_csv(file)
    subj_id = os.path.basename(file).split(".")[0]
    tmp[subject_col] = subj_id
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)
print("Loaded dataframe shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Missing values
# -----------------------------
missing_values = df.isnull().sum()
total_missing = missing_values.sum()

print("\nMissing values per column:\n", missing_values)
print("Total missing:", total_missing)

# Save to CSV
missing_values.to_csv(os.path.join(results_folder, "missing_values.csv"), header=["missing_count"])

# =====================================================================
# 3. CHECK FOR INCORRECT / INVALID VALUES
# =====================================================================

numeric_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']

# 3.1 Non-numeric value detection
invalid_numeric = {}
for col in numeric_cols:
    invalid = df[col].apply(lambda x: isinstance(x, str)).sum()
    invalid_numeric[col] = invalid

print("\nNon-numeric values in numeric sensor columns:")
print(invalid_numeric)

# 3.2 Out-of-range accelerometer values (HARTH uses ±16g)
invalid_range = {}
for col in numeric_cols:
    invalid_low = df[df[col] < -16].shape[0]
    invalid_high = df[df[col] > 16].shape[0]
    invalid_range[col] = invalid_low + invalid_high

print("\nOut-of-range accelerometer values (outside ±16g):")
print(invalid_range)

# 3.3 Duplicate rows
duplicate_rows = df.duplicated().sum()
print("\nNumber of duplicated rows:", duplicate_rows)

# 3.4 Invalid timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
invalid_timestamps = df['timestamp'].isna().sum()

print("\nInvalid timestamp values:", invalid_timestamps)

# Non-monotonic timestamps per subject
timestamp_violations = {}
for subj in df["subject"].unique():
    sub_df = df[df["subject"] == subj]
    non_mono = (sub_df['timestamp'].diff() < pd.Timedelta(0)).sum()
    timestamp_violations[subj] = non_mono

print("\nNon-monotonic timestamps per subject:")
print(timestamp_violations)

# 3.5 Save incorrect value summary
invalid_summary = pd.DataFrame({
    "non_numeric": invalid_numeric,
    "out_of_range": invalid_range
})
invalid_summary.to_csv(os.path.join(results_folder, "invalid_value_summary.csv"))

print("\nIncorrect value check complete. Summary saved to invalid_value_summary.csv")

# -----------------------------
# 4. Class distribution
# -----------------------------
if label_col not in df.columns:
    raise KeyError(f"Label column '{label_col}' not found.")

class_counts = df[label_col].value_counts().sort_index()
class_percent = (df[label_col].value_counts(normalize=True).sort_index() * 100)

print("\nClass counts:\n", class_counts)
print("\nClass percentages (%):\n", class_percent.round(2))

# Save to CSV
class_counts.to_csv(os.path.join(results_folder, "class_distribution_counts.csv"), header=["count"])
class_percent.to_csv(os.path.join(results_folder, "class_distribution_percent.csv"), header=["percent"])

# Plot: Class imbalance
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index.astype(str), class_counts.values)
plt.xlabel("Class label")
plt.ylabel("Count")
plt.title("Class Distribution")

for bar, pct in zip(bars, class_percent.values):
    plt.annotate(f"{pct:.1f}%", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                 textcoords="offset points", xytext=(0, 3),
                 ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "class_distribution.png"), dpi=300)
plt.close()

# -----------------------------
# 5. Subject distribution
# -----------------------------
subject_counts = df[subject_col].value_counts().sort_index()
subject_percent = (df[subject_col].value_counts(normalize=True).sort_index() * 100)

print("\nSubject counts:\n", subject_counts)
print("\nSubject percent (%):\n", subject_percent.round(2))

# Save to CSV
subject_counts.to_csv(os.path.join(results_folder, "subject_distribution_counts.csv"), header=["count"])
subject_percent.to_csv(os.path.join(results_folder, "subject_distribution_percent.csv"), header=["percent"])

# Plot: subject imbalance
if plot_top_k_subjects is not None:
    to_plot = subject_counts.nlargest(plot_top_k_subjects)
    title_suffix = f"(Top {plot_top_k_subjects})"
else:
    to_plot = subject_counts
    title_suffix = ""

plt.figure(figsize=(14, 6))
bars = plt.bar(to_plot.index.astype(str), to_plot.values)
plt.xlabel("Subject")
plt.ylabel("Count")
plt.title(f"Subject Distribution {title_suffix}")
plt.xticks(rotation=45, ha="right")

if len(to_plot) <= 50:
    for bar in bars:
        plt.annotate(f"{bar.get_height():.0f}",
                     (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "subject_distribution.png"), dpi=300)
plt.close()

print("\nAll CSV files and plots have been saved into:", results_folder)