# process_har.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
data_folder = "UCI HAR Dataset"
results_folder = "data_stats/har"

os.makedirs(results_folder, exist_ok=True)
out = Path(results_folder)

# -----------------------------
# Load metadata
# -----------------------------
features = pd.read_csv(
    f"{data_folder}/features.txt",
    sep=r"\s+",
    header=None
)[1].tolist()

activities = pd.read_csv(
    f"{data_folder}/activity_labels.txt",
    sep=r"\s+",
    header=None
)
activity_map = dict(zip(activities[0], activities[1]))

# -----------------------------
# Load train & test splits
# -----------------------------
def load_split(split):
    base = f"{data_folder}/{split}"
    X = pd.read_csv(f"{base}/X_{split}.txt", delim_whitespace=True, header=None)
    y = pd.read_csv(f"{base}/y_{split}.txt", header=None).iloc[:, 0]
    s = pd.read_csv(f"{base}/subject_{split}.txt", header=None).iloc[:, 0]
    return X, y, s

Xtr, ytr, str_ = load_split("train")
Xte, yte, ste = load_split("test")

X = pd.concat([Xtr, Xte], ignore_index=True)
X.columns = features
y = pd.concat([ytr, yte], ignore_index=True)
s = pd.concat([str_, ste], ignore_index=True)

df = pd.concat([s.rename("subject"), y.rename("label"), X], axis=1)
df["activity"] = df["label"].map(activity_map)

# -----------------------------
# Missing values
# -----------------------------
df.isnull().sum().to_csv(out / "missing_values.csv", header=["missing_count"])

# -----------------------------
# Invalid values (simple & comparable)
# -----------------------------
numeric = df[features].apply(pd.to_numeric, errors="coerce")

invalid_summary = {
    "non_numeric": int((numeric.isna() & df[features].notna()).to_numpy().sum()),
    "infinite": int(np.isinf(numeric).to_numpy().sum()),
    "duplicates": int(df.duplicated().to_numpy().sum())
}

pd.Series(invalid_summary).to_csv(out / "invalid_value_summary.csv")

# -----------------------------
# Class distribution
# -----------------------------
class_counts = df["activity"].value_counts().sort_index()
class_percent = class_counts / class_counts.sum() * 100

class_counts.to_csv(out / "class_distribution_counts.csv")
class_percent.to_csv(out / "class_distribution_percent.csv")

plt.figure(figsize=(8, 5))
bars = plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Class Distribution (HAR)")
plt.xlabel("Activity")
plt.ylabel("Count")

for bar, pct in zip(bars, class_percent.values):
    plt.annotate(f"{pct:.1f}%", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                 textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(out / "class_distribution.png", dpi=300)
plt.close()

# -----------------------------
# Subject distribution
# -----------------------------
subject_counts = df["subject"].value_counts().sort_index()
subject_percent = subject_counts / subject_counts.sum() * 100

subject_counts.to_csv(out / "subject_distribution_counts.csv")
subject_percent.to_csv(out / "subject_distribution_percent.csv")

plt.figure(figsize=(10, 5))
bars = plt.bar(subject_counts.index.astype(str), subject_counts.values)
plt.title("Subject Distribution (HAR)")
plt.xlabel("Subject")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(out / "subject_distribution.png", dpi=300)
plt.close()

print("HAR results saved to data_stats/har/")
