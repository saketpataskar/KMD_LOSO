import os
import numpy as np
import pandas as pd
from scipy.stats import entropy

# ==========================
# CONFIG
# ==========================

HARTH_DIR = r"D:\KMD Project v2/harth"
OUTPUT_CSV = r"D:\KMD Project v2/harth_subject_difficulty.csv"

# ==========================
# HELPERS
# ==========================

def find_activity_column(df):
    """
    Robustly detect the activity / label column.
    """
    for col in df.columns:
        c = col.lower()
        if c in ["activity", "activity_id", "label", "class", "activitylabel"]:
            return col
    raise ValueError(f"No activity column found. Columns = {list(df.columns)}")


def find_sensor_columns(df):
    """
    Detect numeric sensor columns automatically.
    Excludes timestamp-like columns.
    """
    sensor_cols = []
    for col in df.columns:
        if any(k in col.lower() for k in ["acc", "gyro", "gyr"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                sensor_cols.append(col)
    return sensor_cols


def score_bucket(value, q1, q3):
    """
    Convert metric into difficulty score {0,1,2}
    """
    if value < q1:
        return 2
    elif value < q3:
        return 1
    else:
        return 0


def compute_subject_metrics(csv_path):
    df = pd.read_csv(csv_path)

    activity_col = find_activity_column(df)
    sensor_cols = find_sensor_columns(df)

    # Metric 1: Number of samples
    n_samples = len(df)

    # Metric 2: Activity diversity
    activity_counts = df[activity_col].value_counts()
    n_activities = activity_counts.shape[0]

    # Metric 3: Activity entropy (imbalance)
    probs = activity_counts.values / activity_counts.values.sum()
    activity_entropy = entropy(probs)

    # Metric 4: Rare-class participation
    # (rarest 25% of classes across subject)
    n_rare_classes = max(1, int(0.25 * n_activities))
    rare_classes = set(activity_counts.nsmallest(n_rare_classes).index)
    rare_present = sum(activity_counts.index.isin(rare_classes))

    # Metric 5: Signal variability
    if sensor_cols:
        signal_variance = df[sensor_cols].var().mean()
    else:
        signal_variance = 0.0

    return {
        "samples": n_samples,
        "activities": n_activities,
        "entropy": activity_entropy,
        "rare_classes": rare_present,
        "signal_variance": signal_variance
    }


# ==========================
# MAIN
# ==========================

rows = []

print("Scoring HARTH subjects...")

# First pass: collect raw metrics
for fname in sorted(os.listdir(HARTH_DIR)):
    if not fname.endswith(".csv"):
        continue

    subject_id = fname.replace(".csv", "")
    path = os.path.join(HARTH_DIR, fname)

    metrics = compute_subject_metrics(path)
    metrics["subject"] = subject_id
    rows.append(metrics)

df = pd.DataFrame(rows)

# ==========================
# QUANTILE THRESHOLDS
# ==========================

sample_q1, sample_q3 = df["samples"].quantile([0.25, 0.75])
entropy_q1, entropy_q3 = df["entropy"].quantile([0.25, 0.75])
var_q1, var_q3 = df["signal_variance"].quantile([0.25, 0.75])

# ==========================
# DIFFICULTY SCORING
# ==========================

scores = []

for _, r in df.iterrows():
    score = 0

    # Fewer samples → harder
    score += score_bucket(r["samples"], sample_q1, sample_q3)

    # More activities → harder
    if r["activities"] <= 3:
        score += 0
    elif r["activities"] <= 6:
        score += 1
    else:
        score += 2

    # Balanced activity distribution → harder
    score += score_bucket(r["entropy"], entropy_q1, entropy_q3)

    # Rare class presence → harder
    if r["rare_classes"] == 0:
        score += 0
    elif r["rare_classes"] == 1:
        score += 1
    else:
        score += 2

    # High motion variance → harder
    score += score_bucket(r["signal_variance"], var_q1, var_q3)

    scores.append(score)

df["difficulty_score"] = scores


def label_difficulty(score):
    if score <= 3:
        return "easy"
    elif score <= 6:
        return "medium"
    else:
        return "hard"


df["difficulty"] = df["difficulty_score"].apply(label_difficulty)

# ==========================
# SAVE OUTPUT
# ==========================

df_sorted = df.sort_values("difficulty_score")
df_sorted.to_csv(OUTPUT_CSV, index=False)

print("\nSaved HARTH subject difficulty ranking to:")
print(OUTPUT_CSV)

print("\nTop 5 easiest subjects:")
print(df_sorted.head(5)[["subject", "difficulty_score", "difficulty"]])

print("\nTop 5 hardest subjects:")
print(df_sorted.tail(5)[["subject", "difficulty_score", "difficulty"]])