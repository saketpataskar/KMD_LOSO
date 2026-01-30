import numpy as np
import pandas as pd
from scipy.stats import entropy

# ==========================
# CONFIG
# ==========================

NPZ_PATH = r"D:\KMD Project v2/ucihar_raw_merged.npz"
OUTPUT_CSV = r"D:\KMD Project v2/har_subject_difficulty.csv"

RARE_CLASS_THRESHOLD = 0.05   # <5% windows = rare class

# ==========================
# LOAD NPZ
# ==========================

print("Loading NPZ:", NPZ_PATH)
data = np.load(NPZ_PATH, allow_pickle=True)

X = data["X"]          # (N, T, C) or (N, C, T)
y = data["y"]          # (N,)
subjects = data["subjects"]  # (N,)

assert len(y) == len(subjects), "Mismatch in NPZ arrays"

# Ensure numeric
y = y.astype(int)
subjects = subjects.astype(int)

# ==========================
# SUBJECT METRICS
# ==========================

rows = []

for sid in np.unique(subjects):
    idx = subjects == sid

    y_s = y[idx]
    X_s = X[idx]

    n_windows = len(y_s)

    # Activity stats
    counts = pd.Series(y_s).value_counts()
    probs = counts / n_windows

    class_entropy = entropy(probs)
    n_classes = counts.size

    # Rare-class ratio
    rare_ratio = probs[probs < RARE_CLASS_THRESHOLD].sum()

    # Signal variance (window-level)
    if X_s.ndim == 3:
        signal_variance = X_s.var(axis=(0, 1)).mean()
    else:
        signal_variance = 0.0

    rows.append({
        "subject": sid,
        "n_windows": n_windows,
        "n_activities": n_classes,
        "class_entropy": class_entropy,
        "rare_class_ratio": rare_ratio,
        "signal_variance": signal_variance
    })

df = pd.DataFrame(rows)

# ==========================
# DIFFICULTY SCORE
# ==========================

def normalize(col):
    return (col - col.min()) / (col.max() - col.min() + 1e-8)

df["inv_windows"] = 1.0 / df["n_windows"]

df["difficulty_score"] = (
    0.30 * normalize(df["class_entropy"]) +
    0.25 * normalize(df["n_activities"]) +
    0.20 * normalize(df["rare_class_ratio"]) +
    0.15 * normalize(df["inv_windows"]) +
    0.10 * normalize(df["signal_variance"])
)

# Difficulty labels
def label(score):
    if score <= df["difficulty_score"].quantile(0.33):
        return "easy"
    elif score <= df["difficulty_score"].quantile(0.66):
        return "medium"
    else:
        return "hard"

df["difficulty"] = df["difficulty_score"].apply(label)

# ==========================
# SAVE
# ==========================

df = df.sort_values("difficulty_score", ascending=False)
df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved HAR subject difficulty to:")
print(OUTPUT_CSV)

print("\nHardest subjects:")
print(df.head(5)[["subject", "difficulty_score", "difficulty"]])

print("\nEasiest subjects:")
print(df.tail(5)[["subject", "difficulty_score", "difficulty"]])