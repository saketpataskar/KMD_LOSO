import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

# -----------------------------
# Plot Styling (Academic / PPT Ready)
# -----------------------------
def set_plot_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": False,
        "figure.dpi": 120,
        "savefig.dpi": 300
    })

set_plot_style()

# -----------------------------
# Configuration
# -----------------------------
data_folder = "harth"
results_folder = "data_stats/harth"
label_col = "label"
subject_col = "subject"
DATASET_NAME = "HARTH Dataset"

# Optional: limit subjects shown
plot_top_k_subjects = None

# Label map file (citation-safe)
LABEL_MAP_FILE = os.path.join(data_folder, "activity_map.csv")

os.makedirs(results_folder, exist_ok=True)
out = Path(results_folder)

# -----------------------------
# Load Activity Label Map (CSV or Fallback)
# -----------------------------
if os.path.exists(LABEL_MAP_FILE):
    label_map_df = pd.read_csv(LABEL_MAP_FILE)
    ACTIVITY_MAP = dict(zip(label_map_df["label"], label_map_df["name"]))
    print("Loaded activity labels from activity_map.csv")
else:
    print("activity_map.csv not found — using built-in HARTH label map")

    ACTIVITY_MAP = {
        1: "walking",
        2: "running",
        3: "shuffling",
        4: "transport (sit)",
        5: "stairs (ascending)",
        6: "standing",
        7: "sitting",
        8: "cycling (sit)",
        13: "lying",
        14: "stairs (descending)",
        130: "cycling stand",
        140: "transport stand"
    }

# -----------------------------
def pretty_barplot(
    labels,
    values,
    title,
    xlabel,
    ylabel,
    save_path,
    percent=None,
    y_max=None,
    y_scale_power=None
):
    fig, ax = plt.subplots(figsize=(11, 5))

    values = np.array(values, dtype=float)

    # -----------------------------
    # Apply scaling if requested
    # -----------------------------
    scale_factor = 1
    if y_scale_power is not None:
        scale_factor = 10 ** y_scale_power

    scaled_values = values / scale_factor
    max_val = max(scaled_values)

    x = np.arange(len(labels))

    base_color = "#4C72B0"
    highlight_color = "#DD8452"

    bars = ax.bar(x, scaled_values, width=0.65, color=base_color, zorder=3)

    # Highlight dominant bar
    max_idx = np.argmax(scaled_values)
    bars[max_idx].set_color(highlight_color)

    # Soft shadow
    for i, v in enumerate(scaled_values):
        ax.bar(x[i] + 0.03, v, width=0.65, color="black", alpha=0.06, zorder=2)

    # Grid & spines
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)

    # Titles & labels
    ax.set_title(title, fontsize=16, weight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=14)

    if y_scale_power:
        ax.set_ylabel(f"{ylabel} (×10^{y_scale_power})", fontsize=12, labelpad=10)
    else:
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")

    # Auto-scale Y-axis
    if y_max is None:
        ax.set_ylim(0, max_val * 1.20)
    else:
        ax.set_ylim(0, y_max / scale_factor)

    # -----------------------------
    # Labels above bars
    # -----------------------------
    for i, bar in enumerate(bars):
        h = bar.get_height()
        txt = f"{percent[i]:.1f}%" if percent is not None else f"{h:.1f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max_val * 0.02,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
            color="black",
            zorder=5
        )

    # Caption
    fig.text(
        0.99, 0.01,
        f"Figure: {title}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="gray"
    )

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace(".png", ".pdf"))
    plt.close()

# -----------------------------
# 1. Load CSVs
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

# -----------------------------
# 2. Missing values
# -----------------------------
missing_values = df.isnull().sum()
missing_values.to_csv(out / "missing_values.csv", header=["missing_count"])

# -----------------------------
# 3. Invalid values
# -----------------------------
numeric_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

invalid_summary = {
    "non_numeric": int((numeric.isna() & df[numeric_cols].notna()).to_numpy().sum()),
    "out_of_range": int(((numeric < -16) | (numeric > 16)).to_numpy().sum()),
    "duplicates": int(df.duplicated().sum())
}

pd.Series(invalid_summary).to_csv(out / "invalid_value_summary.csv")

# -----------------------------
# Timestamp checks
# -----------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

invalid_timestamps = int(df['timestamp'].isna().sum())
pd.Series({"invalid_timestamps": invalid_timestamps}).to_csv(out / "invalid_timestamps.csv")

timestamp_violations = {}
for subj in df[subject_col].unique():
    sub_df = df[df[subject_col] == subj]
    non_mono = int((sub_df['timestamp'].diff() < pd.Timedelta(0)).sum())
    timestamp_violations[subj] = non_mono

pd.Series(timestamp_violations).to_csv(out / "timestamp_violations.csv")

# -----------------------------
# 4. Class distribution
# -----------------------------
class_counts = df[label_col].value_counts().sort_index()
class_percent = class_counts / class_counts.sum() * 100

class_counts.to_csv(out / "class_distribution_samples.csv", header=["samples"])
class_percent.to_csv(out / "class_distribution_percent.csv", header=["percent"])

class_labels_named = [
    ACTIVITY_MAP.get(lbl, f"Class {lbl}")
    for lbl in class_counts.index
]

pretty_barplot(
    np.array(class_labels_named),
    class_counts.values,
    f"Class Distribution ({DATASET_NAME})",
    "Activity",
    "Sample Count",
    out / "class_distribution.png",
    percent=class_percent.values,
    y_scale_power=6   # ← shows (×10⁶) in label
)

# -----------------------------
# 5. Subject distribution
# -----------------------------
subject_counts = df[subject_col].value_counts().sort_index()
subject_percent = subject_counts / subject_counts.sum() * 100

subject_counts.to_csv(out / "subject_distribution_counts.csv", header=["samples"])
subject_percent.to_csv(out / "subject_distribution_percent.csv", header=["percent"])

if plot_top_k_subjects:
    subject_counts = subject_counts.nlargest(plot_top_k_subjects)
    subject_percent = subject_percent.loc[subject_counts.index]
    title_suffix = f"(Top {plot_top_k_subjects})"
else:
    title_suffix = ""

pretty_barplot(
    subject_counts.index.astype(str).values,
    subject_counts.values,
    f"Subject Distribution ({DATASET_NAME}) {title_suffix}",
    "Subject ID",
    "Sample Count",
    out / "subject_distribution.png",
    y_scale_power=5   # ← shows (×10⁵) in label
)

print("\nHARTH results saved to:", results_folder)
print("Class distribution uses SAMPLE COUNTS (HAR-style).")
