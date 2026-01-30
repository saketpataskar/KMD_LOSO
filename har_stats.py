# process_har.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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
# Advanced Pretty Plot Helper
# -----------------------------
def pretty_barplot(labels, values, title, xlabel, ylabel, save_path, percent=None, y_max=None):
    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(labels))
    max_val = max(values)

    base_color = "#4C72B0"
    highlight_color = "#DD8452"

    bars = ax.bar(x, values, width=0.65, color=base_color, zorder=3)

    # Highlight dominant bar
    max_idx = np.argmax(values)
    bars[max_idx].set_color(highlight_color)

    # Soft shadow (depth effect)
    for i, v in enumerate(values):
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
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")

    # Optional fixed Y-axis limit (useful for subject distributions)
    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Value callouts
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if h > 0.05 * max_val:
            txt = f"{percent[i]:.1f}%" if percent is not None else f"{int(h)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max_val * 0.015,
                txt,
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.85
                ),
                zorder=5
            )

    # -----------------------------
    # Figure-level caption (dynamic, no overlap)
    # -----------------------------
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
# Class distribution
# -----------------------------
class_counts = df["activity"].value_counts().sort_index()
class_percent = class_counts / class_counts.sum() * 100

class_counts.to_csv(out / "class_distribution_counts.csv")
class_percent.to_csv(out / "class_distribution_percent.csv")

pretty_barplot(
    class_counts.index.astype(str).values,
    class_counts.values,
    "Class Distribution (UCI HAR Dataset)",
    "Activity",
    "Sample Count",
    out / "class_distribution.png",
    percent=class_percent.values
)

# -----------------------------
# Subject distribution
# -----------------------------
subject_counts = df["subject"].value_counts().sort_index()
subject_percent = subject_counts / subject_counts.sum() * 100

subject_counts.to_csv(out / "subject_distribution_counts.csv")
subject_percent.to_csv(out / "subject_distribution_percent.csv")

pretty_barplot(
    subject_counts.index.astype(str).values,
    subject_counts.values,
    "Subject Distribution (UCI HAR Dataset)",
    "Subject ID",
    "Sample Count",
    out / "subject_distribution.png",
    y_max=450
)

print("HAR results saved to data_stats/har/")
