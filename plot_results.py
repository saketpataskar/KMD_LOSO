# plot_results.py
import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

# ==========================
# GLOBAL STYLE (Academic / Conference Ready)
# ==========================
def set_plot_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": False
    })

set_plot_style()

# ==========================
# LOAD CONFIG
# ==========================
CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DATASET = cfg["dataset"]["name"]

MODEL_NAME = cfg["paths"]["analysis"]["model_name"]

OUT_ROOT = cfg["paths"]["analysis"]["output_root"]
OUT_DIR = os.path.join(OUT_ROOT, f"analysis_outputs_{MODEL_NAME}_{DATASET.lower()}")


# ==========================
# CLASS NAMES (DATASET-DEPENDENT)
# ==========================
CLASS_NAMES_MAP = {
    "HAR": ["Walk", "WalkUp", "WalkDown", "Sit", "Stand", "Lay"],
    "HARTH": [
        "sitting", "walking", "standing", "cycling (sit)", "lying",
        "running", "shuffling", "transport (sit)",
        "stairs (ascending)", "stairs (descending)",
        "cycling (stand)", "transport (stand)"
    ],
}

if DATASET not in CLASS_NAMES_MAP:
    raise ValueError(f"Unknown dataset: {DATASET}")

CLASS_NAMES = CLASS_NAMES_MAP[DATASET]

# ==========================
# UTILS
# ==========================
def load_aggregated_metrics(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Aggregated metrics JSON not found at {path}. Run analysis_results.py first."
        )
    with open(path, "r") as f:
        return json.load(f)


def load_summary_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Some plots skipped.")
        return rows

    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

# ==========================
# BEAUTIFIED PLOTTING HELPERS
# ==========================
def save_figure(fig, path):
    plt.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)
    print("Saved:", path)


# ==========================
# MEAN MACRO F1 (BAR + ERROR)
# ==========================
def plot_mean_macro_f1(rows, out_dir):
    if not rows:
        return

    methods = [r["method"] for r in rows]
    means = [r["mean_macro_f1"] for r in rows]
    stds = [r["std_macro_f1"] for r in rows]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color="#4C72B0", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Mean Macro F1 by Evaluation Method", weight="bold")

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.text(0.99, 0.01, "Figure: Mean Macro-F1 comparison",
             ha="right", va="bottom", fontsize=9, color="gray")

    save_figure(fig, os.path.join(out_dir, "mean_macro_f1_by_method.png"))


# ==========================
# FOLD DISTRIBUTION (BOXPLOT)
# ==========================
def plot_fold_macro_f1_boxplots(all_methods, out_dir):
    labels, data = [], []

    for name, m in all_methods.items():
        if name == "Baseline":
            continue
        f1s = np.array(m.get("macro_f1s", []), dtype=float)
        if f1s.size:
            labels.append(name)
            data.append(f1s)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)

    ax.set_ylabel("Macro F1")
    ax.set_title("Fold-wise Macro F1 Distribution", weight="bold")

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.text(0.99, 0.01, "Figure: Fold-wise Macro-F1 distribution",
             ha="right", va="bottom", fontsize=9, color="gray")

    save_figure(fig, os.path.join(out_dir, "fold_macro_f1_boxplots.png"))


# ==========================
# PER-CLASS F1 — GROUPED BARS (SECONDARY)
# ==========================
# def plot_per_class_f1_bars(all_methods, rows, out_dir, class_names):
#     methods, per_class_means = [], []
#
#     for r in rows:
#         name = r["method"]
#         if name not in all_methods:
#             continue
#         m = all_methods[name]
#
#         # Baseline case (single vector)
#         if name == "Baseline":
#             pc = m.get("per_class_f1", None)
#             if not isinstance(pc, (list, tuple)) or len(pc) != len(class_names):
#                 print(f"Skipping {name}: invalid per_class_f1 shape")
#                 continue
#             pc = np.array(pc, dtype=float)
#
#         # CV case (list of folds)
#         else:
#             pcs_raw = m.get("per_class_f1s", [])
#             pcs_clean = [
#                 row for row in pcs_raw
#                 if isinstance(row, (list, tuple)) and len(row) == len(class_names)
#             ]
#
#             if not pcs_clean:
#                 print(f"Skipping {name}: no valid per_class_f1s rows")
#                 continue
#
#             pcs = np.array(pcs_clean, dtype=float)
#             pc = pcs.mean(axis=0)
#
#         methods.append(name)
#         per_class_means.append(pc)
#
#     if not per_class_means:
#         print("No valid per-class F1 data found for bar plot.")
#         return
#
#     per_class_means = np.stack(per_class_means)
#     M, C = per_class_means.shape
#
#     x = np.arange(C)
#     width = 0.8 / M
#
#     fig, ax = plt.subplots(figsize=(max(10, C), 5))
#     colors = mpl.colormaps["tab10"](np.linspace(0, 1, M))
#
#     for i, method in enumerate(methods):
#         ax.bar(x + i * width, per_class_means[i],
#                width=width, label=method, color=colors[i])
#
#     ax.set_xticks(x + width * (M - 1) / 2)
#     ax.set_xticklabels(class_names, rotation=30, ha="right")
#
#     ax.set_ylabel("Per-Class F1")
#     ax.set_title("Per-Class F1 by Evaluation Method (Bar View)", weight="bold")
#
#     ax.legend(ncol=2, frameon=False)
#     ax.yaxis.grid(True, linestyle="--", alpha=0.3)
#
#     for spine in ["top", "right"]:
#         ax.spines[spine].set_visible(False)
#
#     fig.text(0.99, 0.01, "Figure: Per-class F1 comparison (bar chart)",
#              ha="right", va="bottom", fontsize=9, color="gray")
#
#     save_figure(fig, os.path.join(out_dir, "per_class_f1_by_method_bars.png"))
#
#
# def plot_negative_macro_f1_vs_baseline_vertical(rows, out_dir, baseline_name="Baseline"):
#     if not rows:
#         print("No summary rows, skipping negative bar plot.")
#         return
#
#     # Convert rows to dict
#     row_dict = {r["method"]: r for r in rows}
#
#     if baseline_name not in row_dict:
#         print(f"Baseline method '{baseline_name}' not found.")
#         return
#
#     baseline_mean = row_dict[baseline_name]["mean_macro_f1"]
#
#     methods = []
#     negative_deltas = []
#
#     for r in rows:
#         if r["method"] == baseline_name:
#             continue
#         methods.append(r["method"])
#         negative_deltas.append(r["mean_macro_f1"] - baseline_mean)
#
#     x = np.arange(len(methods))
#
#     plt.figure(figsize=(8, 5))
#     plt.bar(x, negative_deltas)
#
#     # Baseline reference line (x-axis)
#     plt.axhline(0, linestyle="--", linewidth=1)
#
#     plt.xticks(x, methods, rotation=25, ha="right")
#     plt.ylabel("Δ Macro F1 vs Baseline", fontsize=11)
#     plt.title("Performance Relative to Baseline (Negative Bar Chart)", fontsize=13)
#
#     plt.grid(axis="y", linestyle="--", alpha=0.5)
#     plt.tight_layout()
#
#     out_path = os.path.join(out_dir, "negative_macro_f1_vs_baseline_vertical.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#
#     print("Saved:", out_path)

def plot_negative_macro_f1_vs_baseline_vertical(
    rows,
    out_dir,
    baseline_name="Baseline"
):
    if not rows:
        print("No summary rows, skipping negative bar plot.")
        return

    # Convert rows to dict
    row_dict = {r["method"]: r for r in rows}

    if baseline_name not in row_dict:
        print(f"Baseline method '{baseline_name}' not found.")
        return

    baseline_mean = row_dict[baseline_name]["mean_macro_f1"]

    methods = []
    percent_deltas = []

    for r in rows:
        if r["method"] == baseline_name:
            continue

        delta = r["mean_macro_f1"] - baseline_mean
        percent_delta = (delta / baseline_mean) * 100

        methods.append(r["method"])
        percent_deltas.append(percent_delta)

    percent_deltas = np.array(percent_deltas)
    x = np.arange(len(methods))

    # -----------------------------
    # Colors: Blue = Positive, Red = Negative
    # -----------------------------
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in percent_deltas]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(x, percent_deltas, color=colors)

    # Baseline reference line (0%)
    plt.axhline(0, linestyle="--", linewidth=1, color="black")

    # -----------------------------
    # Dynamic symmetric Y-axis
    # -----------------------------
    # -----------------------------
    # Nice symmetric Y-axis scaling
    # -----------------------------
    max_abs = np.max(np.abs(percent_deltas))

    # Round up to next "nice" number (multiple of 5)
    nice_max = np.ceil(max_abs / 10) * 10

    # Add small buffer
    nice_max += 5

    plt.ylim(-nice_max, nice_max)

    # -----------------------------
    # Annotations
    # -----------------------------
    for i, val in enumerate(percent_deltas):
        label = f"{val:+.1f}%"

        offset = max_abs * 0.05
        y = val + offset if val >= 0 else val - offset

        plt.text(
            i,
            y,
            label,
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9
        )

    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Macro F1 Improvement vs Baseline (%)", fontsize=11)
    plt.title("Performance Relative to Baseline", fontsize=13)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        "macro_f1_percentage_vs_baseline_vertical.png"
    )

    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


#early stopping epochs related - boxstrip graph
def plot_early_stopping_boxplot(all_methods, out_dir):
    labels = []
    data = []

    for name, m in all_methods.items():
        es = m.get("early_stop_epochs", [])
        if not es:
            continue

        es_arr = np.array(es, dtype=float)
        if es_arr.size == 0:
            continue

        labels.append(name)
        data.append(es_arr)

    if not data:
        print("No early stopping data found, skipping boxplot.")
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True
    )
    plt.ylabel("Early Stopping Epoch", fontsize=11)
    plt.title("Early Stopping Epoch Distribution by Method", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "early_stopping_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)



# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MODELS = ["CNN", "LSTM", "CNNLSTM", "InceptionTime"]
CV_METHODS = ["KFold", "Stratified", "GroupKFold", "LOSO"]

BAR_COLORS = {
    "KFold": "#4C72B0",
    "Stratified": "#55A868",
    "GroupKFold": "#C44E52",
    "LOSO": "#FFD04C",
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def load_baseline_macro_f1(aggregated_metrics_path):
    with open(aggregated_metrics_path, "r") as f:
        data = json.load(f)
    return data["Baseline"]["macro_f1"]


def load_cv_means(cv_summary_csv):
    df = pd.read_csv(cv_summary_csv)
    return dict(zip(df["method"], df["mean_macro_f1"]))


# ---------------------------------------------------------
# MAIN PLOT FUNCTION
# ---------------------------------------------------------
def plot_grouped_macro_f1_vs_baseline(
    dataset_name,
    results_root,
    out_dir
):


    # model -> cv -> percent delta
    delta_matrix = {
        model: {} for model in MODELS
    }

    # -----------------------------------------------------
    # Collect data
    # -----------------------------------------------------
    for model in MODELS:
        analysis_dir = os.path.join(
            results_root,
            f"analysis_outputs_{model}_{dataset_name.lower()}"
        )

        agg_path = os.path.join(analysis_dir, "aggregated_metrics.json")
        cv_path = os.path.join(analysis_dir, "cv_methods_summary.csv")

        if not os.path.exists(agg_path) or not os.path.exists(cv_path):
            print(f"[WARN] Missing files for {model} ({dataset_name}), skipping.")
            continue

        baseline_f1 = load_baseline_macro_f1(agg_path)
        cv_means = load_cv_means(cv_path)

        for cv in CV_METHODS:
            if cv not in cv_means:
                continue

            delta = cv_means[cv] - baseline_f1
            percent_delta = (delta / baseline_f1) * 100
            delta_matrix[model][cv] = percent_delta

    # -----------------------------------------------------
    # Prepare plotting arrays
    # -----------------------------------------------------
    n_models = len(MODELS)
    n_cv = len(CV_METHODS)

    x = np.arange(n_models)
    bar_width = 0.18

    # Find global y-limit
    all_vals = [
        v for model_vals in delta_matrix.values()
        for v in model_vals.values()
    ]
    max_abs = max(abs(v) for v in all_vals)
    y_lim = np.ceil(max_abs / 5) * 5 + 5

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.figure(figsize=(11, 6))

    for i, cv in enumerate(CV_METHODS):
        values = [
            delta_matrix[model].get(cv, 0.0)
            for model in MODELS
        ]

        positions = x + (i - (n_cv - 1) / 2) * bar_width

        bars = plt.bar(
            positions,
            values,
            bar_width,
            label=cv,
            color=BAR_COLORS[cv]
        )

        # Annotate bars
        for px, val in zip(positions, values):
            plt.text(
                px,
                val + (0.04 * y_lim if val >= 0 else -0.04 * y_lim),
                f"{val:+.1f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8
            )

    # -----------------------------------------------------
    # Styling
    # -----------------------------------------------------
    plt.axhline(0, linestyle="--", color="black", linewidth=1)

    plt.xticks(x, MODELS)
    plt.ylabel("Macro F1 Improvement vs Baseline (%)", fontsize=11)
    plt.title(
        f"Cross-Validation Performance vs Baseline ({dataset_name})",
        fontsize=13
    )

    plt.ylim(-y_lim, y_lim)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="CV Method", frameon=False)

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"grouped_macro_f1_vs_baseline_{dataset_name.lower()}.png"
    )

    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_per_class_f1_heatmap(all_methods, rows, out_dir, class_names):
    methods, matrix = [], []

    for r in rows:
        name = r["method"]
        if name not in all_methods:
            continue
        m = all_methods[name]

        # Baseline
        if name == "Baseline":
            pc = m.get("per_class_f1", None)
            if not isinstance(pc, (list, tuple)) or len(pc) != len(class_names):
                print(f"Skipping {name}: invalid per_class_f1 shape")
                continue
            pc = np.array(pc, dtype=float)

        # CV methods
        else:
            pcs_raw = m.get("per_class_f1s", [])
            pcs_clean = [
                row for row in pcs_raw
                if isinstance(row, (list, tuple)) and len(row) == len(class_names)
            ]

            if not pcs_clean:
                print(f"Skipping {name}: no valid per_class_f1s rows")
                continue

            pcs = np.array(pcs_clean, dtype=float)
            pc = pcs.mean(axis=0)

        methods.append(name)
        matrix.append(pc)

    if not matrix:
        print("No valid per-class F1 data found for heatmap.")
        return

    data = np.stack(matrix)

    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), 5))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)

    ax.set_title("Per-Class F1 Heatmap (Methods × Classes)", weight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Evaluation Method")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center",
                color="white" if data[i, j] < 0.5 else "black",
                fontsize=8
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("F1 Score")

    fig.text(
        0.99, 0.01,
        "Figure: Per-class F1 heatmap (rows: methods, columns: classes)",
        ha="right", va="bottom", fontsize=9, color="gray"
    )

    save_figure(fig, os.path.join(out_dir, "per_class_f1_heatmap.png"))



# ==========================
# CONFUSION MATRIX
# ==========================
def plot_confusion_matrix(cm, classes, title, out_path, normalize=True):
    cm = np.array(cm, dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    ax.set_title(title, weight="bold")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    thresh = cm.max() * 0.65
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fig.text(0.99, 0.01,
             f"Figure: {title}",
             ha="right", va="bottom", fontsize=9, color="gray")

    save_figure(fig, out_path)



def mean_per_class_f1(per_class_f1s):
    """
    Handles variable-length per-class F1 lists (e.g. LOSO with missing classes)
    by padding with NaN and computing nanmean.
    """

    max_len = max(len(fold) for fold in per_class_f1s)

    padded = []
    for fold in per_class_f1s:
        padded_fold = list(fold) + [np.nan] * (max_len - len(fold))
        padded.append(padded_fold)

    return np.nanmean(np.array(padded), axis=0)

def generate_rq1_per_class_excel(
    dataset_name,
    results_root,
    out_dir,
    class_names=None
):
    """
    Creates an Excel table comparing per-class F1 scores
    grouped by ML model and CV method.
    """

    wb = Workbook()
    ws = wb.active
    ws.title = f"{dataset_name} Per-Class F1"

    # ---------------------------
    # Header rows
    # ---------------------------
    title = f"{dataset_name} Dataset Performance – Per Class F1 Comparison"
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=14)
    ws.cell(row=1, column=1, value=title).font = Font(bold=True)
    ws.cell(row=1, column=1).alignment = Alignment(horizontal="center")

    ws.cell(row=2, column=1, value="ML Model").font = Font(bold=True)
    ws.cell(row=2, column=2, value="CV Type").font = Font(bold=True)

    if class_names is None:
        # Default placeholder names
        class_names = [f"Class {i+1}" for i in range(12)]

    for i, cls in enumerate(class_names):
        ws.cell(row=2, column=3 + i, value=cls).font = Font(bold=True)

    # ---------------------------
    # Data rows
    # ---------------------------
    current_row = 3

    for model in MODELS:
        analysis_dir = os.path.join(
            results_root,
            f"analysis_outputs_{model}_{dataset_name.lower()}"
        )

        agg_path = os.path.join(analysis_dir, "aggregated_metrics.json")
        if not os.path.exists(agg_path):
            print(f"[WARN] Missing {agg_path}, skipping {model}")
            continue

        with open(agg_path, "r") as f:
            metrics = json.load(f)

        model_start_row = current_row

        for cv in CV_METHODS:
            if cv not in metrics:
                continue

            per_class = mean_per_class_f1(metrics[cv]["per_class_f1s"])

            ws.cell(row=current_row, column=2, value=cv)

            for i, val in enumerate(per_class):
                ws.cell(
                    row=current_row,
                    column=3 + i,
                    value=round(float(val), 4)
                )

            current_row += 1

        # Merge ML Model cells vertically
        ws.merge_cells(
            start_row=model_start_row,
            start_column=1,
            end_row=current_row - 1,
            end_column=1
        )
        ws.cell(row=model_start_row, column=1, value=model)
        ws.cell(row=model_start_row, column=1).alignment = Alignment(
            vertical="center",
            horizontal="center"
        )

    # ---------------------------
    # Formatting
    # ---------------------------
    for col in range(1, 3 + len(class_names)):
        ws.column_dimensions[get_column_letter(col)].width = 16

    ws.freeze_panes = "C3"

    # ---------------------------
    # Save
    # ---------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"{dataset_name}_Per_Class_F1_Comparison.xlsx"
    )

    wb.save(out_path)
    print("Saved:", out_path)

def plot_early_stopping_cv_comparison(all_methods, out_dir):
    """
    Comparative boxplot of early stopping epochs:
    - One box per CV method
    - Baseline excluded
    - Shows distribution across folds
    """

    labels = []
    data = []

    for name, m in all_methods.items():
        # Exclude Baseline explicitly
        if name.lower() == "baseline":
            continue

        es = m.get("early_stop_epochs", [])
        if not es:
            continue

        es_arr = np.array(es, dtype=float)
        if es_arr.size == 0:
            continue

        labels.append(name)
        data.append(es_arr)

    if not data:
        print("No CV early stopping data found, skipping comparison plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#90dbf4", alpha=0.85),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Early Stopping Epoch", fontsize=11)
    plt.title("Early Stopping Comparison Across CV Methods", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "early_stopping_cv_comparison_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

def plot_model_cv_macro_f1_boxplots(all_methods, out_dir):
    """
    Plot NON-AGGREGATED Macro F1 boxplots.
    Each CV method has one box per model–CV configuration.
    Baseline is ignored.
    """

    cv_order = ["KFold", "Stratified", "GroupKFold", "LOSO"]

    positions = []
    labels = []
    box_data = []

    pos = 1
    gap = 1.5

    for cv in cv_order:
        if cv not in all_methods:
            continue

        f1s = all_methods[cv].get("macro_f1s", [])
        if not f1s:
            continue

        for run_f1 in f1s:
            box_data.append([float(run_f1)])  # one box = one configuration
            positions.append(pos)
            labels.append(cv)
            pos += 1

        pos += gap

    if not box_data:
        print("No model–CV Macro F1 data found, skipping detailed boxplot.")
        return

    plt.figure(figsize=(12, 5))
    plt.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        showmeans=True,
        manage_ticks=False,
    )

    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.ylabel("Macro F1", fontsize=11)
    plt.xlabel("Cross-Validation Method", fontsize=11)
    plt.title(
        "Macro F1 per Model–CV Configuration (No Aggregation, Baseline Excluded)",
        fontsize=13,
    )
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "macro_f1_model_cv_boxplots.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)



CV_COLORS = {
    "KFold": "#4C72B0",
    "Stratified": "#55A868",
    "GroupKFold": "#C44E52",
    "LOSO": "#FFD04C",
}


def plot_grouped_early_stopping_boxplots(
    dataset_name,
    results_root,
    out_dir
):

    dataset_key = dataset_name.lower()   # ← FIXED ONCE

    fig, ax = plt.subplots(figsize=(12, 6))

    box_data = []
    box_positions = []
    box_colors = []

    group_spacing = 1.6
    box_width = 0.18
    x_group_centers = []
    current_x = 1

    for model in MODELS:
        analysis_dir = os.path.join(
            results_root,
            f"analysis_outputs_{model}_{dataset_key}"
        )

        agg_path = os.path.join(analysis_dir, "aggregated_metrics.json")

        if not os.path.exists(agg_path):
            print(f"[WARN] Missing {agg_path}, skipping {model}")
            continue

        with open(agg_path, "r") as f:
            metrics = json.load(f)

        model_start_x = current_x
        valid_boxes = 0

        for i, cv in enumerate(CV_METHODS):
            epochs = metrics.get(cv, {}).get("early_stop_epochs", [])
            if not epochs:
                continue

            box_data.append(np.array(epochs))
            box_positions.append(current_x + valid_boxes * box_width)
            box_colors.append(CV_COLORS[cv])
            valid_boxes += 1

        if valid_boxes > 0:
            model_end_x = current_x + (valid_boxes - 1) * box_width
            x_group_centers.append((model_start_x + model_end_x) / 2)
            current_x += group_spacing

    # ---------------- SAFETY CHECK ----------------
    if not box_data:
        raise RuntimeError(
            "No early stopping data found. "
            "Check results_root and dataset naming."
        )

    if len(box_data) != len(box_positions):
        raise RuntimeError(
            f"Internal error: {len(box_data)} datasets vs "
            f"{len(box_positions)} positions"
        )

    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(x_group_centers)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Early Stopping Epoch")
    ax.set_title(
        f"Early Stopping Epoch Distribution ({dataset_name})",
        weight="bold"
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=CV_COLORS[cv], lw=6)
        for cv in CV_METHODS
    ]
    ax.legend(
        legend_handles,
        CV_METHODS,
        title="CV Method",
        frameon=False
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"grouped_early_stopping_epochs_{dataset_key}.png"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

# ==========================
# MAIN
# ==========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary_csv = os.path.join(OUT_DIR, "cv_methods_summary.csv")
    aggregated_json = os.path.join(OUT_DIR, "aggregated_metrics.json")

    rows = load_summary_rows(summary_csv)
    all_methods = load_aggregated_metrics(aggregated_json)

    #plot_mean_macro_f1(rows, OUT_DIR)
    #plot_early_stopping_boxplot(all_methods, OUT_DIR)

    # Secondary view (bars)
    #plot_per_class_f1_bars(all_methods, rows, OUT_DIR, CLASS_NAMES)
    # plot_negative_macro_f1_vs_baseline_vertical(rows, OUT_DIR, baseline_name="Baseline")
    #
    # plot_early_stopping_boxplot(all_methods, rows, OUT_DIR)
    #
    # # Main research figure (heatmap)
    # plot_per_class_f1_heatmap(all_methods, rows, OUT_DIR, CLASS_NAMES)
    # plot_grouped_macro_f1_vs_baseline(
    #     dataset_name="HAR",
    #     results_root=OUT_ROOT,
    #     out_dir="C:/KMD/KMD_LOSO/results_rq2"
    # )
    #plot_early_stopping_cv_comparison(all_methods, OUT_DIR)

    # plot_model_cv_macro_f1_boxplots(all_methods, OUT_DIR)
    # plot_fold_macro_f1_boxplots(all_methods, OUT_DIR)

    plot_grouped_early_stopping_boxplots(
        dataset_name="HAR",
        results_root=OUT_ROOT,
        out_dir="C:/KMD/KMD_LOSO/results_rq3"
    )


    # generate_rq1_per_class_excel(
    #     dataset_name="HARTH",
    #     results_root=OUT_ROOT,
    #     out_dir="C:/KMD/KMD_LOSO/results_rq1_harth"
    # )

    print("\nAll plots generated into:", OUT_DIR)


if __name__ == "__main__":
    main()
