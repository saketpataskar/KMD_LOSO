# plot_results.py
import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# LOAD CONFIG
# ==========================

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DATASET = cfg["dataset"]["name"]              # HAR | HARTH
MODEL_NAME = cfg["analysis"]["model_name"]

OUT_ROOT = cfg["analysis"]["output_root"][DATASET]
OUT_DIR = os.path.join(OUT_ROOT, f"analysis_outputs_{MODEL_NAME}_{DATASET.lower()}")

# ==========================
# CLASS NAMES (DATASET-DEPENDENT)
# ==========================

CLASS_NAMES_MAP = {
    "HAR": ["Walk", "WalkUp", "WalkDown", "Sit", "Stand", "Lay"],
    "HARTH": [
        "sitting",
        "walking",
        "standing",
        "cycling (sit)",
        "lying",
        "running",
        "shuffling",
        "transport (sit)",
        "stairs (ascending)",
        "stairs (descending)",
        "cycling (stand)",
        "transport (stand)",
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

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        rows = df.to_dict(orient="records")
    except ImportError:
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "n_folds" in r and r["n_folds"] != "":
                    r["n_folds"] = int(r["n_folds"])
                for k in [
                    "mean_macro_f1", "std_macro_f1",
                    "mean_accuracy", "std_accuracy",
                    "mean_early_stop_epoch", "std_early_stop_epoch"
                ]:
                    r[k] = float(r[k]) if r.get(k) not in ("", None, "nan") else None
                rows.append(r)
    return rows

# ==========================
# PLOTTING HELPERS
# ==========================

def plot_mean_macro_f1(rows, out_dir):
    if not rows:
        print("No summary rows, skipping mean macro F1 plot.")
        return

    methods = [r["method"] for r in rows]
    means = [r["mean_macro_f1"] for r in rows]
    stds = [r["std_macro_f1"] for r in rows]

    x = np.arange(len(methods))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Macro F1", fontsize=11)
    plt.title("Mean Macro F1 by Evaluation Method", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "mean_macro_f1_by_method.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_fold_macro_f1_boxplots(all_methods, out_dir):
    labels = []
    data = []

    for name, m in all_methods.items():
        if name == "Baseline":
            continue
        f1s = m.get("macro_f1s", None)
        if f1s is None:
            continue
        f1s = np.array(f1s, dtype=float)
        if f1s.size == 0:
            continue
        labels.append(name)
        data.append(f1s)

    if not data:
        print("No fold macro-F1 data found, skipping boxplot.")
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Macro F1", fontsize=11)
    plt.title("Fold-wise Macro F1 Distribution", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fold_macro_f1_boxplots.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_per_class_f1(all_methods, rows, out_dir, class_names):
    if not rows:
        print("No summary rows, skipping per-class F1 plot.")
        return

    print("\n=== DEBUG: per_class_f1s fold lengths ===")
    for method, m in all_methods.items():
        pcs = m.get("per_class_f1s", [])
        if isinstance(pcs, list):
            print(method, "->", [len(x) if isinstance(x, list) else "INVALID" for x in pcs])

    methods = []
    per_class_means = []

    for r in rows:
        name = r["method"]
        if name not in all_methods:
            continue
        m = all_methods[name]

        if name == "Baseline":
            if "per_class_f1" not in m:
                continue
            pc = np.array(m["per_class_f1"], dtype=float)

        else:
            pcs_raw = m.get("per_class_f1s", [])

            pcs_clean = [
                row for row in pcs_raw
                if isinstance(row, (list, tuple)) and len(row) == len(class_names)
            ]

            if not pcs_clean:
                print(f"Skipping {name}: invalid per_class_f1s shapes -> {[len(r) for r in pcs_raw]}")
                continue

            pcs = np.array(pcs_clean, dtype=float)
            pc = pcs.mean(axis=0)

        methods.append(name)
        per_class_means.append(pc)

    if not per_class_means:
        print("No valid per-class F1 data found, skipping.")
        return

    per_class_means = np.stack(per_class_means, axis=0)
    M, C = per_class_means.shape
    x = np.arange(C)
    width = 0.8 / max(M, 1)

    plt.figure(figsize=(max(10, C), 5))
    for i, method in enumerate(methods):
        plt.bar(x + i * width, per_class_means[i], width=width, label=method)
    plt.xticks(x + width * (M - 1) / 2, class_names, rotation=25, ha="right")
    plt.ylabel("Per-Class F1", fontsize=11)
    plt.title("Per-Class F1 by Evaluation Method", fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "per_class_f1_by_method.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


# ==========================
# FIXED CONFUSION MATRIX (NO % SIGN)
# ==========================

def plot_confusion_matrix(cm, classes, title, out_path, normalize=True):
    cm = np.array(cm, dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        cm = cm / row_sums

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title(title, fontsize=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() * 0.65
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]

            # FIX: show clean decimal (no percentage sign)
            txt = f"{val:.2f}"

            plt.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_confusion_matrices_selected(all_methods, out_dir, class_names,
                                     methods_to_plot=None):
    if methods_to_plot is None:
        methods_to_plot = ["Baseline", "KFold", "Stratified", "GroupKFold", "LOSO"]

    for name in methods_to_plot:
        if name not in all_methods:
            print(f"{name} not found, skipping CM.")
            continue

        m = all_methods[name]

        if name == "Baseline":
            if "confusion_matrix" not in m:
                print(f"No confusion_matrix for {name}, skipping.")
                continue
            cm = np.array(m["confusion_matrix"], dtype=float)

        else:
            cms = np.array(m.get("confusion_mats", []), dtype=float)
            if cms.ndim != 3 or cms.shape[0] == 0:
                print(f"No confusion_mats for {name}, skipping.")
                continue
            cm = cms.mean(axis=0)

        out_path = os.path.join(out_dir, f"confusion_{name}.png")
        plot_confusion_matrix(
            cm,
            class_names,
            title=f"{name} – Confusion Matrix (normalized)",
            out_path=out_path,
            normalize=True,
        )


def plot_loso_subject_f1(all_methods, out_dir):
    if "LOSO" not in all_methods:
        print("LOSO not found, skipping LOSO subject-wise plot.")
        return

    m = all_methods["LOSO"]
    f1s = np.array(m.get("macro_f1s", []), dtype=float)
    if f1s.size == 0:
        print("No LOSO macro-F1 data, skipping.")
        return

    x = np.arange(len(f1s))
    plt.figure(figsize=(10, 4))
    plt.bar(x, f1s)
    plt.xlabel("LOSO Fold (subject index)", fontsize=11)
    plt.ylabel("Macro F1", fontsize=11)
    plt.title("LOSO: Subject-wise Macro F1", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "loso_subject_macro_f1.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_early_stopping(all_methods, rows, out_dir):
    methods = []
    means = []
    stds = []

    for r in rows:
        name = r["method"]
        if name not in all_methods:
            continue
        es = all_methods[name].get("early_stop_epochs", [])
        if not es:
            continue

        es_arr = np.array(es, dtype=float)
        methods.append(name)
        means.append(es_arr.mean())
        stds.append(es_arr.std())

    if not methods:
        print("No early stopping data, skipping.")
        return

    x = np.arange(len(methods))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Early Stopping Epoch", fontsize=11)
    plt.title("Early Stopping Behavior by Method", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "early_stopping_epochs_by_method.png")
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

    plot_mean_macro_f1(rows, OUT_DIR)
    plot_fold_macro_f1_boxplots(all_methods, OUT_DIR)
    plot_per_class_f1(all_methods, rows, OUT_DIR, CLASS_NAMES)
    plot_confusion_matrices_selected(all_methods, OUT_DIR, CLASS_NAMES)
    plot_loso_subject_f1(all_methods, OUT_DIR)
    plot_early_stopping(all_methods, rows, OUT_DIR)

    print("\nAll plots generated into:", OUT_DIR)


if __name__ == "__main__":
    main()
