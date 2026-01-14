# analysis_results.py
import os
import json
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ==========================
# CONFIG – EDIT THESE PATHS
# ==========================

# CNN baseline directory
# (for now this should contain the CNN baseline: metrics.json, history.npz, best_model.pt)
# BASELINE_DIR = r"D:/DKE/KMD/KMD_LOSO/baseline_results"
#
# # Root where ALL CV summaries and fold folders live
# # e.g. contains: GroupKFold_CNN_summary.json, LOSO_CNN_summary.Rjson, etc.
# RESULTS_ROOT = r"D:/DKE/KMD/KMD_LOSO/results"
#
# # Which CV methods you want to analyze for CNN.
# # It’s fine if some summaries don’t exist yet – they’ll just be skipped.
#
#
# # Output directory for summary CSV + aggregated JSON
# OUT_DIR = r"D:/DKE/KMD/KMD_LOSO/analysis_outputs"

BASELINE_DIR = r"D:/DKE/KMD/KMD_LOSO/baseline_results_LSTM"
RESULTS_ROOT = r"D:/DKE/KMD/KMD_LOSO/results"
OUT_DIR = r"D:/DKE/KMD/KMD_LOSO/analysis_outputs_LSTM"
MODEL_NAME = "LSTM"
METHOD_NAMES = ["KFold", "Stratified", "GroupKFold", "LOSO"]

# ==========================
# UTILS
# ==========================

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_baseline_metrics(baseline_dir: str):
    """
    Load baseline (no CV) metrics for the CNN model.

    Expects:
      baseline_dir/metrics.json
      baseline_dir/history.npz   (for early stopping epoch, optional)
    """
    metrics_path = os.path.join(baseline_dir, "metrics.json")
    hist_path = os.path.join(baseline_dir, "history.npz")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Baseline metrics.json not found at {metrics_path}")

    m = load_json(metrics_path)

    best_epoch = None
    if os.path.exists(hist_path):
        hist = np.load(hist_path)
        if "val_f1" in hist:
            val_f1 = hist["val_f1"]
            best_epoch = int(np.argmax(val_f1) + 1)

    out = {
        "name": "Baseline",
        "macro_f1": float(m["macro_f1"]),
        "accuracy": float(m["accuracy"]),
        "per_class_f1": np.array(m["per_class_f1"]).tolist(),
        "confusion_matrix": np.array(m["confusion_matrix"]).tolist(),
        "early_stop_epochs": [best_epoch] if best_epoch is not None else [],
    }
    return out


def load_cv_method(method_name: str, results_root: str, model_name: str = "LSTM"):
    """
    Load one CV method (for CNN) from results_root.

    Expects in RESULTS_ROOT:
      <method_name>_<model_name>_summary.json
      <cv_type>_<model>_fold0/history.npz
      <cv_type>_<model>_fold0/metrics.json (already aggregated in summary)

    Example filenames for CNN:
      GroupKFold_CNN_summary.json
      GroupKFold_CNN_fold0/
      GroupKFold_CNN_fold1/
      ...
    """
    summary_name = f"{method_name}_{model_name}_summary.json"
    summary_path = os.path.join(results_root, summary_name)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary JSON for {method_name} (model={model_name}) not found at {summary_path}"
        )

    summary = load_json(summary_path)
    folds = summary["folds"]
    n_folds = summary["n_folds"]

    macro_f1s = []
    accuracies = []
    per_class_f1s = []
    confusion_mats = []
    early_epochs = []

    cv_type = summary["cv_type"]   # e.g. "GroupKFold"
    model = summary["model"]       # should be "CNN"

    for fold_idx, fold_metrics in enumerate(folds):
        macro_f1s.append(fold_metrics["macro_f1"])
        accuracies.append(fold_metrics["accuracy"])
        per_class_f1s.append(fold_metrics["per_class_f1"])
        confusion_mats.append(fold_metrics["confusion_matrix"])

        # get early stopping epoch from history.npz if present
        fold_dir = os.path.join(results_root, f"{cv_type}_{model}_fold{fold_idx}")
        hist_path = os.path.join(fold_dir, "history.npz")
        if os.path.exists(hist_path):
            hist = np.load(hist_path)
            if "val_f1" in hist:
                val_f1 = hist["val_f1"]
                best_epoch = int(np.argmax(val_f1) + 1)
                early_epochs.append(best_epoch)

    out = {
        "name": method_name,
        "n_folds": n_folds,
        "macro_f1s": macro_f1s,
        "accuracies": accuracies,
        "per_class_f1s": per_class_f1s,
        "confusion_mats": confusion_mats,
        "early_stop_epochs": early_epochs,
    }
    return out


def build_summary_table(all_methods: dict):
    """
    all_methods: dict name -> metrics dict

    Returns: list of summary rows for CSV.
    """
    rows = []
    for name, m in all_methods.items():
        if name == "Baseline":
            mean_f1 = float(m["macro_f1"])
            std_f1 = 0.0
            mean_acc = float(m["accuracy"])
            std_acc = 0.0
            n_folds = 1
            es = m["early_stop_epochs"]
            mean_es = float(np.mean(es)) if es else None
            std_es = float(np.std(es)) if es else None
        else:
            f1s = np.array(m["macro_f1s"], dtype=float)
            accs = np.array(m["accuracies"], dtype=float)
            n_folds = m["n_folds"]
            mean_f1 = float(f1s.mean())
            std_f1 = float(f1s.std())
            mean_acc = float(accs.mean())
            std_acc = float(accs.std())
            es_arr = np.array(m["early_stop_epochs"], dtype=float)
            mean_es = float(es_arr.mean()) if es_arr.size > 0 else None
            std_es = float(es_arr.std()) if es_arr.size > 0 else None

        rows.append({
            "method": name,
            "n_folds": n_folds,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_early_stop_epoch": mean_es,
            "std_early_stop_epoch": std_es,
        })
    return rows


def save_summary_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        print("No rows to save.")
        return

    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
    else:
        import csv
        keys = rows[0].keys()
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved summary CSV to {out_path}")


def save_aggregated_json(all_methods, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_methods, f, indent=2)
    print(f"Saved aggregated metrics JSON to {out_path}")


# ==========================
# MAIN
# ==========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_methods = {}

    # 1) Baseline (CNN)
    print("Loading baseline metrics (CNN) from:", BASELINE_DIR)
    baseline = load_baseline_metrics(BASELINE_DIR)
    all_methods["Baseline"] = baseline

    # 2) CV methods (CNN)
    for method_name in METHOD_NAMES:
        try:
            print(f"Loading CV summary for {method_name} (CNN) from:", RESULTS_ROOT)
            m = load_cv_method(method_name, RESULTS_ROOT, model_name="CNN")
            all_methods[method_name] = m
        except FileNotFoundError as e:
            # It’s okay if some methods (e.g. KFold, LOSO) haven’t been run yet.
            print("  ->", e)

    # 3) Build and save summary table
    rows = build_summary_table(all_methods)
    summary_csv_path = os.path.join(OUT_DIR, "cv_methods_summary.csv")
    save_summary_csv(rows, summary_csv_path)

    # 4) Save full aggregated metrics for plotting
    aggregated_json_path = os.path.join(OUT_DIR, "aggregated_metrics.json")
    save_aggregated_json(all_methods, aggregated_json_path)

    # 5) Quick printout
    print("\n=== CNN Summary (mean macro-F1, accuracy) ===")
    for r in rows:
        f1 = r["mean_macro_f1"]
        f1s = r["std_macro_f1"]
        acc = r["mean_accuracy"]
        accs = r["std_accuracy"]
        print(f"{r['method']:>10} | F1={f1:.3f}±{f1s:.3f} | Acc={acc:.3f}±{accs:.3f}")


if __name__ == "__main__":
    main()
