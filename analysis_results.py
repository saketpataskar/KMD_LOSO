# analysis_results.py
import os
import json
import yaml
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ==========================
# UTILS
# ==========================

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_baseline_metrics(baseline_dir: str):
    metrics_path = os.path.join(baseline_dir, "metrics.json")
    hist_path = os.path.join(baseline_dir, "history.npz")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Baseline metrics.json not found at {metrics_path}")

    m = load_json(metrics_path)

    best_epoch = None
    if os.path.exists(hist_path):
        hist = np.load(hist_path)
        if "val_f1" in hist:
            best_epoch = int(np.argmax(hist["val_f1"]) + 1)

    return {
        "name": "Baseline",
        "macro_f1": float(m["macro_f1"]),
        "accuracy": float(m["accuracy"]),
        "per_class_f1": np.array(m["per_class_f1"]).tolist(),
        "confusion_matrix": np.array(m["confusion_matrix"]).tolist(),
        "early_stop_epochs": [best_epoch] if best_epoch is not None else [],
    }


def load_cv_method(method_name: str, results_root: str, model_name: str):
    summary_name = f"{method_name}_{model_name}_summary.json"
    summary_path = os.path.join(results_root, summary_name)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    summary = load_json(summary_path)

    macro_f1s, accuracies, per_class_f1s, confusion_mats, early_epochs = [], [], [], [], []

    cv_type = summary["cv_type"]
    model = summary["model"]

    for fold_idx, fold_metrics in enumerate(summary["folds"]):
        macro_f1s.append(fold_metrics["macro_f1"])
        accuracies.append(fold_metrics["accuracy"])
        per_class_f1s.append(fold_metrics["per_class_f1"])
        confusion_mats.append(fold_metrics["confusion_matrix"])

        fold_dir = os.path.join(results_root, f"{cv_type}_{model}_fold{fold_idx}")
        hist_path = os.path.join(fold_dir, "history.npz")
        if os.path.exists(hist_path):
            hist = np.load(hist_path)
            if "val_f1" in hist:
                early_epochs.append(int(np.argmax(hist["val_f1"]) + 1))

    return {
        "name": method_name,
        "n_folds": summary["n_folds"],
        "macro_f1s": macro_f1s,
        "accuracies": accuracies,
        "per_class_f1s": per_class_f1s,
        "confusion_mats": confusion_mats,
        "early_stop_epochs": early_epochs,
    }


def build_summary_table(all_methods: dict):
    rows = []
    for name, m in all_methods.items():
        if name == "Baseline":
            rows.append({
                "method": name,
                "n_folds": 1,
                "mean_macro_f1": m["macro_f1"],
                "std_macro_f1": 0.0,
                "mean_accuracy": m["accuracy"],
                "std_accuracy": 0.0,
                "mean_early_stop_epoch": np.mean(m["early_stop_epochs"]) if m["early_stop_epochs"] else None,
                "std_early_stop_epoch": np.std(m["early_stop_epochs"]) if m["early_stop_epochs"] else None,
            })
        else:
            f1s = np.array(m["macro_f1s"], float)
            accs = np.array(m["accuracies"], float)
            es = np.array(m["early_stop_epochs"], float)

            rows.append({
                "method": name,
                "n_folds": m["n_folds"],
                "mean_macro_f1": f1s.mean(),
                "std_macro_f1": f1s.std(),
                "mean_accuracy": accs.mean(),
                "std_accuracy": accs.std(),
                "mean_early_stop_epoch": es.mean() if es.size else None,
                "std_early_stop_epoch": es.std() if es.size else None,
            })
    return rows


def save_summary_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if HAS_PANDAS:
        pd.DataFrame(rows).to_csv(out_path, index=False)
    else:
        import csv
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def save_aggregated_json(all_methods, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_methods, f, indent=2)


# ==========================
# MAIN
# ==========================

def main():
    cfg = load_config("config.yaml")

    dataset = cfg["dataset"]["name"]
    model_name = cfg["analysis"]["model_name"]
    methods = cfg["analysis"]["methods"]

    results_root = cfg["paths"]["results_root"][dataset]
    baseline_root = cfg["paths"]["baseline_root"][dataset]
    out_root = cfg["analysis"]["output_root"][dataset]

    out_dir = os.path.join(out_root, f"{model_name}_{dataset.lower()}")
    baseline_dir = os.path.join(baseline_root, f"baseline_results_{model_name}_{dataset.lower()}")

    os.makedirs(out_dir, exist_ok=True)

    all_methods = {}

    print(f"Loading baseline metrics {model_name} ({dataset})")
    all_methods["Baseline"] = load_baseline_metrics(baseline_dir)

    for method_name in methods:
        try:
            print(f"Loading CV {method_name} ({model_name})")
            all_methods[method_name] = load_cv_method(
                method_name,
                results_root,
                model_name=model_name
            )
        except FileNotFoundError as e:
            print("  ->", e)

    rows = build_summary_table(all_methods)
    save_summary_csv(rows, os.path.join(out_dir, "cv_methods_summary.csv"))
    save_aggregated_json(all_methods, os.path.join(out_dir, "aggregated_metrics.json"))

    print(f"\n=== {model_name} ({dataset}) Summary ===")
    for r in rows:
        print(f"{r['method']:>10} | F1={r['mean_macro_f1']:.3f} | Acc={r['mean_accuracy']:.3f}")


if __name__ == "__main__":
    main()
