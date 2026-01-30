import os
import yaml
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
CONFIG_PATH = "config.yaml"
OUTPUT_EXCEL = "RQ1_Per_Class_F1_Table.xlsx"

MODELS = ["CNN", "LSTM", "CNNLSTM", "InceptionTime"]
DATASETS = ["har", "harth"]

# -----------------------------
# LOAD YAML
# -----------------------------
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RESULTS_ROOT = config["paths"]["analysis"]["output_root"]

print(f" Results Root: {RESULTS_ROOT}")

# -----------------------------
# HELPERS
# -----------------------------
def get_analysis_folder(model, dataset):
    return f"analysis_outputs_{model}_{dataset}"

# -----------------------------
# MAIN
# -----------------------------
rows = []

for dataset in DATASETS:
    for model in MODELS:
        folder_name = get_analysis_folder(model, dataset)
        folder_path = os.path.join(RESULTS_ROOT, folder_name)

        print(f"\n Reading: {folder_path}")

        if not os.path.exists(folder_path):
            print(" Folder not found")
            continue

        csv_path = os.path.join(folder_path, "cv_methods_summary.csv")

        if not os.path.exists(csv_path):
            print(" Missing: cv_methods_summary.csv")
            continue

        df = pd.read_csv(csv_path)

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Expected columns
        required = {"cv_method", "class", "per_class_f1"}
        if not required.issubset(set(df.columns)):
            print(f" Unexpected CSV format: {df.columns}")
            continue

        for _, row in df.iterrows():
            rows.append({
                "Dataset": dataset.upper(),
                "Model": model,
                "CV Method": row["cv_method"],
                "Class": row["class"],
                "Per-Class F1": round(float(row["per_class_f1"]), 4)
            })

# -----------------------------
# SAVE
# -----------------------------
if not rows:
    print("\n No data collected — Excel not generated")
else:
    out_df = pd.DataFrame(rows)

    out_df.sort_values(
        by=["Dataset", "Model", "CV Method", "Class"],
        inplace=True
    )

    out_df.to_excel(OUTPUT_EXCEL, index=False)

    print("\n RQ1 Excel Table Generated")
    print(f" File: {OUTPUT_EXCEL}")
    print(f" Total Rows: {len(out_df)}")
