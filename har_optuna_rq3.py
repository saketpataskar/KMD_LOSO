import os
import json
import optuna
import numpy as np
import pandas as pd
import torch

from utils import (
    load_npz,
    get_kfold_splits,
    get_stratified_splits,
    get_groupkfold_splits,
    get_loso_splits,
    fit_channel_scaler,
    apply_channel_scaler
)

from dataset import UCIHARDataset
from models import HAR_CNN, HAR_LSTM, HAR_CNN_LSTM
from trainer import run_train_validation

# ==========================
# CONFIG
# ==========================

NPZ_PATH = r"D:\KMD Project v2\ucihar_raw_merged.npz"
DIFFICULTY_CSV = r"D:\KMD Project v2\har_subject_difficulty.csv"

OUTPUT_DIR = r"D:\KMD Project v2\optuna_rq3_results"
MODEL_NAME = "CNN"        # CNN / LSTM / CNNLSTM
CV_TYPE = "GroupKFold"    # KFold / Stratified / GroupKFold / LOSO
N_SPLITS = 5              # ignored for LOSO
N_TRIALS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# SUBJECT SELECTION (RQ3)
# ==========================

def select_subjects(csv_path, n_total=15):
    df = pd.read_csv(csv_path)

    df_hard = df[df["difficulty"] == "hard"].head(n_total // 3)
    df_medium = df[df["difficulty"] == "medium"].head(n_total // 3)
    df_easy = df[df["difficulty"] == "easy"].head(n_total // 3)

    selected = pd.concat([df_hard, df_medium, df_easy])
    subjects = selected["subject"].astype(int).tolist()

    print(f"[RQ3] Selected subjects ({len(subjects)}): {subjects}")
    return set(subjects)

# ==========================
# DATA LOADING + FILTERING
# ==========================

print("Loading NPZ...")
X, y, subjects, _ = load_npz(NPZ_PATH)

print("Selecting RQ3 subjects...")
RQ3_SUBJECTS = select_subjects(DIFFICULTY_CSV)

mask = np.isin(subjects, list(RQ3_SUBJECTS))
X = X[mask]
y = y[mask]
subjects = subjects[mask]

print(f"[RQ3] Windows used: {len(y)}")
print(f"[RQ3] Subjects used: {np.unique(subjects)}")

NUM_CLASSES = len(np.unique(y))
IN_CHANNELS = X.shape[2]

# ==========================
# CV SPLITS
# ==========================

def get_splits():
    if CV_TYPE == "KFold":
        return list(get_kfold_splits(X, y, n_splits=N_SPLITS))
    elif CV_TYPE == "Stratified":
        return list(get_stratified_splits(X, y, n_splits=N_SPLITS))
    elif CV_TYPE == "GroupKFold":
        return list(get_groupkfold_splits(X, y, subjects, n_splits=N_SPLITS))
    elif CV_TYPE == "LOSO":
        return list(get_loso_splits(X, y, subjects))
    else:
        raise ValueError("Unknown CV_TYPE")

SPLITS = get_splits()

# ==========================
# OPTUNA OBJECTIVE
# ==========================

def objective(trial):

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    patience = trial.suggest_int("patience", 5, 12)

    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(SPLITS):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = fit_channel_scaler(X_train)
        X_train = apply_channel_scaler(X_train, scaler)
        X_val = apply_channel_scaler(X_val, scaler)

        if MODEL_NAME == "CNN":
            ds_train = UCIHARDataset(X_train, y_train, channels_first=True)
            ds_val = UCIHARDataset(X_val, y_val, channels_first=True)
            model = HAR_CNN(IN_CHANNELS, NUM_CLASSES)
        elif MODEL_NAME == "LSTM":
            ds_train = UCIHARDataset(X_train, y_train, channels_first=False)
            ds_val = UCIHARDataset(X_val, y_val, channels_first=False)
            model = HAR_LSTM(IN_CHANNELS, NUM_CLASSES)
        elif MODEL_NAME == "CNNLSTM":
            ds_train = UCIHARDataset(X_train, y_train, channels_first=False)
            ds_val = UCIHARDataset(X_val, y_val, channels_first=False)
            model = HAR_CNN_LSTM(IN_CHANNELS, NUM_CLASSES)
        else:
            raise ValueError("Unknown model")

        fold_dir = os.path.join(OUTPUT_DIR, f"trial{trial.number}_fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        _, metrics, _, _ = run_train_validation(
            model,
            ds_train,
            ds_val,
            device=DEVICE,
            out_dir=fold_dir,
            epochs=50,
            batch_size=batch_size,
            lr=lr,
            patience=patience
        )

        fold_f1s.append(metrics["val_f1"])

    mean_f1 = float(np.mean(fold_f1s))
    trial.set_user_attr("fold_f1s", fold_f1s)

    return mean_f1

# ==========================
# RUN OPTUNA
# ==========================

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

# ==========================
# SAVE RESULTS
# ==========================

best = {
    "best_value": study.best_value,
    "best_params": study.best_params,
    "cv_type": CV_TYPE,
    "model": MODEL_NAME,
    "subjects": sorted(list(RQ3_SUBJECTS))
}

with open(os.path.join(OUTPUT_DIR, "best_result.json"), "w") as f:
    json.dump(best, f, indent=2)

print("\n=== OPTUNA RQ3 COMPLETE ===")
print("Best F1:", study.best_value)
print("Best params:", study.best_params)
