# cv_engine.py
import os, json
import numpy as np
import torch
from utils import load_npz, get_kfold_splits, get_stratified_splits, get_groupkfold_splits, get_loso_splits, fit_channel_scaler, apply_channel_scaler, compute_metrics, save_metrics
from dataset import UCIHARDataset
from models import HAR_CNN, HAR_LSTM, HAR_CNN_LSTM
from trainer import run_train_validation

def run_cv(npz_path, output_dir, cv_type="LOSO", model_name="CNN", n_splits=5, device="cuda"):
    X, y, subjects, idx_map = load_npz(npz_path)
    os.makedirs(output_dir, exist_ok=True)
    if cv_type == "KFold":
        splits = list(get_kfold_splits(X, y, n_splits=n_splits))
    elif cv_type == "Stratified":
        splits = list(get_stratified_splits(X, y, n_splits=n_splits))
    elif cv_type == "GroupKFold":
        splits = list(get_groupkfold_splits(X, y, subjects, n_splits=n_splits))
    elif cv_type == "LOSO":
        splits = list(get_loso_splits(X, y, subjects))
    else:
        raise ValueError("Unknown cv_type")

    all_fold_metrics = []
    print(model_name, " ", cv_type)
    for fold, (train_idx, val_idx) in enumerate(splits):
        print("Starting fold", fold, "train windows", len(train_idx), "val windows", len(val_idx))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # fit scaler on train-only
        scaler = fit_channel_scaler(X_train)
        X_train_s = apply_channel_scaler(X_train, scaler)
        X_val_s = apply_channel_scaler(X_val, scaler)
        # create datasets
        # for CNN we will use channels_first in dataset (C,T) so set channels_first True for conv
        if model_name.upper() == "CNN":
            ds_train = UCIHARDataset(X_train_s, y_train, transform=None, channels_first=True)
            ds_val = UCIHARDataset(X_val_s, y_val, transform=None, channels_first=True)
            model = HAR_CNN(in_channels=X.shape[2], num_classes=len(np.unique(y)))
        elif model_name.upper() == "LSTM":
            ds_train = UCIHARDataset(X_train_s, y_train, transform=None, channels_first=False)
            ds_val = UCIHARDataset(X_val_s, y_val, transform=None, channels_first=False)
            model = HAR_LSTM(in_channels=X.shape[2], num_classes=len(np.unique(y)))
        elif model_name.upper() == "CNNLSTM":
            ds_train = UCIHARDataset(X_train_s, y_train, transform=None, channels_first=False)  # model expects (B,T,C)
            ds_val = UCIHARDataset(X_val_s, y_val, transform=None, channels_first=False)
            model = HAR_CNN_LSTM(in_channels=X.shape[2], num_classes=len(np.unique(y)))
        else:
            raise ValueError("Unknown model_name")

        out_fold_dir = os.path.join(output_dir, f"{cv_type}_{model_name}_fold{fold}")
        os.makedirs(out_fold_dir, exist_ok=True)

        # train
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        trained_model, metrics, (y_true, y_pred), history = run_train_validation(
            model, ds_train, ds_val, device, out_fold_dir,
            epochs=50, batch_size=64, lr=1e-3, patience=8
        )

        # compute and save full metrics (incl confusion matrix)
        met = compute_metrics(y_true, y_pred, labels=list(range(len(np.unique(y)))))
        met["training_metrics"] = metrics
        all_fold_metrics.append(met)
        save_metrics(met, os.path.join(out_fold_dir, "metrics.json"))
        print("Fold finished. val_f1:", metrics["val_f1"])

    # aggregate
    agg = {
        "cv_type": cv_type,
        "model": model_name,
        "n_folds": len(all_fold_metrics),
        "folds": all_fold_metrics
    }
    save_metrics(agg, os.path.join(output_dir, f"{cv_type}_{model_name}_summary.json"))
    return agg
