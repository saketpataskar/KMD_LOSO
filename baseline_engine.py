# baseline_engine.py
import os
import torch
import numpy as np

from utils import load_npz, fit_channel_scaler, apply_channel_scaler, compute_metrics, save_metrics
from dataset import UCIHARDataset
from models import HAR_CNN, HAR_LSTM, HAR_CNN_LSTM
from trainer import run_train_validation

def run_baseline(npz_path, output_dir, model_name="CNN", device="cuda"):
    X, y, subjects, idx_map = load_npz(npz_path)
    os.makedirs(output_dir, exist_ok=True)

    # Identify train/test groups from original dataset
    train_idx = np.where(idx_map == "train")[0]
    test_idx = np.where(idx_map == "test")[0]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # No leakage! Fit scaler only on training data
    scaler = fit_channel_scaler(X_train)
    X_train_s = apply_channel_scaler(X_train, scaler)
    X_test_s = apply_channel_scaler(X_test, scaler)

    # Choose model + dataset format
    if model_name.upper() == "CNN":
        ds_train = UCIHARDataset(X_train_s, y_train, channels_first=True)
        ds_test = UCIHARDataset(X_test_s, y_test, channels_first=True)
        model = HAR_CNN(in_channels=X.shape[2], num_classes=len(np.unique(y)))
    elif model_name.upper() == "LSTM":
        ds_train = UCIHARDataset(X_train_s, y_train, channels_first=False)
        ds_test = UCIHARDataset(X_test_s, y_test, channels_first=False)
        model = HAR_LSTM(in_channels=X.shape[2], num_classes=len(np.unique(y)))
    elif model_name.upper() == "CNNLSTM":
        ds_train = UCIHARDataset(X_train_s, y_train, channels_first=False)
        ds_test = UCIHARDataset(X_test_s, y_test, channels_first=False)
        model = HAR_CNN_LSTM(in_channels=X.shape[2], num_classes=len(np.unique(y)))

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Train using train split, validate on test split
    trained_model, metrics, (y_true, y_pred), history = run_train_validation(
        model, ds_train, ds_test, device, output_dir,
        epochs=50, batch_size=64, lr=1e-3, patience=8
    )

    # Full test metrics
    met = compute_metrics(y_true, y_pred, labels=list(range(len(np.unique(y)))))
    met["training_metrics"] = metrics
    save_metrics(met, os.path.join(output_dir, "metrics.json"))

    print("Baseline evaluation finished.")
    print("Accuracy:", met["accuracy"])
    print("Macro F1:", met["macro_f1"])

    return met
