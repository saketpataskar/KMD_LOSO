# utils.py
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import json, os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]          # (N, 128, C)
    y = d["y"].astype(int)
    subjects = d["subjects"].astype(int)
    idx_map = d.get("idx_map", None)
    return X, y, subjects, idx_map

# CV splitters
def get_kfold_splits(X, y, n_splits=5, shuffle=True, random_state=0):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for tr, val in kf.split(X):
        yield tr, val

def get_stratified_splits(X, y, n_splits=5, random_state=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, val in skf.split(X, y):
        yield tr, val

def get_groupkfold_splits(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    for tr, val in gkf.split(X, y, groups):
        yield tr, val

def get_loso_splits(X, y, groups):
    logo = LeaveOneGroupOut()
    for tr, val in logo.split(X, y, groups):
        yield tr, val

# per-fold scaler (no leakage)
def fit_channel_scaler(X_train):
    # X_train: (N_train, T, C)
    n_chan = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_chan))
    return scaler

def apply_channel_scaler(X, scaler):
    n_chan = X.shape[2]
    return scaler.transform(X.reshape(-1, n_chan)).reshape(X.shape)

# metrics + saving
def compute_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred, labels=labels) if labels is not None else confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "per_class_f1": per_class.tolist(), "confusion_matrix": cm.tolist(), "report": report}

def save_metrics(metrics_dict, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(metrics_dict, f, indent=2)
