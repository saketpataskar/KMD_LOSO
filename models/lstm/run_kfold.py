import sys, os

# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

print("Project root added:", PROJECT_ROOT)

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn
from utils.results_logger import append_row

from models.lstm.models_lstm import HARTH_LSTM


# -------------------------------------------------
# 1. Load HARTH
# -------------------------------------------------
print("Loading HARTH...")

HARTH_DIR = os.path.join(PROJECT_ROOT, "harth")
X, y, subs = load_harth_raw(HARTH_DIR)

X, y, subs = filter_valid_labels(X, y, subs)

Xw, yw, subw = segment_windows(X, y, subs, 250, 250)
Xw = normalize_windows(Xw)
X_cnn = format_for_cnn(Xw)

print("Dataset ready:", X_cnn.shape)

X = X_cnn
y = yw


# -------------------------------------------------
# 2. K-Fold setup
# -------------------------------------------------
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold = 0
fold_accs = []
fold_f1s = []
fold_epochs = []


print(f"\n===== Running LSTM {k}-Fold CV =====\n")


# -------------------------------------------------
# 3. Loop over folds
# -------------------------------------------------
for train_idx, val_idx in kf.split(X):
    fold += 1
    print(f"\n===== Fold {fold} =====")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    device = torch.device("cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=64, shuffle=True)

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = HARTH_LSTM(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    patience = 5
    best_val = float("inf")
    best_epoch = 0
    wait = 0

    print("Training...")

    # -------------------------------------------------
    # Train loop
    # -------------------------------------------------
    for epoch in range(1, 31):
        model.train()
        run_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * xb.size(0)

        train_loss = run_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            val_loss = criterion(logits, y_val_t).item()

        print(f"Epoch {epoch:02d} | Train={train_loss:.4f} | Val={val_loss:.4f}")

        append_row("LSTM_KFold", {
            "Entry": "Epoch",
            "Fold": fold,
            "Epoch": epoch,
            "Train Loss": float(train_loss),
            "Val Loss": float(val_loss)
        })

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break

    # -------------------------------------------------
    # Evaluate fold
    # -------------------------------------------------
    with torch.no_grad():
        preds = model(X_val_t).argmax(dim=1).cpu().numpy()
        y_true = y_val_t.numpy()

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")

    print(f"Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}, BestEpoch={best_epoch}")

    fold_accs.append(acc)
    fold_f1s.append(f1)
    fold_epochs.append(best_epoch)

    append_row("LSTM_KFold", {
        "Entry": "FoldSummary",
        "Fold": fold,
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "Best Epoch": int(best_epoch)
    })


# -------------------------------------------------
# 4. Final Summary
# -------------------------------------------------
print("\n===== LSTM K-Fold Summary =====")
print("Mean Accuracy:", np.mean(fold_accs))
print("Std Accuracy:", np.std(fold_accs))
print("Mean F1:", np.mean(fold_f1s))
print("Std F1:", np.std(fold_f1s))
print("Mean Best Epoch:", np.mean(fold_epochs))

append_row("LSTM_KFold", {
    "Entry": "FinalSummary",
    "Mean Acc": float(np.mean(fold_accs)),
    "Std Acc": float(np.std(fold_accs)),
    "Mean F1": float(np.mean(fold_f1s)),
    "Std F1": float(np.std(fold_f1s)),
    "Mean BestEpoch": float(np.mean(fold_epochs))
})
