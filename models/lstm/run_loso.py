import sys, os

# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

print("Project root added:", PROJECT_ROOT)

# -------------------------------------------------
# Imports
# -------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score

from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn
from utils.results_logger import append_row

from models.lstm.models_lstm import HARTH_LSTM



# -------------------------------------------------
# 1. Load HARTH dataset
# -------------------------------------------------
print("Loading HARTH dataset...")

HARTH_DIR = os.path.join(PROJECT_ROOT, "harth")
X, y, subs = load_harth_raw(HARTH_DIR)

# Filter invalid labels
X, y, subs = filter_valid_labels(X, y, subs)

# Window segmentation
Xw, yw, subw = segment_windows(X, y, subs, window_size=250, step_size=250)

# Normalize + reshape to (N, 6, 250)
Xw = normalize_windows(Xw)
X_cnn = format_for_cnn(Xw)

print("Dataset ready:", X_cnn.shape)

X = X_cnn
y = yw
subjects = subw

unique_subjects = sorted(np.unique(subjects))

print("\n===== Running LSTM LOSO =====")
print("Total subjects:", len(unique_subjects), "\n")


# -------------------------------------------------
# Storage for summary
# -------------------------------------------------
loso_acc = []
loso_f1 = []
loso_epochs = []



# -------------------------------------------------
# 2. Loop over subjects (LOSO)
# -------------------------------------------------
for test_sub in unique_subjects:

    print(f"\n===== Test Subject {test_sub} =====")

    # Split data
    train_idx = np.where(subjects != test_sub)[0]
    test_idx  = np.where(subjects == test_sub)[0]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # If a subject has too few windows, skip
    if len(test_idx) < 5:
        print(f"Skipping subject {test_sub} (too few samples)")
        continue

    device = torch.device("cpu")

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=64, shuffle=True)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = HARTH_LSTM(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping parameters
    patience = 5
    best_val = float("inf")
    best_epoch = 0
    wait = 0

    print("Training...")

    # ---------------------------------------------------------
    # Train on training subjects only
    # ---------------------------------------------------------
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

        # No validation set for LOSO → we use TRAIN loss for early stop
        val_loss = train_loss

        print(f"Epoch {epoch:02d} | Train={train_loss:.4f}")

        append_row("LSTM_LOSO", {
            "Entry": "Epoch",
            "Subject": int(test_sub),
            "Epoch": epoch,
            "Train Loss": float(train_loss)
        })

        # Early stopping (based on train loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break

    # ---------------------------------------------------------
    # Evaluate on left-out subject
    # ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()
        y_true = y_test_t.numpy()

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")

    print(f"Subject {test_sub} → Acc={acc:.4f}, F1={f1:.4f}, BestEpoch={best_epoch}")

    loso_acc.append(acc)
    loso_f1.append(f1)
    loso_epochs.append(best_epoch)

    append_row("LSTM_LOSO", {
        "Entry": "SubjectSummary",
        "Subject": int(test_sub),
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "Best Epoch": int(best_epoch)
    })



# ---------------------------------------------------------
# 3. Final summary across all subjects
# ---------------------------------------------------------
print("\n===== LSTM LOSO Summary =====")
print("Mean Accuracy:", np.mean(loso_acc))
print("Std  Accuracy:", np.std(loso_acc))
print("Mean Macro-F1:", np.mean(loso_f1))
print("Std  Macro-F1:", np.std(loso_f1))
print("Mean Best Epoch:", np.mean(loso_epochs))


append_row("LSTM_LOSO", {
    "Entry": "FinalSummary",
    "Mean Acc": float(np.mean(loso_acc)),
    "Std Acc": float(np.std(loso_acc)),
    "Mean F1": float(np.mean(loso_f1)),
    "Std F1": float(np.std(loso_f1)),
    "Mean BestEpoch": float(np.mean(loso_epochs))
})
