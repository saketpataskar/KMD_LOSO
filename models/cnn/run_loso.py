import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# Add project root to imports
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

# -----------------------------
# Import project modules
# -----------------------------
from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn
from utils.results_logger import append_row

from models.cnn.models_cnn import CNN_Only


# ============================================================
# 1. LOAD + PREPROCESS HARTH
# ============================================================
print("\nLoading HARTH dataset...")

HARTH_DIR = os.path.join(PROJECT_ROOT, "harth")
X, y, subjects = load_harth_raw(HARTH_DIR)

# Remove unwanted labels (keep 0–7)
X, y, subjects = filter_valid_labels(X, y, subjects)

# Segment into windows
Xw, yw, subw = segment_windows(X, y, subjects, window_size=250, step_size=250)

# Normalize per channel
Xw = normalize_windows(Xw)

# CNN format: (N, 6, 250)
X_cnn = format_for_cnn(Xw)

print("Windows:", X_cnn.shape)
print("Unique subjects:", np.unique(subw))


# ============================================================
# 2. LOSO CONFIG
# ============================================================
unique_subjects = sorted(np.unique(subw))
print(f"\n===== Running CNN LOSO ({len(unique_subjects)} subjects) =====")

device = torch.device("cpu")

loso_acc = []
loso_f1 = []
loso_epochs = []


# ============================================================
# 3. LOSO LOOP
# ============================================================
for test_subj in unique_subjects:

    print(f"\n===== Test Subject: {test_subj} =====")

    # Split windows
    train_idx = np.where(subw != test_subj)[0]
    test_idx = np.where(subw == test_subj)[0]

    X_train = X_cnn[train_idx]
    y_train = yw[train_idx]

    X_test = X_cnn[test_idx]
    y_test = yw[test_idx]

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=64, shuffle=True)

    # Create model
    model = CNN_Only().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping
    patience = 5
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    # Since LOSO has no validation split,
    # we use training loss to detect early stopping.
    print("Training...")

    for epoch in range(1, 31):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Log epoch
        append_row("CNN_LOSO", {
            "Entry": "Epoch",
            "Subject": int(test_subj),
            "Epoch": epoch,
            "Train Loss": float(train_loss),
        })

        print(f"Epoch {epoch:02d} | Train Loss = {train_loss:.4f}")

        # Early stopping based on train loss
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break

    # ========================================================
    # Evaluate for this subject
    # ========================================================
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()
        true = y_test_t.numpy()

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")

    print(f"Subject {test_subj}: Acc={acc:.4f}, F1={f1:.4f}, BestEpoch={best_epoch}")

    loso_acc.append(acc)
    loso_f1.append(f1)
    loso_epochs.append(best_epoch)

    # Log fold summary
    append_row("CNN_LOSO", {
        "Entry": "FoldSummary",
        "Subject": int(test_subj),
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "Best Epoch": int(best_epoch)
    })


# ============================================================
# 4. FINAL SUMMARY
# ============================================================
print("\n===== LOSO Summary =====")
print("Mean Accuracy:", np.mean(loso_acc))
print("Std Accuracy:", np.std(loso_acc))
print("Mean Macro-F1:", np.mean(loso_f1))
print("Std Macro-F1:", np.std(loso_f1))
print("Mean Best Epoch:", np.mean(loso_epochs))

append_row("CNN_LOSO", {
    "Entry": "FinalSummary",
    "Mean Acc": float(np.mean(loso_acc)),
    "Std Acc": float(np.std(loso_acc)),
    "Mean F1": float(np.mean(loso_f1)),
    "Std F1": float(np.std(loso_f1)),
    "Mean BestEpoch": float(np.mean(loso_epochs)),
})
