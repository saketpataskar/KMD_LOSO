import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn

from models.cnn_lstm.models_cnn_lstm import CNN_LSTM
from utils.results_logger import append_row


# =====================================================
# Load HARTH
# =====================================================
print("Loading HARTH dataset...")

X, y, subs = load_harth_raw("../../harth")
X, y, subs = filter_valid_labels(X, y, subs)
Xw, yw, subw = segment_windows(X, y, subs, window_size=250, step_size=250)
Xw = normalize_windows(Xw)
X_cnn = format_for_cnn(Xw)

print("Data ready:", X_cnn.shape)


# =====================================================
# GROUP K-FOLD (5 folds, subject-wise)
# =====================================================
gkf = GroupKFold(n_splits=5)
device = torch.device("cpu")

fold_results = []
fold_number = 0

subjects = np.array(subw)

print("\n=========== STARTING GROUP K-FOLD ===========\n")

for train_idx, val_idx in gkf.split(X_cnn, yw, groups=subjects):
    fold_number += 1
    print(f"\n===== Fold {fold_number} =====")

    X_train, X_val = X_cnn[train_idx], X_cnn[val_idx]
    y_train, y_val = yw[train_idx], yw[val_idx]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    train_dl = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=64, shuffle=True)

    # Model + optimizer
    model = CNN_LSTM(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping
    patience = 5
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    # ---------------- TRAINING ----------------
    for epoch in range(1, 31):
        model.train()
        running_loss = 0.0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_dl.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t.to(device))
            val_loss = criterion(logits, y_val_t.to(device)).item()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true = y_val_t.numpy()
            val_acc = accuracy_score(y_true, preds)
            val_macro_f1 = f1_score(y_true, preds, average="macro")

        print(
            f"Fold {fold_number} | Epoch {epoch} | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"Acc={val_acc:.4f} | F1={val_macro_f1:.4f}"
        )

        # ---------- Log epoch ----------
        append_row("GroupKFold", {
            "Fold": fold_number,
            "Entry": "Epoch",
            "Epoch": epoch,
            "Train Loss": float(train_loss),
            "Val Loss": float(val_loss),
            "Val Acc": float(val_acc),
            "Val Macro-F1": float(val_macro_f1)
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break

    # ---------- Final fold summary ----------
    fold_results.append((val_acc, val_macro_f1, best_epoch))

    append_row("GroupKFold", {
        "Fold": fold_number,
        "Entry": "Final",
        "Val Acc": float(val_acc),
        "Val Macro-F1": float(val_macro_f1),
        "Best Epoch": int(best_epoch)
    })


# =====================================================
# Final summary
# =====================================================
accs = [x[0] for x in fold_results]
f1s = [x[1] for x in fold_results]
epochs = [x[2] for x in fold_results]

print("\n===== Group K-Fold Summary =====")
print("Mean Acc:", np.mean(accs))
print("Std Acc:", np.std(accs))
print("Mean Macro-F1:", np.mean(f1s))
print("Std Macro-F1:", np.std(f1s))
print("Mean Best Epoch:", np.mean(epochs))

append_row("GroupKFold", {
    "Fold": "MEAN",
    "Entry": "Summary",
    "Val Acc": float(np.mean(accs)),
    "Val Macro-F1": float(np.mean(f1s)),
    "Best Epoch": float(np.mean(epochs))
})

append_row("GroupKFold", {
    "Fold": "STD",
    "Entry": "Summary",
    "Val Acc": float(np.std(accs)),
    "Val Macro-F1": float(np.std(f1s)),
    "Best Epoch": float(np.std(epochs))
})
