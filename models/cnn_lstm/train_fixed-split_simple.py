from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn
from utils.dataset_splits import subject_independent_split

from models.cnn_lstm.models_cnn_lstm import CNN_LSTM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from utils.results_logger import append_row


# =====================================================
# 1. Load & preprocess HARTH dataset
# =====================================================
print("Loading HARTH dataset...")

X, y, subs = load_harth_raw("../../harth")
X, y, subs = filter_valid_labels(X, y, subs)
Xw, yw, subw = segment_windows(X, y, subs, window_size=250, step_size=250)
Xw = normalize_windows(Xw)
X_cnn = format_for_cnn(Xw)

print("Data ready! Windows shape:", X_cnn.shape)


# =====================================================
# 2. Subject-independent split (Train / Val / Test)
# =====================================================
X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test = subject_independent_split(
    X_cnn, yw, subw
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# Convert to tensors
device = torch.device("cpu")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.long)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.long)

train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)


# =====================================================
# 3. Model, loss, optimizer
# =====================================================
model = CNN_LSTM(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# =====================================================
# 4. Train with early stopping + log each epoch
# =====================================================
patience = 5
best_val_loss = float("inf")
best_epoch = 0
wait = 0

print("\nStarting training...\n")

for epoch in range(1, 31):
    # ---- Train ----
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

    # ---- Validate ----
    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device))
        val_loss = criterion(logits, y_val.to(device)).item()

    print(f"Epoch {epoch:02d} | Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

    # ---- Log epoch results ----
    append_row("FixedSplit", {
        "Entry": "Epoch",
        "Epoch": epoch,
        "Train Loss": float(train_loss),
        "Val Loss": float(val_loss)
    })

    # ---- Early stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("\nEarly stopping triggered!")
            break

print(f"\nBest Epoch: {best_epoch}")


# =====================================================
# 5. Final Test Evaluation + Logging
# =====================================================
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
    y_true = y_test.numpy()

test_acc = accuracy_score(y_true, preds)
test_f1 = f1_score(y_true, preds, average="macro")

print("\n=========== TEST RESULTS ===========")
print("Accuracy :", test_acc)
print("Macro-F1 :", test_f1)
print("====================================\n")

# ---- Log final results ----
append_row("FixedSplit", {
    "Entry": "Final",
    "Model": "CNN-LSTM",
    "Accuracy": float(test_acc),
    "Macro-F1": float(test_f1),
    "Best Epoch": int(best_epoch)
})
