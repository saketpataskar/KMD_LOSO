import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

# === IMPORT UTILITIES ===
from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn
from utils.dataset_splits import subject_independent_split
from utils.results_logger import append_row

# === IMPORT CNN MODEL ===
from models.cnn.models_cnn import CNN_Only


# ===========================================
# LABEL MAPPING: HARTH → 0–7
# ===========================================
def map_labels(y):
    mapping = {
        0: 0,
        1: 0, 2: 1, 3: 2, 4: 3,
        5: 4, 6: 5, 7: 6, 8: 7
    }

    return np.array([mapping[int(lbl)] for lbl in y])


# ===========================================
# 1. Load dataset
# ===========================================
print("Loading HARTH dataset...")

# Determine the REAL project root and HARTH path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(project_root, "harth")

print("HARTH path:", data_path)

X, y, subs = load_harth_raw(data_path)

# Remove invalid labels + map to 0–7
X, y, subs = filter_valid_labels(X, y, subs)
y = map_labels(y)

# ===========================================
# 2. Windowing + normalization
# ===========================================
Xw, yw, subw = segment_windows(X, y, subs, window_size=250, step_size=250)
yw = map_labels(yw)   # ensure window labels also mapped

Xw = normalize_windows(Xw)
X_cnn = format_for_cnn(Xw)

print("Windows:", X_cnn.shape)


# ===========================================
# 3. Train/Val/Test SPLIT
# ===========================================
X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test = \
    subject_independent_split(X_cnn, yw, subw)

print("Train:", X_train.shape)
print("Val:",   X_val.shape)
print("Test:",  X_test.shape)


# ===========================================
# Convert to tensors
# ===========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)


# ===========================================
# 4. CNN MODEL
# ===========================================
model = CNN_Only(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ===========================================
# 5. Training with Early Stopping
# ===========================================
best_val_loss = float("inf")
best_epoch = 0
wait = 0
patience = 5

print("\nTraining CNN...\n")

for epoch in range(1, 31):

    # ---- TRAIN ----
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

    # ---- VALIDATION ----
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()

    print(f"Epoch {epoch:02d} | Train={train_loss:.4f} | Val={val_loss:.4f}")

    # Log epoch to Excel
    append_row("CNN_FixedSplit", {
        "Entry": "Epoch",
        "Epoch": epoch,
        "Train Loss": float(train_loss),
        "Val Loss": float(val_loss)
    })

    # ---- Early Stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping!")
            break


print("\nBest Epoch:", best_epoch)


# ===========================================
# 6. FINAL TEST PERFORMANCE
# ===========================================
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
    y_true = y_test.numpy()

acc = accuracy_score(y_true, preds)
macro_f1 = f1_score(y_true, preds, average="macro")

print("\n===== FINAL CNN TEST RESULTS =====")
print("Accuracy:", acc)
print("Macro-F1:", macro_f1)
print("Best Epoch:", best_epoch)
print("================================\n")

# Save final row to Excel
append_row("CNN_FixedSplit", {
    "Entry": "Final",
    "Model": "CNN",
    "Accuracy": float(acc),
    "Macro-F1": float(macro_f1),
    "Best Epoch": int(best_epoch)
})
