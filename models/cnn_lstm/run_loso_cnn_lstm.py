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

print("Dataset ready:", X_cnn.shape)

subjects = np.unique(subw)
device = torch.device("cpu")

results = []

print("\n========= STARTING LOSO (22 subjects) =========\n")

for test_subject in subjects:
    print(f"\n===== Test Subject: {test_subject} =====")

    # ---------------------------
    # Split LOSO
    # ---------------------------
    test_idx = np.where(subw == test_subject)[0]
    train_idx = np.where(subw != test_subject)[0]

    X_train, X_test = X_cnn[train_idx], X_cnn[test_idx]
    y_train, y_test = yw[train_idx], yw[test_idx]

    # Further split 10% of training as validation
    val_size = int(0.1 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train2 = X_train[val_size:]
    y_train2 = y_train[val_size:]

    # Convert to tensors
    X_train_t = torch.tensor(X_train2, dtype=torch.float32)
    y_train_t = torch.tensor(y_train2, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dl = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    # ---------------------------
    # Model
    # ---------------------------
    model = CNN_LSTM(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    patience = 5
    best_val_loss = float("inf")
    wait = 0
    best_epoch = 0

    # ---------------------------
    # Training loop
    # ---------------------------
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
            f"Sub {test_subject} | Epoch {epoch} | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"Acc={val_acc:.4f} | F1={val_macro_f1:.4f}"
        )

        # ---------- Log epoch ----------
        append_row("LOSO", {
            "Subject": int(test_subject),
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

    # ---------------------------
    # Final TEST evaluation
    # ---------------------------
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true = y_test_t.numpy()

    test_acc = accuracy_score(y_true, preds)
    test_f1 = f1_score(y_true, preds, average="macro")

    print(f"Subject {test_subject} | Test Acc={test_acc:.4f} | Test F1={test_f1:.4f}")

    # Log final
    append_row("LOSO", {
        "Subject": int(test_subject),
        "Entry": "Final",
        "Test Acc": float(test_acc),
        "Test Macro-F1": float(test_f1),
        "Best Epoch": int(best_epoch)
    })

    results.append((test_acc, test_f1, best_epoch))


# =====================================================
# Summary across subjects
# =====================================================
accs = [x[0] for x in results]
f1s = [x[1] for x in results]
epochs = [x[2] for x in results]

append_row("LOSO", {
    "Subject": "MEAN",
    "Entry": "Summary",
    "Test Acc": float(np.mean(accs)),
    "Test Macro-F1": float(np.mean(f1s)),
    "Best Epoch": float(np.mean(epochs))
})

append_row("LOSO", {
    "Subject": "STD",
    "Entry": "Summary",
    "Test Acc": float(np.std(accs)),
    "Test Macro-F1": float(np.std(f1s)),
    "Best Epoch": float(np.std(epochs))
})

print("\n===== LOSO Evaluation Complete =====")
print("Mean Acc:", np.mean(accs))
print("Mean Macro-F1:", np.mean(f1s))
print("Std Acc :", np.std(accs))
print("Std F1  :", np.std(f1s))
