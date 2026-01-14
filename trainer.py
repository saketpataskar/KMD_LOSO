# trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time, os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_one_epoch(model, optim, loss_fn, dataloader, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds); all_labels.append(yb.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1

def eval_one_epoch(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds); all_labels.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, all_labels, all_preds

def run_train_validation(model, train_ds, val_ds, device, out_dir,
                         epochs=50, batch_size=64, lr=1e-3, patience=8):
    os.makedirs(out_dir, exist_ok=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1.0
    best_epoch = 0
    best_state = None
    history = {"train_loss":[], "train_acc":[], "train_f1":[], "val_loss":[], "val_acc":[], "val_f1":[]}
    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, optim, loss_fn, train_loader, device)
        val_loss, val_acc, val_f1, y_true, y_pred = eval_one_epoch(model, loss_fn, val_loader, device)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc); history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc); history["val_f1"].append(val_f1)
        print(f"Epoch {epoch} | tr_loss {tr_loss:.4f} tr_f1 {tr_f1:.4f} | val_loss {val_loss:.4f} val_f1 {val_f1:.4f} (time {time.time()-t0:.1f}s)")
        # early stopping on macro F1
        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().state_dict() for k,v in model.named_children()} if False else model.state_dict()
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        if epoch - best_epoch >= patience:
            print("Early stopping triggered.")
            break
    # After training, load best state
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    # final eval
    val_loss, val_acc, val_f1, y_true, y_pred = eval_one_epoch(model, loss_fn, val_loader, device)
    # save history and metrics
    np.savez(os.path.join(out_dir, "history.npz"), **history)
    metrics = {"val_loss": float(val_loss), "val_acc": float(val_acc), "val_f1": float(val_f1)}
    return model, metrics, (y_true, y_pred), history
