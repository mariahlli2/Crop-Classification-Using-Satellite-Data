
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             f1_score, classification_report, confusion_matrix)


def train_epoch(model, loader, optimizer, criterion, device, use_covariates):
    
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch in loader:
        if use_covariates:
            X, mask, cov, y = batch
            X, mask, cov, y = X.to(device), mask.to(device), cov.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X, mask, cov)
        else:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X, mask)
            
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
        
    return total_loss / total, correct / total


def evaluate(model, loader, criterion=None, device=None, use_covariates=True):
    """Evaluate model."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            if use_covariates:
                X, mask, cov, y = batch
                X, mask, cov, y = X.to(device), mask.to(device), cov.to(device), y.to(device)
                out = model(X, mask, cov)
            else:
                X, mask, y = batch
                X, mask, y = X.to(device), mask.to(device), y.to(device)
                out = model(X, mask)
                
            if criterion is not None:
                total_loss += criterion(out, y).item() * len(y)
                
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    oa = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    loss = total_loss / len(labels) if criterion is not None else None
    
    return oa, kappa, f1_macro, f1_weighted, preds, labels, loss


def train_model(model, train_loader, val_loader, cfg, device, use_covariates):
    """Train model with early stopping."""
    # Class weights for imbalance
    train_labels = train_loader.dataset.labels
    num_classes = int(train_labels.max().item()) + 1
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"], 
                                   weight_decay=cfg.get("WEIGHT_DECAY", 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.get("LR_FACTOR", 0.7),
        patience=cfg.get("LR_PATIENCE", 5), min_lr=cfg.get("LR_MIN", 1e-6),
    )

    best_val_kappa = -1
    best_state = None
    patience = cfg.get("PATIENCE", cfg["N_EPOCHS"])
    min_delta = cfg.get("MIN_DELTA", 0.0)
    early_stopping = cfg.get("EARLY_STOPPING", False)
    no_improve = 0
    
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [],
        "val_oa": [], "val_kappa": [], "val_f1": [], "val_f1_weighted": [],
    }

    for epoch in range(1, cfg["N_EPOCHS"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, use_covariates)
        val_oa, val_kappa, val_f1, val_f1_w, _, _, val_loss = evaluate(
            model, val_loader, criterion, device, use_covariates)
        
        scheduler.step(val_kappa)

        for k, v in zip(history.keys(), [tr_loss, tr_acc, val_loss, val_oa, val_kappa, val_f1, val_f1_w]):
            history[k].append(v)

        if val_kappa > best_val_kappa + min_delta:
            best_val_kappa = val_kappa
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == cfg["N_EPOCHS"] or (early_stopping and no_improve >= patience):
            print(f"  Epoch {epoch:3d}/{cfg['N_EPOCHS']} | "
                  f"train_loss={tr_loss:.4f} | train_acc={tr_acc:.4f} | "
                  f"val_OA={val_oa:.4f} | val_Kappa={val_kappa:.4f} | "
                  f"val_macroF1={val_f1:.4f}")

        if early_stopping and no_improve >= patience:
            print(f"  Early stopping triggered after {epoch} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, history


def print_metrics(oa, kappa, f1, f1_w, preds, labels, class_names, state):
    """Print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  RESULTS - {state}")
    print(f"  OA          = {oa:.4f}")
    print(f"  Kappa       = {kappa:.4f}")
    print(f"  Macro-F1    = {f1:.4f}")
    print(f"  Weighted-F1 = {f1_w:.4f}")
    print(f"\n  Per-class report:")
    label_ids = list(range(len(class_names)))
    names = [class_names[i] if i < len(class_names) else f"cls{i}" for i in label_ids]
    print(classification_report(labels, preds, labels=label_ids,
                                target_names=names, zero_division=0))


def compute_metrics_dict(oa, kappa, f1, f1_w, n_params, train_time, inf_time):
    """Return metrics as dictionary."""
    return {
        "OA": round(oa, 4),
        "Kappa": round(kappa, 4),
        "F1_macro": round(f1, 4),
        "F1_weighted": round(f1_w, 4),
        "n_params": n_params,
        "train_time_s": round(train_time, 2),
        "inference_time_s": round(inf_time, 3),
    }