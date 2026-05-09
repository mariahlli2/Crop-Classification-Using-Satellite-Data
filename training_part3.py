import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             f1_score, classification_report)



def mixup_batch(x, y, alpha=0.15):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam



class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes, smoothing=0.05, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.weight    = weight

    def forward(self, logits, target):
        log_prob = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_val  = self.smoothing / (self.n_classes - 1)
            soft_target = torch.full_like(log_prob, smooth_val)
            soft_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            w    = self.weight[target]
            loss = -(soft_target * log_prob).sum(dim=1)
            return (loss * w).sum() / w.sum()
        return -(soft_target * log_prob).sum(dim=1).mean()


def train_epoch(model, loader, optimizer, criterion, device, use_covariates,
                mixup_alpha=0.15, grad_clip=1.0, cov_noise_std=0.05):

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        if use_covariates:
            X, mask, cov, y = batch
            X, mask, cov, y = X.to(device), mask.to(device), cov.to(device), y.to(device)
            if cov_noise_std > 0:
                cov = cov + torch.randn_like(cov) * cov_noise_std
        else:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)

        X_mix, y_a, y_b, lam = mixup_batch(X, y, alpha=mixup_alpha)

        optimizer.zero_grad()
        out = model(X_mix, mask, cov) if use_covariates else model(X_mix, mask)

        loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * len(y)
        pred = out.argmax(1)
        correct += (lam * (pred == y_a).float() +
                    (1 - lam) * (pred == y_b).float()).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion=None, device=None, use_covariates=True):
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

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    oa          = accuracy_score(labels, preds)
    kappa       = cohen_kappa_score(labels, preds)
    f1_macro    = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    loss        = total_loss / len(labels) if criterion is not None else None

    return oa, kappa, f1_macro, f1_weighted, preds, labels, loss



def train_model(model, train_loader, val_loader, cfg, device, use_covariates):
   
    train_labels = train_loader.dataset.labels
    num_classes  = int(train_labels.max().item()) + 1
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes

    smoothing = cfg.get("LABEL_SMOOTHING", 0.0)
    criterion = LabelSmoothingCE(num_classes, smoothing=smoothing,
                                  weight=class_weights.to(device))

    
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": cfg.get("WEIGHT_DECAY", 1e-4)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.get("LR", 5e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.get("LR_FACTOR", 0.5),
        patience=cfg.get("LR_PATIENCE", 10), min_lr=cfg.get("LR_MIN", 1e-6),
    )

    best_val_kappa = -1
    best_state     = None
    patience       = cfg.get("PATIENCE", cfg["N_EPOCHS"])
    min_delta      = cfg.get("MIN_DELTA", 0.0)
    early_stopping = cfg.get("EARLY_STOPPING", False)
    no_improve     = 0

    mixup_alpha   = cfg.get("MIXUP_ALPHA",   0.0)
    grad_clip     = cfg.get("GRAD_CLIP",      1.0)
    cov_noise_std = cfg.get("COV_NOISE_STD",  0.0)

    history = {
        "train_loss": [], "train_acc": [], "val_loss": [],
        "val_oa": [], "val_kappa": [], "val_f1": [], "val_f1_weighted": [],
    }

    for epoch in range(1, cfg["N_EPOCHS"] + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, use_covariates,
            mixup_alpha=mixup_alpha,
            grad_clip=grad_clip,
            cov_noise_std=cov_noise_std if use_covariates else 0.0,
        )
        val_oa, val_kappa, val_f1, val_f1_w, _, _, val_loss = evaluate(
            model, val_loader, criterion, device, use_covariates)

        scheduler.step(val_kappa)

        for k, v in zip(history.keys(),
                        [tr_loss, tr_acc, val_loss, val_oa, val_kappa, val_f1, val_f1_w]):
            history[k].append(v)

        if val_kappa > best_val_kappa + min_delta:
            best_val_kappa = val_kappa
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == cfg["N_EPOCHS"] or \
                (early_stopping and no_improve >= patience):
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
    return {
        "OA":               round(oa,    4),
        "Kappa":            round(kappa, 4),
        "F1_macro":         round(f1,    4),
        "F1_weighted":      round(f1_w,  4),
        "n_params":         n_params,
        "train_time_s":     round(train_time, 2),
        "inference_time_s": round(inf_time,   3),
    }