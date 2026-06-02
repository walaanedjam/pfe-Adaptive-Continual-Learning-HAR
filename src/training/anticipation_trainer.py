"""
Training and evaluation loop for the activity anticipation head (Module 2).

The backbone is FROZEN during anticipation training — we only train
the LSTM head on top of the pre-trained backbone embeddings.
This keeps the shared representation intact for continual learning.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.har_model import HARContinualModel
from ..evaluation.metrics import macro_f1


def train_anticipation(
    model:       HARContinualModel,
    datasets:    dict,          # {obs_ratio: (train_ds, val_ds)}
    n_epochs:    int   = 30,
    batch_size:  int   = 64,
    lr:          float = 5e-4,
    device:      str   = "cpu",
    verbose:     bool  = True,
) -> Dict[float, Dict]:
    """
    Train the anticipation head for each observation ratio.

    Backbone weights are FROZEN. Only anticipation_head parameters
    are updated.

    Returns:
        {obs_ratio: {'history': {...}, 'val_f1': float, 'val_acc': float}}
    """
    device = torch.device(device)
    model  = model.to(device)

    # Freeze backbone and HAR head — only train anticipation head
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.har_head.parameters():
        p.requires_grad = False
    for p in model.anticipation_head.parameters():
        p.requires_grad = True

    results = {}

    for obs_ratio, (train_ds, val_ds) in datasets.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Observation ratio p={obs_ratio:.2f} "
                  f"({int(obs_ratio*100)}% of window)")
            print(train_ds.summary())

        if len(train_ds) == 0:
            print("  No samples — skipping.")
            continue

        # Re-init anticipation head weights for each ratio
        _reinit_head(model.anticipation_head)

        optimizer = AdamW(model.anticipation_head.parameters(),
                          lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs,
                                       eta_min=lr * 0.01)

        train_loader = train_ds.dataloader(batch_size=batch_size, shuffle=True)
        val_loader   = val_ds.dataloader(batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

        for epoch in range(1, n_epochs + 1):
            # ---- Train ----
            model.train()
            # Keep backbone in eval mode (frozen)
            model.backbone.eval()
            model.har_head.eval()

            epoch_losses = []
            for X_seq, y_next in train_loader:
                X_seq  = X_seq.to(device)   # (B, S, obs_len, C)
                y_next = y_next.to(device)

                optimizer.zero_grad()
                logits = model.anticipate(X_seq)
                loss   = F.cross_entropy(logits, y_next)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.anticipation_head.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()

            # ---- Validate ----
            val_loss, val_f1, val_acc = _eval_anticipation(
                model, val_loader, device)

            history["train_loss"].append(np.mean(epoch_losses))
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_acc"].append(val_acc)

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  epoch {epoch:3d}/{n_epochs} | "
                      f"train_loss={history['train_loss'][-1]:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"val_F1={val_f1:.4f} | "
                      f"val_acc={val_acc:.4f}")

        best_f1  = max(history["val_f1"])
        best_acc = max(history["val_acc"])
        results[obs_ratio] = {
            "history": history,
            "val_f1":  best_f1,
            "val_acc": best_acc,
        }
        if verbose:
            print(f"  Best val F1={best_f1:.4f}  |  Best val acc={best_acc:.4f}")

    # Unfreeze backbone for future continual learning
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.har_head.parameters():
        p.requires_grad = True

    return results


def _eval_anticipation(model, loader, device):
    model.eval()
    all_preds, all_labels, losses = [], [], []

    with torch.no_grad():
        for X_seq, y_next in loader:
            X_seq  = X_seq.to(device)
            y_next = y_next.to(device)
            logits = model.anticipate(X_seq)
            losses.append(F.cross_entropy(logits, y_next).item())
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_next.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc    = float((y_pred == y_true).mean())
    f1     = macro_f1(y_true, y_pred)
    return float(np.mean(losses)), f1, acc


def _reinit_head(head: nn.Module):
    """Re-initialise LSTM and linear weights for a fresh training run."""
    for m in head.modules():
        if isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def print_anticipation_summary(results: Dict[float, Dict]):
    """Print a clean comparison table across observation ratios."""
    print("\n=== Anticipation Head Results ===")
    print(f"  {'Obs ratio':>10}  {'Val F1':>8}  {'Val Acc':>8}")
    print(f"  {'-'*30}")
    for ratio in sorted(results):
        r = results[ratio]
        print(f"  {ratio*100:>8.0f}%  {r['val_f1']:>8.4f}  {r['val_acc']:>8.4f}")
    print()
