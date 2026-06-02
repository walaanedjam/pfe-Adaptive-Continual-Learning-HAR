"""
Training loops for HAR model.

Two modes:
  1. pretrain()        — standard supervised training on merged datasets
  2. continual_train() — incremental training on a sequence of tasks
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.har_model  import HARContinualModel
from ..evaluation.metrics import macro_f1, ContinualResultsMatrix


# ---------------------------------------------------------------------------
# Pre-training (standard supervised)
# ---------------------------------------------------------------------------

def pretrain(
    model:       HARContinualModel,
    train_ds,
    val_ds,
    n_epochs:    int   = 50,
    batch_size:  int   = 64,
    lr:          float = 1e-3,
    weight_decay: float = 1e-4,
    device:      str   = "cpu",
    verbose:     bool  = True,
) -> Dict[str, List[float]]:
    """
    Standard supervised pre-training on the full merged dataset.

    After pre-training, the backbone has a strong shared representation.
    We then switch to prototype memory for continual inference.

    Returns history dict with train/val loss and F1 per epoch.
    """
    device = torch.device(device)
    model  = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    train_loader = train_ds.dataloader(batch_size=batch_size, shuffle=True)
    val_loader   = val_ds.dataloader(batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, n_epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = F.cross_entropy(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ---- Validate ----
        val_loss, val_f1 = _evaluate(model, val_loader, device)

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"train_loss={history['train_loss'][-1]:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_F1={val_f1:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ---- Initialize prototypes from training data ----
    _init_prototypes(model, train_loader, device)
    if verbose:
        print(f"\nPrototypes initialized for {model.har_head.prototype_memory.n_classes()} classes.")

    return history


def _evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            losses.append(F.cross_entropy(logits, y).item())
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return float(np.mean(losses)), macro_f1(y_true, y_pred)


def _init_prototypes(model, loader, device):
    """Initialize prototype memory from the full training set after pre-training."""
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            emb  = model.backbone(X)
            model.har_head.update_prototypes(emb, y)


# ---------------------------------------------------------------------------
# Continual training
# ---------------------------------------------------------------------------

def continual_train(
    model:             HARContinualModel,
    task_sequence:     list,   # list of HARDataset tasks
    test_datasets:     list,   # list of HARDataset test sets (one per task)
    n_epochs_per_task: int   = 10,
    batch_size:        int   = 64,
    replay_batch:      int   = 32,
    lr:                float = 1e-4,
    device:            str   = "cpu",
    verbose:           bool  = True,
) -> ContinualResultsMatrix:
    """
    Continual learning over a sequence of tasks.

    For each task:
      1. Fine-tune backbone + head with CE + replay + contrastive loss
      2. Update prototype memory
      3. Add task samples to replay buffer
      4. Evaluate on ALL tasks seen so far → fill results matrix

    Returns ContinualResultsMatrix for BWT / FWT / forgetting computation.
    """
    device = torch.device(device)
    model  = model.to(device)

    n_tasks = len(task_sequence)
    matrix  = ContinualResultsMatrix(n_tasks)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for task_i, task_ds in enumerate(task_sequence):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Task {task_i + 1}/{n_tasks} | "
                  f"classes: {sorted(np.unique(task_ds.y).tolist())} | "
                  f"samples: {len(task_ds)}")

        loader = task_ds.dataloader(batch_size=batch_size, shuffle=True)

        # ---- Train on current task ----
        for epoch in range(n_epochs_per_task):
            epoch_losses = []
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                losses = model.continual_step(
                    X_batch, y_batch, optimizer,
                    replay_batch_size=replay_batch,
                    device=device,
                )
                epoch_losses.append(losses["loss_total"])

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  epoch {epoch+1:2d}/{n_epochs_per_task} | "
                      f"loss={np.mean(epoch_losses):.4f} | "
                      f"buffer={len(model.replay_buffer)}")

        # ---- Adapt prototypes after backbone update ----
        if len(model.replay_buffer) > 0:
            rx, ry = model.replay_buffer.sample(
                min(512, len(model.replay_buffer)), device=device
            )
            model.har_head.prototype_memory.adapt_prototypes(
                model.backbone, rx, ry, device
            )

        # ---- Evaluate on all tasks seen so far ----
        for task_j in range(task_i + 1):
            test_j = test_datasets[task_j]
            f1     = _eval_continual(model, test_j, device)
            matrix.record(task_i, task_j, f1)
            if verbose:
                print(f"  eval task {task_j}: F1={f1:.4f}")

    if verbose:
        print("\n" + matrix.summary())

    return matrix


def _eval_continual(model: HARContinualModel, ds, device: torch.device) -> float:
    """Evaluate using prototype-based nearest-mean classifier."""
    if model.har_head.prototype_memory.n_classes() == 0:
        return 0.0

    model.eval()
    loader = ds.dataloader(batch_size=128, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model.predict(X, mode="prototype")
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return macro_f1(y_true, y_pred)
