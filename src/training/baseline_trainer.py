"""
Naive sequential fine-tuning baseline (no replay, no prototypes).

This demonstrates catastrophic forgetting without our mitigation strategy.
Used as the lower-bound comparison for BWT/Forgetting metrics in the thesis.

The model is simply fine-tuned on each task in sequence with standard
cross-entropy — no memory, no contrastive loss, no prototype adaptation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ..models.har_model import HARContinualModel
from ..evaluation.metrics import macro_f1, ContinualResultsMatrix


def naive_finetune(
    model:             HARContinualModel,
    task_sequence:     list,
    test_datasets:     list,
    n_epochs_per_task: int   = 10,
    batch_size:        int   = 64,
    lr:                float = 5e-5,
    device:            str   = "cpu",
    verbose:           bool  = True,
) -> ContinualResultsMatrix:
    """
    Sequential fine-tuning with NO replay, NO contrastive loss,
    NO prototype memory — pure catastrophic forgetting baseline.

    Uses a standard softmax classifier head (not prototypes).
    """
    device  = torch.device(device)
    model   = model.to(device)
    n_tasks = len(task_sequence)
    matrix  = ContinualResultsMatrix(n_tasks)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for task_i, task_ds in enumerate(task_sequence):
        if verbose:
            print(f"\n{'='*50}")
            print(f"[Baseline] Task {task_i+1}/{n_tasks} | "
                  f"classes: {sorted(np.unique(task_ds.y).tolist())} | "
                  f"samples: {len(task_ds)}")

        loader = task_ds.dataloader(batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(n_epochs_per_task):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                # Standard forward — softmax head, NO replay
                logits = model(X_batch)
                loss   = F.cross_entropy(logits, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate on ALL tasks seen so far using softmax (not prototypes)
        for task_j in range(task_i + 1):
            f1 = _eval_softmax(model, test_datasets[task_j], device)
            matrix.record(task_i, task_j, f1)
            if verbose:
                print(f"  eval task {task_j}: F1={f1:.4f}")

    if verbose:
        print("\n" + matrix.summary())

    return matrix


def _eval_softmax(model: HARContinualModel, ds, device: torch.device) -> float:
    model.eval()
    loader = ds.dataloader(batch_size=128, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    return macro_f1(np.concatenate(all_labels), np.concatenate(all_preds))
