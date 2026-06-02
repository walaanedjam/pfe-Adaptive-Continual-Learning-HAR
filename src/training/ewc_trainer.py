"""
EWC — Elastic Weight Consolidation (Kirkpatrick et al., 2017).

Baseline SOTA pour la comparaison avec notre méthode.

EWC ajoute une pénalité de régularisation qui ralentit l'apprentissage
sur les paramètres importants pour les tâches précédentes.
La "Fisher Information Matrix" estime l'importance de chaque paramètre.

Perte totale :
  L = L_CE(tâche_actuelle) + (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

  où F_i = importance du paramètre i, θ*_i = valeur optimale sur tâche précédente.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..models.har_model   import HARContinualModel
from ..evaluation.metrics import macro_f1, ContinualResultsMatrix


class EWCTrainer:
    """
    Entraîneur EWC pour l'apprentissage continu.

    Args:
        model:  modèle HAR (backbone + classifieur softmax)
        ewc_lambda: force de la pénalité de régularisation
    """

    def __init__(self, model: HARContinualModel, ewc_lambda: float = 5000.0):
        self.model      = model
        self.ewc_lambda = ewc_lambda
        self.fisher:    Dict[str, torch.Tensor] = {}
        self.opt_params: Dict[str, torch.Tensor] = {}

    def _compute_fisher(self, loader, device: torch.device, n_batches: int = 50):
        """Calcule la diagonale de la Fisher Information Matrix."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters()
                  if p.requires_grad}

        count = 0
        for X, y in loader:
            if count >= n_batches:
                break
            X, y = X.to(device), y.to(device)
            self.model.zero_grad()
            logits = self.model(X)
            loss   = F.cross_entropy(logits, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1

        for n in fisher:
            fisher[n] /= max(count, 1)

        # Accumule les Fisher sur toutes les tâches vues
        for n in fisher:
            if n in self.fisher:
                self.fisher[n] += fisher[n]
            else:
                self.fisher[n]  = fisher[n]

    def _save_optimal_params(self):
        """Sauvegarde les paramètres optimaux après entraînement sur une tâche."""
        self.opt_params = {n: p.detach().clone()
                           for n, p in self.model.named_parameters()
                           if p.requires_grad}

    def _ewc_loss(self) -> torch.Tensor:
        """Calcule la pénalité EWC."""
        if not self.fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.opt_params:
                loss += (self.fisher[n] *
                         (p - self.opt_params[n].to(p.device)) ** 2).sum()
        return (self.ewc_lambda / 2.0) * loss

    def train_task(self, loader, device: torch.device,
                   n_epochs: int = 10, lr: float = 5e-5):
        """Entraîne sur une tâche avec pénalité EWC."""
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        for _ in range(n_epochs):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self.model(X)
                loss   = F.cross_entropy(logits, y) + self._ewc_loss()
                loss.backward()
                optimizer.step()

        # Après entraînement : Fisher + sauvegarde des params
        self._compute_fisher(loader, device)
        self._save_optimal_params()


def run_ewc(model: HARContinualModel,
            task_sequence: list,
            test_datasets: list,
            n_epochs_per_task: int = 10,
            batch_size: int = 64,
            lr: float = 5e-5,
            ewc_lambda: float = 5000.0,
            device: str = "cpu",
            verbose: bool = True) -> ContinualResultsMatrix:
    """Lance l'entraînement EWC sur une séquence de tâches."""

    device_t = torch.device(device)
    model    = model.to(device_t)
    trainer  = EWCTrainer(model, ewc_lambda=ewc_lambda)
    n_tasks  = len(task_sequence)
    matrix   = ContinualResultsMatrix(n_tasks)

    for task_i, task_ds in enumerate(task_sequence):
        if verbose:
            print(f"\n[EWC] Tâche {task_i+1}/{n_tasks} | "
                  f"classes: {sorted(np.unique(task_ds.y).tolist())} | "
                  f"samples: {len(task_ds)}")

        loader = task_ds.dataloader(batch_size=batch_size, shuffle=True)
        trainer.train_task(loader, device_t, n_epochs=n_epochs_per_task, lr=lr)

        # Évaluer sur toutes les tâches vues (softmax)
        for task_j in range(task_i + 1):
            f1 = _eval_softmax(model, test_datasets[task_j], device_t)
            matrix.record(task_i, task_j, f1)
            if verbose:
                print(f"  eval tâche {task_j}: F1={f1:.4f}")

    if verbose:
        print("\n" + matrix.summary())
    return matrix


def _eval_softmax(model, ds, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in ds.dataloader(batch_size=128, shuffle=False):
            p = model(X.to(device)).argmax(dim=-1).cpu().numpy()
            preds.append(p); labels.append(y.numpy())
    return macro_f1(np.concatenate(labels), np.concatenate(preds))
