"""
iCaRL — Incremental Classifier and Representation Learning
(Rebuffi et al., 2017) — baseline SOTA pour la comparaison.

Idée principale :
  1. Exemplar selection par herding : pour chaque classe, on conserve
     les k exemples dont la moyenne est la plus proche du prototype
     (sélection greedy).
  2. Classification par nearest-mean (comme nos prototypes).
  3. Distillation knowledge + replay des exemplaires.

Différence avec notre méthode :
  - iCaRL : sélection intelligente des exemplaires (herding)
  - Notre méthode : reservoir sampling + contrastive loss + re-adaptation
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..models.har_model   import HARContinualModel
from ..evaluation.metrics import macro_f1, ContinualResultsMatrix


class iCaRLTrainer:
    """
    iCaRL simplifié adapté à notre architecture HAR.

    Args:
        model:    modèle HAR
        memory_size: nombre total d'exemplaires à conserver
        distill_weight: poids de la distillation knowledge
    """

    def __init__(self, model: HARContinualModel,
                 memory_size: int = 2000,
                 distill_weight: float = 1.0):
        self.model          = model
        self.memory_size    = memory_size
        self.distill_weight = distill_weight

        # Exemplaires par classe {class_id: (X, y)}
        self.exemplars_X: dict = {}
        self.exemplars_y: dict = {}

        # Ancien modèle pour la distillation
        self.old_model: HARContinualModel | None = None

    # ------------------------------------------------------------------
    # Herding exemplar selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _select_exemplars(self, X: np.ndarray, y: np.ndarray,
                           class_id: int, n_exemplars: int,
                           device: torch.device) -> tuple:
        """
        Herding : sélectionne les n_exemplars exemples dont la
        moyenne des embeddings est la plus proche du prototype.
        """
        self.model.eval()
        X_t   = torch.from_numpy(X).to(device)
        embs  = self.model.backbone(X_t).cpu().numpy()   # (N, d)
        mu    = embs.mean(axis=0)                         # prototype

        selected_idx = []
        cumsum = np.zeros_like(mu)

        for k in range(min(n_exemplars, len(X))):
            # Greedy: choisir l'exemple qui minimise ||cumsum/(k+1) - mu||
            candidates = []
            for i in range(len(X)):
                if i in selected_idx:
                    continue
                candidate_sum = cumsum + embs[i]
                dist = np.linalg.norm(candidate_sum / (k + 1) - mu)
                candidates.append((dist, i))
            candidates.sort(key=lambda x: x[0])
            best_idx = candidates[0][1]
            selected_idx.append(best_idx)
            cumsum += embs[best_idx]

        return X[selected_idx], y[selected_idx]

    def _update_exemplars(self, X_task: np.ndarray, y_task: np.ndarray,
                           device: torch.device):
        """Met à jour les exemplaires avec herding après chaque tâche."""
        all_classes = list(self.exemplars_X.keys()) + \
                      list(np.unique(y_task).tolist())
        all_classes = sorted(set(all_classes))

        # Quota par classe
        m = max(1, self.memory_size // len(all_classes))

        # Ajouter les nouvelles classes
        for cls in np.unique(y_task).tolist():
            mask = y_task == cls
            ex_X, ex_y = self._select_exemplars(
                X_task[mask], y_task[mask], cls, m, device)
            self.exemplars_X[cls] = ex_X
            self.exemplars_y[cls] = ex_y

        # Réduire les anciennes classes au nouveau quota
        for cls in list(self.exemplars_X.keys()):
            if len(self.exemplars_X[cls]) > m:
                self.exemplars_X[cls] = self.exemplars_X[cls][:m]
                self.exemplars_y[cls] = self.exemplars_y[cls][:m]

    def _get_exemplar_batch(self, batch_size: int, device: torch.device):
        """Tire un batch depuis tous les exemplaires."""
        if not self.exemplars_X:
            return None, None
        all_X = np.concatenate(list(self.exemplars_X.values()), axis=0)
        all_y = np.concatenate(list(self.exemplars_y.values()), axis=0)
        idx   = np.random.choice(len(all_X), min(batch_size, len(all_X)),
                                  replace=False)
        return (torch.from_numpy(all_X[idx]).to(device),
                torch.from_numpy(all_y[idx]).to(device))

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    def _distill_loss(self, X: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
        """
        Knowledge Distillation : force le nouveau modèle à reproduire
        les sorties du vieux modèle sur les exemplaires.
        """
        if self.old_model is None:
            return torch.tensor(0.0, device=device)

        self.old_model.eval()
        with torch.no_grad():
            old_logits = self.old_model(X)
            old_probs  = torch.sigmoid(old_logits)

        new_logits = self.model(X)
        new_probs  = torch.sigmoid(new_logits)

        loss = F.binary_cross_entropy(
            new_probs.clamp(1e-7, 1 - 1e-7),
            old_probs.detach()
        )
        return loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(self, X_task: np.ndarray, y_task: np.ndarray,
                   device: torch.device,
                   n_epochs: int = 10, lr: float = 5e-5,
                   batch_size: int = 64):
        """Entraîne sur une nouvelle tâche avec distillation + exemplaires."""
        self.model = self.model.to(device)

        # Sauvegarder l'ancien modèle pour la distillation
        if self.exemplars_X:
            import copy
            self.old_model = copy.deepcopy(self.model).to(device)
            self.old_model.eval()

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Préparer le dataset complet (nouvelles données + exemplaires)
        all_X = list(self.exemplars_X.values())
        all_y = list(self.exemplars_y.values())
        if all_X:
            X_all = np.concatenate([X_task] + all_X, axis=0)
            y_all = np.concatenate([y_task] + all_y, axis=0)
        else:
            X_all, y_all = X_task, y_task

        # Shuffle
        idx   = np.random.permutation(len(X_all))
        X_all, y_all = X_all[idx], y_all[idx]

        self.model.train()
        for _ in range(n_epochs):
            for start in range(0, len(X_all), batch_size):
                X_b = torch.from_numpy(
                    X_all[start:start+batch_size]).to(device)
                y_b = torch.from_numpy(
                    y_all[start:start+batch_size]).to(device)

                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(X_b), y_b)

                # Distillation sur les exemplaires
                ex_X, _ = self._get_exemplar_batch(batch_size, device)
                if ex_X is not None:
                    loss = loss + \
                           self.distill_weight * self._distill_loss(ex_X, device)

                loss.backward()
                optimizer.step()

        # Mettre à jour les exemplaires (herding)
        self._update_exemplars(X_task, y_task, device)

        # Mettre à jour les prototypes pour la classification NMC
        self._update_prototypes(device)

    @torch.no_grad()
    def _update_prototypes(self, device: torch.device):
        """Recalcule les prototypes iCaRL depuis les exemplaires."""
        self.model.eval()
        for cls, X_ex in self.exemplars_X.items():
            X_t  = torch.from_numpy(X_ex).to(device)
            embs = self.model.backbone(X_t)
            self.model.har_head.prototype_memory.prototypes[cls] = \
                embs.mean(dim=0).cpu()

    @torch.no_grad()
    def predict(self, X: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Classification NMC (nearest-mean classifier)."""
        self.model.eval()
        emb = self.model.backbone(X.to(device))
        return self.model.har_head.prototype_memory.predict(emb)


def run_icarl(model: HARContinualModel,
              task_sequence: list,
              test_datasets: list,
              n_epochs_per_task: int = 10,
              batch_size: int = 64,
              lr: float = 5e-5,
              memory_size: int = 2000,
              device: str = "cpu",
              verbose: bool = True) -> ContinualResultsMatrix:
    """Lance iCaRL sur une séquence de tâches."""
    device_t = torch.device(device)
    model    = model.to(device_t)
    trainer  = iCaRLTrainer(model, memory_size=memory_size)
    n_tasks  = len(task_sequence)
    matrix   = ContinualResultsMatrix(n_tasks)

    for task_i, task_ds in enumerate(task_sequence):
        if verbose:
            print(f"\n[iCaRL] Tâche {task_i+1}/{n_tasks} | "
                  f"classes: {sorted(np.unique(task_ds.y).tolist())} | "
                  f"samples: {len(task_ds)}")

        trainer.train_task(
            task_ds.X, task_ds.y, device_t,
            n_epochs=n_epochs_per_task,
            lr=lr, batch_size=batch_size)

        # Évaluation NMC sur toutes les tâches vues
        for task_j in range(task_i + 1):
            f1 = _eval_nmc(trainer, test_datasets[task_j], device_t)
            matrix.record(task_i, task_j, f1)
            if verbose:
                print(f"  eval tâche {task_j}: F1={f1:.4f}")

    if verbose:
        print("\n" + matrix.summary())
    return matrix


def _eval_nmc(trainer: iCaRLTrainer, ds, device: torch.device) -> float:
    if not trainer.model.har_head.prototype_memory.prototypes:
        return 0.0
    trainer.model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in ds.dataloader(batch_size=128, shuffle=False):
            p = trainer.predict(X, device).cpu().numpy()
            preds.append(p); labels.append(y.numpy())
    return macro_f1(np.concatenate(labels), np.concatenate(preds))
