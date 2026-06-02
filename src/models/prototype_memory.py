"""
Prototype Memory for continual HAR (Module 1).

Inspired by LAPNet-HAR (Adaimi & Thomaz, 2022).

Each activity class is represented by a single prototype vector —
the running mean of all embeddings seen for that class.
Classification is nearest-mean in the embedding space.

Key properties:
  - Fixed-capacity prediction head: adding new classes doesn't change
    the backbone or require softmax re-training
  - Prototypes evolve online with new data (no full retraining needed)
  - Contrastive loss enforces inter-class separation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemory(nn.Module):
    """
    Manages class prototype vectors and performs nearest-mean classification.

    Args:
        d_model: embedding dimension (must match backbone output)
        n_classes: maximum number of classes (can grow dynamically)
        distance: 'euclidean' or 'cosine'
    """

    def __init__(self, d_model: int = 128, distance: str = "euclidean"):
        super().__init__()
        self.d_model   = d_model
        self.distance  = distance

        # Registered as buffers so they are saved with state_dict but not trained
        # prototypes[c] = mean embedding vector for class c
        self.prototypes: Dict[int, torch.Tensor] = {}
        # Running count for online mean update
        self.counts:     Dict[int, int]          = {}

    # ------------------------------------------------------------------
    # Prototype updates
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Update prototypes with a batch of embeddings.

        Uses online (Welford) mean update — no need to store all past embeddings.

        Args:
            embeddings: (B, d_model)
            labels:     (B,) — integer class labels
        """
        for cls_id in labels.unique():
            cls_id = int(cls_id.item())
            mask   = labels == cls_id
            batch_mean = embeddings[mask].mean(dim=0).cpu()

            if cls_id not in self.prototypes:
                self.prototypes[cls_id] = batch_mean.clone()
                self.counts[cls_id]     = int(mask.sum().item())
            else:
                n_old  = self.counts[cls_id]
                n_new  = int(mask.sum().item())
                n_total = n_old + n_new
                # Weighted running mean
                self.prototypes[cls_id] = (
                    self.prototypes[cls_id] * n_old + batch_mean * n_new
                ) / n_total
                self.counts[cls_id] = n_total

    @torch.no_grad()
    def adapt_prototypes(self, backbone: nn.Module,
                          replay_X: torch.Tensor,
                          replay_y: torch.Tensor,
                          device: torch.device):
        """Recompute prototypes from replay buffer embeddings.

        Called after backbone updates to prevent prototype obsolescence
        (the embedding space shifts, so old prototypes become stale).

        Args:
            backbone:  the current backbone model
            replay_X:  (N, T, C) replay window tensors
            replay_y:  (N,) labels
            device:    computation device
        """
        backbone.eval()
        replay_X = replay_X.to(device)
        replay_y = replay_y.to(device)

        with torch.no_grad():
            embeddings = backbone(replay_X)

        # Reset and recompute from scratch
        self.prototypes = {}
        self.counts     = {}
        self.update(embeddings, replay_y)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _proto_matrix(self, device: torch.device) -> Tuple[torch.Tensor, List[int]]:
        """Stack prototypes into a matrix for batch distance computation."""
        cls_ids = sorted(self.prototypes.keys())
        matrix  = torch.stack([self.prototypes[c] for c in cls_ids]).to(device)
        return matrix, cls_ids

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Nearest-mean classification.

        Args:
            embeddings: (B, d_model)

        Returns:
            (B,) predicted class IDs
        """
        if not self.prototypes:
            raise RuntimeError("PrototypeMemory has no prototypes yet — call update() first.")

        proto_matrix, cls_ids = self._proto_matrix(embeddings.device)
        cls_ids_tensor = torch.tensor(cls_ids, dtype=torch.long, device=embeddings.device)

        if self.distance == "cosine":
            emb_norm   = F.normalize(embeddings, dim=-1)     # (B, d)
            proto_norm = F.normalize(proto_matrix, dim=-1)   # (K, d)
            # Cosine similarity → negate for nearest-min
            scores = emb_norm @ proto_norm.T                 # (B, K)
            nearest = scores.argmax(dim=-1)
        else:  # euclidean
            # (B, 1, d) - (1, K, d) → (B, K, d) → (B, K)
            diffs   = embeddings.unsqueeze(1) - proto_matrix.unsqueeze(0)
            dists   = diffs.pow(2).sum(dim=-1)               # (B, K)
            nearest = dists.argmin(dim=-1)                   # (B,)

        return cls_ids_tensor[nearest]

    def predict_with_distances(self, embeddings: torch.Tensor):
        """Return (predicted_labels, distance_matrix) for confidence estimation."""
        proto_matrix, cls_ids = self._proto_matrix(embeddings.device)
        cls_ids_tensor = torch.tensor(cls_ids, dtype=torch.long, device=embeddings.device)

        diffs  = embeddings.unsqueeze(1) - proto_matrix.unsqueeze(0)
        dists  = diffs.pow(2).sum(dim=-1)
        nearest = dists.argmin(dim=-1)

        return cls_ids_tensor[nearest], dists

    # ------------------------------------------------------------------
    # Contrastive loss
    # ------------------------------------------------------------------

    def contrastive_loss(self, embeddings: torch.Tensor,
                          labels: torch.Tensor,
                          temperature: float = 0.07) -> torch.Tensor:
        """Supervised contrastive loss (Khosla et al., 2020).

        Pulls embeddings of the same class together and pushes different
        classes apart in the embedding space — enforces inter-class separation
        as new classes arrive (key for catastrophic forgetting mitigation).

        Args:
            embeddings:  (B, d_model) — L2-normalized embeddings
            labels:      (B,) — integer class labels
            temperature: softmax temperature

        Returns:
            scalar contrastive loss
        """
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        emb = F.normalize(embeddings, dim=-1)                # (B, d)
        sim = (emb @ emb.T) / temperature                    # (B, B)

        # Mask: 1 where same class, 0 on diagonal
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        diag_mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask  = label_eq & diag_mask

        # If no positive pair in the batch, skip
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Log-softmax over all other samples (excluding self)
        sim = sim.masked_fill(~diag_mask, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Mean log-probability over positive pairs per anchor
        # Use masked_fill instead of multiplication to avoid -inf * 0 = NaN
        log_prob_pos = log_prob.masked_fill(~pos_mask, 0.0)
        loss = -log_prob_pos.sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        return loss.mean()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def known_classes(self) -> List[int]:
        return sorted(self.prototypes.keys())

    def n_classes(self) -> int:
        return len(self.prototypes)

    def state(self) -> dict:
        return {"prototypes": self.prototypes, "counts": self.counts}

    def load_state(self, state: dict):
        self.prototypes = state["prototypes"]
        self.counts     = state["counts"]
