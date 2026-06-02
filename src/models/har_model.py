"""
Full HAR Model — two-headed architecture:

  Backbone (IMUTransformerEncoder)
    │
    ├─► Module 1: ContinualHARHead (prototype memory + softmax for pre-training)
    └─► Module 2: AnticipationHead  (predict next activity from partial window)

Usage modes:
  1. Pre-training:   use standard CrossEntropyLoss on the softmax head
  2. Continual:      switch to prototype memory for online class-incremental learning
  3. Anticipation:   forward partial windows through the anticipation head
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone              import IMUTransformerEncoder, build_backbone
from .prototype_memory      import PrototypeMemory
from .replay_buffer         import ReplayBuffer
from .uncertainty_replay    import UncertaintyWeightedReplayBuffer


# ---------------------------------------------------------------------------
# Module 1: Continual HAR Head
# ---------------------------------------------------------------------------

class ContinualHARHead(nn.Module):
    """
    Combines a linear softmax classifier (for supervised pre-training) with
    a PrototypeMemory (for online continual learning).

    During pre-training: use forward() → CrossEntropyLoss
    During continual:   use prototype_memory.predict() → nearest-mean
    """

    def __init__(self, d_model: int = 128, n_classes: int = 13):
        super().__init__()
        self.d_model   = d_model
        self.n_classes = n_classes

        # Linear classifier for supervised pre-training
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

        # Prototype memory for continual inference
        self.prototype_memory = PrototypeMemory(d_model=d_model)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Standard classification forward (pre-training).

        Returns:
            (B, n_classes) logits
        """
        return self.classifier(embeddings)

    def update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.prototype_memory.update(embeddings, labels)

    def continual_predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.prototype_memory.predict(embeddings)

    def contrastive_loss(self, embeddings: torch.Tensor,
                          labels: torch.Tensor, temperature: float = 0.07):
        return self.prototype_memory.contrastive_loss(embeddings, labels, temperature)


# ---------------------------------------------------------------------------
# Module 2: Activity Anticipation Head
# ---------------------------------------------------------------------------

class AnticipationHead(nn.Module):
    """
    Predict the next activity class from a partial window.

    Architecture:
      partial window (B, T_partial, C)
        → shared backbone (produces embedding)
        → LSTM temporal predictor  (models progression of activity sequence)
        → Linear → softmax over activity classes

    During training:
      - Input: first p% of a complete window (p ∈ {25, 50, 75})
      - Target: the majority-label of the *next* window
      - Loss: CrossEntropyLoss

    At inference (real-time):
      - Feed partial signal as it accumulates
      - Predict and update every step_size samples
    """

    def __init__(self, d_model: int = 128, n_classes: int = 13,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Linear(lstm_hidden, n_classes),
        )

        # Learnable mixture: how much to weight the backbone embedding
        # vs. the LSTM context for the final prediction
        self.context_gate = nn.Linear(d_model + lstm_hidden, 1)

    def forward(self, embedding_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding_seq: (B, S, d_model)  — sequence of S window embeddings
                           (embed each partial window before calling this)

        Returns:
            (B, n_classes) — logits for the next activity
        """
        lstm_out, _ = self.lstm(embedding_seq)   # (B, S, lstm_hidden)
        last        = lstm_out[:, -1]             # (B, lstm_hidden)
        return self.classifier(last)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class HARContinualModel(nn.Module):
    """
    Full two-headed model for continual HAR + activity anticipation.

    Args:
        backbone_cfg:    dict of backbone hyperparameters (see build_backbone)
        n_classes:       number of activity classes in the initial training set
        replay_capacity: size of the experience replay buffer
        contrastive_weight: weight of contrastive loss relative to CE loss
    """

    def __init__(
        self,
        backbone_cfg:        Optional[Dict] = None,
        n_classes:           int   = 13,
        replay_capacity:     int   = 2000,
        contrastive_weight:  float = 0.1,
        uncertainty_replay:  bool  = False,
        uncertainty_alpha:   float = 1.0,
    ):
        super().__init__()

        self.backbone            = build_backbone(backbone_cfg)
        d_model                  = self.backbone.d_model
        self.har_head            = ContinualHARHead(d_model, n_classes)
        self.anticipation_head   = AnticipationHead(d_model, n_classes)
        self.contrastive_weight  = contrastive_weight
        self.uncertainty_replay  = uncertainty_replay

        BufferClass = UncertaintyWeightedReplayBuffer if uncertainty_replay \
                      else ReplayBuffer
        kwargs = dict(alpha=uncertainty_alpha) if uncertainty_replay else {}
        self.replay_buffer = BufferClass(
            capacity=replay_capacity,
            n_classes=n_classes,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard supervised forward.

        Args:
            x: (B, T, C) — full windows

        Returns:
            (B, n_classes) logits
        """
        embeddings = self.backbone(x)
        return self.har_head(embeddings)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone embeddings only. Used for prototype updates."""
        return self.backbone(x)

    def anticipate(self, window_seq: torch.Tensor) -> torch.Tensor:
        """Predict next activity from a sequence of partial-window embeddings.

        Args:
            window_seq: (B, S, T_partial, C) — S partial windows per sample

        Returns:
            (B, n_classes) logits
        """
        B, S, T, C = window_seq.shape
        # Embed each partial window individually
        flat       = window_seq.view(B * S, T, C)
        emb_flat   = self.backbone(flat)               # (B*S, d_model)
        emb_seq    = emb_flat.view(B, S, -1)           # (B, S, d_model)
        return self.anticipation_head(emb_seq)

    # ------------------------------------------------------------------
    # Continual learning step
    # ------------------------------------------------------------------

    def continual_step(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        optimizer: torch.optim.Optimizer,
                        replay_batch_size: int = 32,
                        device: torch.device = None) -> Dict[str, float]:
        """
        One continual learning update step.

        Combines:
          1. CE loss on current batch
          2. CE loss on replay batch (anti-forgetting)
          3. Contrastive loss on current embeddings (inter-class separation)

        Returns dict of individual loss values for logging.
        """
        if device is None:
            device = next(self.parameters()).device

        self.train()
        optimizer.zero_grad()

        # ---- Current batch ----
        emb_curr    = self.backbone(x)
        logits_curr = self.har_head(emb_curr)
        loss_ce     = F.cross_entropy(logits_curr, y)

        # ---- Replay batch (if buffer has data) ----
        loss_replay = torch.tensor(0.0, device=device)
        if len(self.replay_buffer) >= replay_batch_size:
            if self.uncertainty_replay:
                rx, ry = self.replay_buffer.sample_uncertain(
                    self, replay_batch_size, device=device)
            else:
                rx, ry = self.replay_buffer.sample(replay_batch_size, device=device)
            emb_replay    = self.backbone(rx)
            logits_replay = self.har_head(emb_replay)
            loss_replay   = F.cross_entropy(logits_replay, ry)

        # ---- Contrastive loss on current batch ----
        loss_contrast = self.har_head.contrastive_loss(emb_curr, y)

        total_loss = loss_ce + loss_replay + self.contrastive_weight * loss_contrast
        total_loss.backward()
        optimizer.step()

        # ---- Update prototypes ----
        with torch.no_grad():
            self.har_head.update_prototypes(emb_curr.detach(), y)

        # ---- Add current batch to replay buffer ----
        self.replay_buffer.add_batch(
            x.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
        )

        return {
            "loss_ce":       loss_ce.item(),
            "loss_replay":   loss_replay.item(),
            "loss_contrast": loss_contrast.item(),
            "loss_total":    total_loss.item(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor, mode: str = "prototype") -> torch.Tensor:
        """
        Args:
            x:    (B, T, C)
            mode: 'prototype' (continual) or 'softmax' (pre-training)

        Returns:
            (B,) predicted class IDs
        """
        self.eval()
        emb = self.backbone(x)
        if mode == "prototype":
            return self.har_head.continual_predict(emb)
        else:
            return self.har_head(emb).argmax(dim=-1)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "model_state":    self.state_dict(),
            "prototype_state": self.har_head.prototype_memory.state(),
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "HARContinualModel":
        ckpt  = torch.load(path, map_location="cpu")
        model = cls(**kwargs)
        model.load_state_dict(ckpt["model_state"])
        model.har_head.prototype_memory.load_state(ckpt["prototype_state"])
        return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(n_classes: int = 13,
                d_model: int = 128,
                n_heads: int = 4,
                n_blocks: int = 4,
                replay_capacity: int = 2000,
                contrastive_weight: float = 0.1,
                uncertainty_replay: bool = False,
                uncertainty_alpha: float = 1.0) -> HARContinualModel:
    """Build the full model with sensible defaults."""
    backbone_cfg = dict(d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
                        ffn_dim=d_model * 2, in_channels=6)
    return HARContinualModel(
        backbone_cfg=backbone_cfg,
        n_classes=n_classes,
        replay_capacity=replay_capacity,
        contrastive_weight=contrastive_weight,
        uncertainty_replay=uncertainty_replay,
        uncertainty_alpha=uncertainty_alpha,
    )
