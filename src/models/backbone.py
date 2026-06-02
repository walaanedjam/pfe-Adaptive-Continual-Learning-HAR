"""
Transformer backbone for IMU-based HAR.

Architecture (input shape: B × T × C):
  1. Channel projection: C → d_model
  2. Learnable positional encoding
  3. N × Transformer encoder blocks (self-attention + FFN)
  4. CLS token pooling → d_model embedding vector

Design follows Dirgová et al. (2022) "Wearable sensor-based HAR with Transformer".
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal + learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Fixed sinusoidal basis
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        # Learnable scaling — starts at 1 so training begins from sinusoidal baseline
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.scale * self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block: MultiHeadAttention + FFN with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm (more stable than post-norm for small datasets)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class IMUTransformerEncoder(nn.Module):
    """
    Full Transformer encoder for multivariate IMU time series.

    Input:  (B, T, C)   — batch of windows, T timesteps, C channels
    Output: (B, d_model) — one embedding vector per window (via CLS token)

    Args:
        in_channels: number of input sensor channels (default 6: acc_xyz + gyro_xyz)
        d_model:     internal model dimension (default 128)
        n_heads:     attention heads (default 4, must divide d_model)
        n_blocks:    number of Transformer encoder blocks (default 4)
        ffn_dim:     feed-forward hidden size (default 256 = 2 × d_model)
        dropout:     dropout probability
        max_len:     maximum sequence length (≥ WINDOW_SIZE = 150)
    """

    def __init__(
        self,
        in_channels: int = 6,
        d_model:     int = 128,
        n_heads:     int = 4,
        n_blocks:    int = 4,
        ffn_dim:     int = 256,
        dropout:     float = 0.1,
        max_len:     int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        # CLS token — prepended to each sequence before attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Project raw channels to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 1, dropout=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — windowed IMU signal

        Returns:
            (B, d_model) — CLS embedding
        """
        B = x.size(0)

        # Project input channels to d_model
        x = self.input_proj(x)                     # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)           # (B, T+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return CLS token embedding
        return x[:, 0]                             # (B, d_model)

    def get_attention_weights(self, x: torch.Tensor):
        """Forward pass that also returns attention weights from each block.
        Useful for visualizing which timesteps the model attends to."""
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_enc(x)

        attn_weights = []
        for block in self.blocks:
            normed = block.norm1(x)
            _, w = block.attn(normed, normed, normed, need_weights=True, average_attn_weights=True)
            attn_weights.append(w.detach().cpu())
            x = block(x)

        return self.norm(x)[:, 0], attn_weights


def build_backbone(cfg: dict = None) -> IMUTransformerEncoder:
    """Convenience factory — accepts a config dict or uses defaults."""
    defaults = dict(
        in_channels=6,
        d_model=128,
        n_heads=4,
        n_blocks=4,
        ffn_dim=256,
        dropout=0.1,
    )
    if cfg:
        defaults.update(cfg)
    return IMUTransformerEncoder(**defaults)
