"""
Dataset for activity anticipation training.

For each consecutive pair of windows (w_t, w_{t+1}):
  - Input:  first p% of w_t  →  partial observation
  - Target: label of w_{t+1} →  next activity to predict

We build sequences of S=5 consecutive partial windows as context,
letting the LSTM in the anticipation head model temporal progression.

Observation ratios: p ∈ {0.25, 0.50, 0.75}
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except (ImportError, OSError):
    TORCH_OK = False


class AnticipationDataset:
    """
    Builds (context_windows, next_label) pairs from a HARDataset.

    For every window at position t, if the next window t+1 has a
    DIFFERENT label (i.e. an activity transition is imminent), we
    create a training sample:
      - context: S partial windows ending at t, each truncated to p*T samples
      - target:  label of window t+1

    This teaches the model to anticipate transitions from context.

    Args:
        X:             (N, T, C) windows
        y:             (N,)     labels
        obs_ratio:     fraction of each window to observe (0.25, 0.50, 0.75)
        seq_len:       S — number of consecutive windows as context
        transitions_only: if True, only keep samples where label changes
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 obs_ratio: float = 0.50,
                 seq_len: int = 5,
                 transitions_only: bool = False):

        self.obs_ratio = obs_ratio
        self.seq_len   = seq_len
        T              = X.shape[1]
        self.obs_len   = max(1, int(T * obs_ratio))

        contexts, targets = [], []

        # Slide over all valid positions
        for i in range(seq_len, len(X) - 1):
            # Context: seq_len consecutive windows, each truncated
            ctx_wins = X[i - seq_len: i]          # (S, T, C)
            ctx_trunc = ctx_wins[:, :self.obs_len, :]  # (S, obs_len, C)

            next_label = int(y[i])                 # label of next window

            if transitions_only:
                current_label = int(y[i - 1])
                if next_label == current_label:
                    continue

            contexts.append(ctx_trunc)
            targets.append(next_label)

        self.X = np.array(contexts, dtype=np.float32)  # (N, S, obs_len, C)
        self.y = np.array(targets,  dtype=np.int64)     # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        if TORCH_OK:
            return (torch.from_numpy(self.X[idx]),
                    torch.tensor(self.y[idx], dtype=torch.long))
        return self.X[idx], self.y[idx]

    def dataloader(self, batch_size: int = 64, shuffle: bool = True):
        if not TORCH_OK:
            raise RuntimeError("PyTorch not available.")
        from torch.utils.data import DataLoader
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    def summary(self) -> str:
        unique, counts = np.unique(self.y, return_counts=True)
        lines = [
            f"AnticipationDataset — {len(self)} samples",
            f"  obs_ratio={self.obs_ratio}  seq_len={self.seq_len}",
            f"  context shape: {self.X.shape}",
            f"  classes: {dict(zip(unique.tolist(), counts.tolist()))}",
        ]
        return "\n".join(lines)


def build_anticipation_datasets(X: np.ndarray,
                                 y: np.ndarray,
                                 test_ratio: float = 0.2,
                                 seq_len: int = 5,
                                 seed: int = 42,
                                 transitions_only: bool = True):
    """
    Build train/val anticipation datasets for all three obs ratios.

    Returns:
        {0.25: (train_ds, val_ds), 0.50: ..., 0.75: ...}
    """
    rng    = np.random.default_rng(seed)
    N      = len(X)
    idx    = np.arange(N)
    rng.shuffle(idx)
    n_val  = int(N * test_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    result = {}
    for ratio in [0.25, 0.50, 0.75]:
        train_ds = AnticipationDataset(X_train, y_train,
                                        obs_ratio=ratio, seq_len=seq_len,
                                        transitions_only=transitions_only)
        val_ds   = AnticipationDataset(X_val,   y_val,
                                        obs_ratio=ratio, seq_len=seq_len,
                                        transitions_only=transitions_only)
        result[ratio] = (train_ds, val_ds)

    return result
