"""
Experience Replay Buffer for continual HAR (anti-catastrophic forgetting).

Uses reservoir sampling to maintain a fixed-size buffer of past windows
that is class-balanced. During continual training, a replay batch is mixed
into every training step so the model never fully forgets old classes.

Reference: LAPNet-HAR (Adaimi & Thomaz, 2022) — rehearsal-based CL for HAR.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size ring buffer with class-balanced reservoir sampling.

    Args:
        capacity:         total number of windows the buffer holds
        n_classes:        number of activity classes (for balanced allocation)
        window_size:      T — timesteps per window (default 150)
        n_channels:       C — IMU channels (default 6)
        balanced:         if True, allocate capacity equally per class
    """

    def __init__(
        self,
        capacity:    int = 2000,
        n_classes:   int = 12,
        window_size: int = 150,
        n_channels:  int = 6,
        balanced:    bool = True,
    ):
        self.capacity    = capacity
        self.n_classes   = n_classes
        self.window_size = window_size
        self.n_channels  = n_channels
        self.balanced    = balanced

        # Per-class capacity when balanced
        self.per_class   = capacity // n_classes if balanced else capacity

        # Storage — pre-allocated for efficiency
        self._X:      np.ndarray = np.empty((0, window_size, n_channels), dtype=np.float32)
        self._y:      np.ndarray = np.empty((0,), dtype=np.int64)
        self._counts: dict       = {}  # class_id → number of samples stored

        # Global sample counter for reservoir sampling
        self._total_seen: int = 0

    # ------------------------------------------------------------------
    # Adding samples
    # ------------------------------------------------------------------

    def add_batch(self, X: np.ndarray, y: np.ndarray):
        """Add a batch of windows to the buffer.

        Uses reservoir sampling within each class when balanced=True,
        otherwise global reservoir sampling.

        Args:
            X: (N, T, C) windows
            y: (N,)      labels
        """
        if self.balanced:
            self._add_balanced(X, y)
        else:
            self._add_reservoir(X, y)

    def _add_balanced(self, X: np.ndarray, y: np.ndarray):
        """Class-balanced reservoir sampling."""
        unique_classes = np.unique(y)
        for cls in unique_classes:
            mask  = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
            self._add_class_reservoir(X_cls, y_cls, int(cls))

    def _add_class_reservoir(self, X: np.ndarray, y: np.ndarray, cls_id: int):
        """Reservoir sampling for a single class."""
        rng     = np.random.default_rng()
        n_new   = len(X)
        n_stored = self._counts.get(cls_id, 0)

        # Indices of existing samples for this class
        existing_mask = self._y == cls_id
        X_existing = self._X[existing_mask]
        y_existing = self._y[existing_mask]

        # Merge existing + new
        X_all = np.concatenate([X_existing, X], axis=0)
        y_all = np.concatenate([y_existing, y], axis=0)

        # Subsample to per_class capacity
        if len(X_all) > self.per_class:
            idx   = rng.choice(len(X_all), self.per_class, replace=False)
            X_all = X_all[idx]
            y_all = y_all[idx]

        # Remove old entries for this class and add updated ones
        keep_mask = ~existing_mask
        self._X = np.concatenate([self._X[keep_mask], X_all], axis=0)
        self._y = np.concatenate([self._y[keep_mask], y_all], axis=0)
        self._counts[cls_id] = len(X_all)

    def _add_reservoir(self, X: np.ndarray, y: np.ndarray):
        """Global reservoir sampling (no class balancing)."""
        rng = np.random.default_rng()
        for i in range(len(X)):
            self._total_seen += 1
            if len(self._X) < self.capacity:
                self._X = np.concatenate([self._X, X[i:i+1]], axis=0)
                self._y = np.concatenate([self._y, y[i:i+1]], axis=0)
            else:
                j = rng.integers(0, self._total_seen)
                if j < self.capacity:
                    self._X[j] = X[i]
                    self._y[j] = y[i]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, as_tensor: bool = True,
               device: Optional[torch.device] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n random windows from the buffer.

        Args:
            n:         number of windows to sample
            as_tensor: if True, return torch Tensors; else numpy arrays
            device:    target device for tensors

        Returns:
            X_replay: (n, T, C)
            y_replay: (n,)
        """
        if len(self) == 0:
            raise RuntimeError("Replay buffer is empty.")

        n = min(n, len(self))
        idx = np.random.choice(len(self), n, replace=False)

        X_sample = self._X[idx]
        y_sample = self._y[idx]

        if as_tensor:
            X_t = torch.from_numpy(X_sample)
            y_t = torch.from_numpy(y_sample)
            if device is not None:
                X_t = X_t.to(device)
                y_t = y_t.to(device)
            return X_t, y_t

        return X_sample, y_sample

    def sample_by_class(self, n_per_class: int, as_tensor: bool = True,
                         device: Optional[torch.device] = None):
        """Sample n_per_class windows from each class in the buffer."""
        X_parts, y_parts = [], []
        for cls_id in self.known_classes():
            mask = self._y == cls_id
            X_cls, y_cls = self._X[mask], self._y[mask]
            n = min(n_per_class, len(X_cls))
            idx = np.random.choice(len(X_cls), n, replace=False)
            X_parts.append(X_cls[idx])
            y_parts.append(y_cls[idx])

        if not X_parts:
            raise RuntimeError("Buffer is empty.")

        X_all = np.concatenate(X_parts, axis=0)
        y_all = np.concatenate(y_parts, axis=0)

        if as_tensor:
            X_t = torch.from_numpy(X_all)
            y_t = torch.from_numpy(y_all)
            if device is not None:
                X_t = X_t.to(device)
                y_t = y_t.to(device)
            return X_t, y_t

        return X_all, y_all

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._X)

    def known_classes(self):
        return sorted(self._counts.keys())

    def class_counts(self) -> dict:
        return dict(self._counts)

    def summary(self) -> str:
        lines = [f"ReplayBuffer — {len(self)}/{self.capacity} windows"]
        for cls_id in self.known_classes():
            lines.append(f"  class {cls_id}: {self._counts[cls_id]} samples")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        np.savez(path, X=self._X, y=self._y,
                 counts=np.array(list(self._counts.items()), dtype=object))

    @classmethod
    def load(cls, path: str, **kwargs) -> "ReplayBuffer":
        data    = np.load(path, allow_pickle=True)
        buf     = cls(**kwargs)
        buf._X  = data["X"]
        buf._y  = data["y"]
        buf._counts = dict(data["counts"].tolist())
        return buf
