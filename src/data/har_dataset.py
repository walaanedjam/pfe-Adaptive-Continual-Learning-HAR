"""
PyTorch-compatible Dataset for HAR.

Supports three continual learning scenarios:
  - 'standard'    : full dataset, random split
  - 'user_incr'   : split by subject → tasks arrive as new users
  - 'class_incr'  : split by activity class → tasks arrive as new classes
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


from .homogenization import UNIFIED_LABELS


class HARDataset:
    """Holds a slice of the homogenized HAR dataset.

    Works with or without PyTorch (returns numpy arrays when torch is absent).

    Args:
        X: (N, T, C) windows
        y: (N,)      labels
        subjects: (N,) subject IDs
        origins: (N,) dataset names
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 subjects: np.ndarray,
                 origins: np.ndarray):
        self.X        = X.astype(np.float32)
        self.y        = y.astype(np.int64)
        self.subjects = subjects
        self.origins  = origins

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        if TORCH_AVAILABLE:
            import torch
            return (torch.from_numpy(self.X[idx]),
                    torch.tensor(self.y[idx], dtype=torch.long))
        return self.X[idx], self.y[idx]

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def train_test_split(self, test_ratio: float = 0.2, seed: int = 42):
        """Split dataset into train/test.

        If more than one subject exists: split by subject (no leakage).
        If only one subject (e.g. single-user task): split by sample index.
        """
        rng = np.random.default_rng(seed)
        unique_subjects = np.unique(self.subjects)

        if len(unique_subjects) > 1:
            rng.shuffle(unique_subjects)
            n_test        = max(1, int(len(unique_subjects) * test_ratio))
            test_subjects = set(unique_subjects[:n_test])
            train_mask    = np.array([s not in test_subjects for s in self.subjects])
        else:
            # Single subject — split by sample index
            idx       = np.arange(len(self.X))
            rng.shuffle(idx)
            n_test    = max(1, int(len(idx) * test_ratio))
            test_idx  = set(idx[:n_test])
            train_mask = np.array([i not in test_idx for i in range(len(self.X))])

        return self._subset(train_mask), self._subset(~train_mask)

    def get_task_by_user(self, subject_id: int) -> "HARDataset":
        """Return a dataset slice for a single subject."""
        mask = self.subjects == subject_id
        return self._subset(mask)

    def get_task_by_class(self, label_ids: List[int]) -> "HARDataset":
        """Return a dataset slice for a set of activity classes."""
        mask = np.isin(self.y, label_ids)
        return self._subset(mask)

    def _subset(self, mask: np.ndarray) -> "HARDataset":
        return HARDataset(
            self.X[mask], self.y[mask],
            self.subjects[mask], self.origins[mask],
        )

    # ------------------------------------------------------------------
    # Continual learning task sequences
    # ------------------------------------------------------------------

    def user_incremental_tasks(self) -> List["HARDataset"]:
        """One task per subject, ordered by subject ID."""
        return [self.get_task_by_user(s) for s in sorted(np.unique(self.subjects))]

    def class_incremental_tasks(self,
                                 classes_per_task: int = 2,
                                 seed: int = 42) -> List["HARDataset"]:
        """Group activity classes into tasks of `classes_per_task` classes each."""
        rng = np.random.default_rng(seed)
        unique_classes = np.unique(self.y)
        rng.shuffle(unique_classes)

        tasks = []
        for i in range(0, len(unique_classes), classes_per_task):
            batch = unique_classes[i: i + classes_per_task].tolist()
            task  = self.get_task_by_class(batch)
            if len(task) > 0:
                tasks.append(task)
        return tasks

    # ------------------------------------------------------------------
    # Dataloader (requires PyTorch)
    # ------------------------------------------------------------------

    def dataloader(self, batch_size: int = 64,
                   shuffle: bool = True,
                   num_workers: int = 0):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install torch to use dataloaders.")
        from torch.utils.data import DataLoader
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          pin_memory=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        label_counts = {
            UNIFIED_LABELS.get(int(c), str(c)): int((self.y == c).sum())
            for c in np.unique(self.y)
        }
        lines = [
            f"HARDataset — {len(self)} windows",
            f"  Shape:    {self.X.shape}",
            f"  Subjects: {sorted(np.unique(self.subjects).tolist())}",
            f"  Origins:  {sorted(np.unique(self.origins).tolist())}",
            "  Classes:",
        ]
        for name, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {name:<25s}: {count:>6d}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory from processed files
# ---------------------------------------------------------------------------

def dataset_from_files(processed_dir: str) -> HARDataset:
    from .homogenization import load_processed
    X, y, subjects, origins = load_processed(processed_dir)
    return HARDataset(X, y, subjects, origins)


def dataset_from_arrays(X, y, subjects, origins) -> HARDataset:
    return HARDataset(X, y, subjects, origins)
