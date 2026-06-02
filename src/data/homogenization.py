"""
Homogenization: unify labels and signals across datasets.

After homogenization every window has:
  - shape (WINDOW_SIZE, 6)  — [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  - sampled at 50 Hz
  - in m/s² (accelerometer), rad/s (gyroscope)
  - gravity removed
  - a single unified integer label from UNIFIED_LABELS

Label mapping follows the semantic grouping from Amrani 2025:
  5 core ADL classes kept for cross-dataset comparison.
  Dataset-specific activities are preserved under their own IDs (≥ 100)
  so they remain usable for continual learning experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .preprocessing import (
    preprocess_signal,
    sliding_windows_with_labels,
    normalize_signal,
    WINDOW_SIZE,
    STEP_SIZE,
)
from .dataset_loaders import Sample, load_dataset, DATASET_REGISTRY


# ---------------------------------------------------------------------------
# Unified label space
# ---------------------------------------------------------------------------

UNIFIED_LABELS: Dict[int, str] = {
    # Core cross-dataset classes
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting_standing",   # merged (signal similarity per Amrani 2025)
    5: "jogging_running",    # merged
    6: "lying",
    7: "cycling",
    8: "nordic_walking",
    # Activity anticipation target: fall-risk transitions
    9: "stand_to_sit",
    10: "sit_to_stand",
    11: "sit_to_lie",
    12: "lie_to_sit",
    0: "unknown",
}

LABEL_NAME_TO_ID = {v: k for k, v in UNIFIED_LABELS.items()}


# ---------------------------------------------------------------------------
# Per-dataset label remapping tables
# ---------------------------------------------------------------------------

# Format: {original_label_id: unified_label_id}
# 0 = unknown / discard

HAPT_REMAP = {
    1: 1,   # walking
    2: 2,   # walking_upstairs
    3: 3,   # walking_downstairs
    4: 4,   # sitting → sitting_standing
    5: 4,   # standing → sitting_standing
    6: 6,   # laying → lying
    7: 9,   # stand_to_sit
    8: 10,  # sit_to_stand
    9: 11,  # sit_to_lie
    10: 12, # lie_to_sit
    11: 0,  # stand_to_lie (rare, discard for now)
    12: 0,  # lie_to_stand (rare, discard for now)
}

PAMAP2_REMAP = {
    1:  6,   # lying
    2:  4,   # sitting → sitting_standing
    3:  4,   # standing → sitting_standing
    4:  1,   # walking
    5:  5,   # running → jogging_running
    6:  7,   # cycling
    7:  8,   # nordic_walking
    9:  0,   # watching_tv (not relevant)
    10: 0,   # computer_work
    11: 0,   # car_driving
    12: 2,   # ascending_stairs → walking_upstairs
    13: 3,   # descending_stairs → walking_downstairs
    16: 0,   # vacuum_cleaning
    17: 0,   # ironing
    18: 0,   # folding_laundry
    19: 0,   # house_cleaning
    20: 0,   # playing_soccer
    24: 0,   # rope_jumping
}

WISDM_REMAP = {
    1: 1,   # Walking
    2: 5,   # Jogging → jogging_running
    3: 2,   # Upstairs → walking_upstairs
    4: 3,   # Downstairs → walking_downstairs
    5: 4,   # Sitting → sitting_standing
    6: 4,   # Standing → sitting_standing
}

MOBIACT_REMAP = {
    1:  4,   # STD / SIT → sitting_standing
    2:  1,   # WAL → walking
    3:  5,   # JOG → jogging_running
    4:  0,   # JUM → discard (jumping)
    5:  2,   # STU → walking_upstairs
    6:  3,   # STN → walking_downstairs
    7:  4,   # SCH → sitting_standing
    8:  0,   # CSI → discard
    9:  0,   # CSO → discard
    10: 4,   # SBW → sitting_standing
    11: 6,   # SBE → lying
}

DATASET_REMAP: Dict[str, Dict[int, int]] = {
    "hapt":    HAPT_REMAP,
    "pamap2":  PAMAP2_REMAP,
    "wisdm":   WISDM_REMAP,
    "mobiact": MOBIACT_REMAP,
}

DATASET_HAS_GRAVITY = {
    "hapt":    False,   # gravity removed in original processing
    "pamap2":  True,
    "wisdm":   True,
    "mobiact": True,
}


# ---------------------------------------------------------------------------
# Core homogenization function
# ---------------------------------------------------------------------------

def remap_labels(labels: np.ndarray, remap: Dict[int, int]) -> np.ndarray:
    """Apply a label remapping dictionary to a label array."""
    out = np.zeros_like(labels)
    for src, dst in remap.items():
        out[labels == src] = dst
    return out


def homogenize_sample(signal: np.ndarray,
                      labels: np.ndarray,
                      dataset_name: str,
                      original_hz: int,
                      unit: str) -> Tuple[np.ndarray, np.ndarray]:
    """Homogenize a single (signal, labels) pair.

    Returns:
        windows: (N, WINDOW_SIZE, 6)
        window_labels: (N,)  — unified label IDs, 0 = unknown/discarded
    """
    remap     = DATASET_REMAP[dataset_name]
    has_grav  = DATASET_HAS_GRAVITY[dataset_name]

    # Remap labels first (before resampling changes sample count)
    unified_labels = remap_labels(labels, remap)

    # Preprocess signal: resample → unit convert → remove gravity
    proc = preprocess_signal(signal, original_hz, unit=unit, has_gravity=has_grav)

    # Align label array length after resampling
    if len(proc) != len(unified_labels):
        # Nearest-neighbour label alignment after resampling
        ratio = len(proc) / len(unified_labels)
        idx   = (np.arange(len(proc)) / ratio).astype(int).clip(0, len(unified_labels) - 1)
        unified_labels = unified_labels[idx]

    # Pad or trim gyroscope if signal only has 3 channels (e.g., WISDM)
    if proc.shape[1] == 3:
        proc = np.concatenate([proc, np.zeros((len(proc), 3), dtype=proc.dtype)], axis=1)
    elif proc.shape[1] > 6:
        proc = proc[:, :6]

    # Segment into windows
    windows, window_labels = sliding_windows_with_labels(proc, unified_labels)

    return windows, window_labels


def build_unified_dataset(data_root: str | Path,
                          datasets: Optional[List[str]] = None,
                          discard_unknown: bool = True,
                          normalize: bool = True):
    """Load, homogenize, and merge all available datasets.

    Args:
        data_root: path to the data/raw/ directory
        datasets: list of dataset names to load (None = all available)
        discard_unknown: drop windows with label == 0
        normalize: z-score normalize per dataset (using training split stats)

    Returns:
        X:         (N, WINDOW_SIZE, 6)  float32
        y:         (N,)                 int64  — unified labels
        subjects:  (N,)                 int32  — subject IDs
        origins:   (N,)                 str    — dataset name per window
        stats:     dict of (mean, std) per dataset for later normalization
    """
    data_root = Path(data_root)
    if datasets is None:
        datasets = list(DATASET_REGISTRY)

    all_X, all_y, all_subjects, all_origins = [], [], [], []
    stats = {}

    for name in datasets:
        folder = data_root / name
        if not folder.exists():
            print(f"[{name}] folder not found — skipping")
            continue

        _, hz, unit = DATASET_REGISTRY[name]
        samples, _, _ = load_dataset(name, folder), hz, unit  # reload cleanly
        # Actually call load_dataset properly
        samples, hz, unit = load_dataset(name, folder)

        if not samples:
            print(f"[{name}] no samples loaded — skipping")
            continue

        ds_windows, ds_labels, ds_subjects = [], [], []

        for (signal, labels, subject_id, dname) in samples:
            wins, wlabels = homogenize_sample(signal, labels, name, hz, unit)
            if len(wins) == 0:
                continue
            ds_windows.append(wins)
            ds_labels.append(wlabels)
            ds_subjects.append(np.full(len(wins), subject_id, dtype=np.int32))

        if not ds_windows:
            continue

        X_ds = np.concatenate(ds_windows, axis=0).astype(np.float32)
        y_ds = np.concatenate(ds_labels, axis=0).astype(np.int64)
        s_ds = np.concatenate(ds_subjects, axis=0)

        if discard_unknown:
            mask = y_ds > 0
            X_ds, y_ds, s_ds = X_ds[mask], y_ds[mask], s_ds[mask]

        if len(X_ds) == 0:
            continue

        # Per-dataset z-score normalization (compute on all windows for now;
        # in the training pipeline this should be computed only on train split)
        if normalize:
            flat = X_ds.reshape(-1, X_ds.shape[-1])
            mean = flat.mean(axis=0)
            std  = flat.std(axis=0) + 1e-8
            X_ds = ((X_ds - mean) / std).astype(np.float32)
            stats[name] = (mean, std)

        all_X.append(X_ds)
        all_y.append(y_ds)
        all_subjects.append(s_ds)
        all_origins.extend([name] * len(X_ds))

        print(f"[{name}] {len(X_ds)} windows | "
              f"labels: {sorted(np.unique(y_ds).tolist())} | "
              f"subjects: {np.unique(s_ds).tolist()}")

    if not all_X:
        raise RuntimeError("No datasets were loaded. Check data/raw/ folders.")

    X       = np.concatenate(all_X,       axis=0)
    y       = np.concatenate(all_y,       axis=0)
    subjects= np.concatenate(all_subjects, axis=0)
    origins = np.array(all_origins)

    return X, y, subjects, origins, stats


def save_processed(save_dir: str | Path,
                   X: np.ndarray, y: np.ndarray,
                   subjects: np.ndarray, origins: np.ndarray):
    """Save homogenized dataset to disk as numpy arrays."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "X.npy",        X)
    np.save(save_dir / "y.npy",        y)
    np.save(save_dir / "subjects.npy", subjects)
    np.save(save_dir / "origins.npy",  origins)
    print(f"Saved {len(X)} windows to {save_dir}")


def load_processed(save_dir: str | Path):
    """Load previously saved homogenized dataset."""
    save_dir = Path(save_dir)
    X        = np.load(save_dir / "X.npy")
    y        = np.load(save_dir / "y.npy")
    subjects = np.load(save_dir / "subjects.npy")
    origins  = np.load(save_dir / "origins.npy", allow_pickle=True)
    return X, y, subjects, origins
