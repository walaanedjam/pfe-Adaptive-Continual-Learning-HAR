"""
Dataset loaders for the four public HAR datasets.

Each loader returns a list of (signal, label_array, subject_id, dataset_name) tuples,
where:
  - signal      : (T, C) float32 numpy array — raw accelerometer (+gyroscope) data
  - label_array : (T,)   int32  numpy array  — per-sample activity label
  - subject_id  : int    — subject identifier within the dataset
  - dataset_name: str    — dataset identifier

Expected folder layout under data/raw/:
  hapt/    → UCI HAPT dataset
  mobiact/ → MobiAct dataset
  pamap2/  → PAMAP2 dataset
  wisdm/   → WISDM dataset

Download links are printed by download_instructions() below.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# (signal, labels, subject_id, dataset_name)
Sample = Tuple[np.ndarray, np.ndarray, int, str]


def download_instructions():
    print("""
=== Dataset Download Instructions ===

1. HAPT (UCI Human Activities and Postural Transitions)
   URL: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
   → extract to  data/raw/hapt/

2. MobiAct
   URL: https://bmi.hmu.gr/the-mobiact-dataset-v2-0/
   → extract to  data/raw/mobiact/

3. PAMAP2
   URL: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
   → extract to  data/raw/pamap2/

4. WISDM (Smartphone and Smartwatch Activity and Biometrics)
   URL: https://www.cis.fordham.edu/wisdm/dataset.php
   → extract to  data/raw/wisdm/
""")


# ---------------------------------------------------------------------------
# HAPT loader
# ---------------------------------------------------------------------------
# Structure after extraction:
#   RawData/
#     acc_expXX_userYY.txt   — triaxial accelerometer  (m/s², 50 Hz)
#     gyro_expXX_userYY.txt  — triaxial gyroscope       (rad/s, 50 Hz)
#   labels.txt — columns: exp_id, user_id, activity_id, start, end

HAPT_LABELS = {
    1: "walking", 2: "walking_upstairs", 3: "walking_downstairs",
    4: "sitting", 5: "standing", 6: "laying",
    7: "stand_to_sit", 8: "sit_to_stand", 9: "sit_to_lie",
    10: "lie_to_sit", 11: "stand_to_lie", 12: "lie_to_stand",
}

HAPT_HZ   = 50
HAPT_UNIT = "ms2"   # already in m/s²


def load_hapt(root: str | Path) -> List[Sample]:
    root = Path(root)
    raw_dir = root / "RawData"
    label_file = root / "RawData" / "labels.txt"

    if not label_file.exists():
        label_file = root / "labels.txt"

    labels_df = pd.read_csv(
        label_file, sep=r"\s+", header=None,
        names=["exp", "user", "activity", "start", "end"],
    )

    samples: List[Sample] = []

    for exp_id in labels_df["exp"].unique():
        user_id = int(labels_df[labels_df["exp"] == exp_id]["user"].iloc[0])
        tag = f"exp{exp_id:02d}_user{user_id:02d}"

        acc_file  = raw_dir / f"acc_{tag}.txt"
        gyro_file = raw_dir / f"gyro_{tag}.txt"

        if not acc_file.exists():
            continue

        acc  = pd.read_csv(acc_file,  sep=r"\s+", header=None).values.astype(np.float32)
        gyro = pd.read_csv(gyro_file, sep=r"\s+", header=None).values.astype(np.float32) \
               if gyro_file.exists() else np.zeros_like(acc)

        signal = np.concatenate([acc, gyro], axis=1)  # (T, 6)
        label_arr = np.zeros(len(signal), dtype=np.int32)

        exp_labels = labels_df[labels_df["exp"] == exp_id]
        for _, row in exp_labels.iterrows():
            s, e = int(row["start"]) - 1, int(row["end"])
            label_arr[s:e] = int(row["activity"])

        # Keep only labelled segments
        mask = label_arr > 0
        if mask.sum() < 150:
            continue

        samples.append((signal[mask], label_arr[mask], user_id, "hapt"))

    return samples


# ---------------------------------------------------------------------------
# PAMAP2 loader
# ---------------------------------------------------------------------------
# Files: subject101.dat … subject109.dat
# Columns: timestamp, activity_id, heart_rate,
#          then groups of (temp, acc1_x, acc1_y, acc1_z, acc2_x, ..., gyro_x, gyro_y, gyro_z, ...)
# We use columns for the hand IMU (wrist): acc2 (cols 21-23) and gyro (cols 27-29) — 100 Hz

PAMAP2_LABELS = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking",
    9: "watching_tv", 10: "computer_work", 11: "car_driving",
    12: "ascending_stairs", 13: "descending_stairs", 16: "vacuum_cleaning",
    17: "ironing", 18: "folding_laundry", 19: "house_cleaning",
    20: "playing_soccer", 24: "rope_jumping",
}

PAMAP2_HZ   = 100
PAMAP2_UNIT = "ms2"  # m/s²


def load_pamap2(root: str | Path) -> List[Sample]:
    root = Path(root)
    protocol_dir = root / "Protocol"
    if not protocol_dir.exists():
        protocol_dir = root

    samples: List[Sample] = []

    for f in sorted(protocol_dir.glob("subject10*.dat")):
        user_id = int(f.stem[-2:])  # subject101 → 1
        df = pd.read_csv(f, sep=r"\s+", header=None)

        activity = df.iloc[:, 1].values.astype(np.int32)

        # wrist IMU: acc (cols 21-23, m/s²) + gyro (cols 27-29, rad/s)
        # Some files use 0-indexed differently; fall back gracefully
        try:
            acc  = df.iloc[:, 21:24].values.astype(np.float32)
            gyro = df.iloc[:, 27:30].values.astype(np.float32)
        except Exception:
            continue

        signal = np.concatenate([acc, gyro], axis=1)  # (T, 6)

        # Replace NaN with linear interpolation
        signal = pd.DataFrame(signal).interpolate(limit_direction="both").values.astype(np.float32)

        mask = activity > 0
        if mask.sum() < 300:
            continue

        samples.append((signal[mask], activity[mask], user_id, "pamap2"))

    return samples


# ---------------------------------------------------------------------------
# WISDM loader
# ---------------------------------------------------------------------------
# File: WISDM_ar_v1.1_raw.txt (comma-separated, sometimes messy trailing ;)
# Columns: user, activity, timestamp, x, y, z
# Activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
# Freq: 20 Hz (phone), we resample to 50 Hz in preprocessing

WISDM_LABEL_MAP = {
    "Walking": 1, "Jogging": 2, "Upstairs": 3,
    "Downstairs": 4, "Sitting": 5, "Standing": 6,
}

WISDM_HZ   = 20
WISDM_UNIT = "ms2"


def load_wisdm(root: str | Path) -> List[Sample]:
    root = Path(root)

    raw_file = root / "WISDM_ar_v1.1_raw.txt"
    if not raw_file.exists():
        # Try one level deeper (e.g. wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt)
        raw_file = next(root.rglob("WISDM_ar_v1.1_raw.txt"), None)
    if raw_file is None:
        raw_file = next(root.rglob("*raw*.txt"), None)
    if raw_file is None:
        return []

    rows = []
    with open(raw_file) as fh:
        for line in fh:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                user     = int(parts[0].strip())
                activity = parts[1].strip()
                x        = float(parts[3].strip())
                y        = float(parts[4].strip())
                z        = float(parts[5].strip())
                rows.append((user, activity, x, y, z))
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=["user", "activity", "x", "y", "z"])

    samples: List[Sample] = []
    for user_id, group in df.groupby("user"):
        group = group.reset_index(drop=True)
        activity_codes = group["activity"].map(WISDM_LABEL_MAP).fillna(0).values.astype(np.int32)
        signal = group[["x", "y", "z"]].values.astype(np.float32)

        # WISDM has only accelerometer; pad gyro channels with zeros
        gyro = np.zeros((len(signal), 3), dtype=np.float32)
        signal = np.concatenate([signal, gyro], axis=1)

        mask = activity_codes > 0
        if mask.sum() < 100:
            continue

        samples.append((signal[mask], activity_codes[mask], int(user_id), "wisdm"))

    return samples


# ---------------------------------------------------------------------------
# MobiAct loader
# ---------------------------------------------------------------------------
# Each CSV file corresponds to one trial: columns include timestamp, acc_x/y/z, gyro_x/y/z
# Activity label is encoded in the filename: e.g., STD_1_1.csv = Standing, subject 1, trial 1

MOBIACT_LABEL_MAP = {
    "STD": 1,   # standing
    "WAL": 2,   # walking
    "JOG": 3,   # jogging
    "JUM": 4,   # jumping
    "STU": 5,   # stairs up
    "STN": 6,   # stairs down
    "SCH": 7,   # sit on chair
    "CSI": 8,   # car step in
    "CSO": 9,   # car step out
    "SBW": 10,  # sit on bench
    "SBE": 11,  # sit on bed
    "SIT": 1,   # sitting (alias → standing/sitting merged)
}

MOBIACT_HZ   = 87
MOBIACT_UNIT = "ms2"


def load_mobiact(root: str | Path) -> List[Sample]:
    root = Path(root)

    samples: List[Sample] = []
    csv_files = list(root.rglob("*.csv"))

    for f in csv_files:
        stem_parts = f.stem.split("_")
        if len(stem_parts) < 3:
            continue

        activity_code = stem_parts[0].upper()
        if activity_code not in MOBIACT_LABEL_MAP:
            continue

        try:
            subject_id = int(stem_parts[1])
        except ValueError:
            continue

        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        # Flexible column detection
        acc_cols  = [c for c in df.columns if "acc" in c.lower() and any(ax in c.lower() for ax in ["x", "y", "z"])]
        gyro_cols = [c for c in df.columns if "gyro" in c.lower() and any(ax in c.lower() for ax in ["x", "y", "z"])]

        if len(acc_cols) < 3:
            continue

        acc  = df[acc_cols[:3]].values.astype(np.float32)
        gyro = df[gyro_cols[:3]].values.astype(np.float32) if len(gyro_cols) >= 3 \
               else np.zeros((len(acc), 3), dtype=np.float32)

        signal = np.concatenate([acc, gyro], axis=1)
        label_code = MOBIACT_LABEL_MAP[activity_code]
        labels = np.full(len(signal), label_code, dtype=np.int32)

        if len(signal) < 87:  # less than 1 second at 87 Hz
            continue

        samples.append((signal, labels, subject_id, "mobiact"))

    return samples


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "hapt":    (load_hapt,    HAPT_HZ,    HAPT_UNIT),
    "pamap2":  (load_pamap2,  PAMAP2_HZ,  PAMAP2_UNIT),
    "wisdm":   (load_wisdm,   WISDM_HZ,   WISDM_UNIT),
    "mobiact": (load_mobiact, MOBIACT_HZ, MOBIACT_UNIT),
}


def load_dataset(name: str, root: str | Path) -> Tuple[List[Sample], int, str]:
    """Load a dataset by name.

    Returns:
        samples: list of (signal, labels, subject_id, dataset_name)
        hz:      original sampling frequency
        unit:    unit of measurement ("g" or "ms2")
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    loader_fn, hz, unit = DATASET_REGISTRY[name]
    samples = loader_fn(root)
    return samples, hz, unit


def load_all(data_root: str | Path) -> List[Tuple[List[Sample], int, str, str]]:
    """Try loading all datasets that exist under data_root.

    Returns list of (samples, hz, unit, name) for each found dataset.
    """
    data_root = Path(data_root)
    results = []
    for name in DATASET_REGISTRY:
        folder = data_root / name
        if not folder.exists():
            continue
        samples, hz, unit = load_dataset(name, folder)
        if samples:
            results.append((samples, hz, unit, name))
            print(f"[{name}] loaded {len(samples)} subject-trials")
    return results
