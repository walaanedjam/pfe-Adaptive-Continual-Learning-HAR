# Data

This folder contains the raw and processed IMU datasets used for training and evaluation.

## Structure

```
data/
├── raw/          ← original downloaded datasets
│   ├── hapt/     ← UCI HAR with Postural Transitions (50 Hz, 30 subjects)
│   └── wisdm/    ← WISDM Activity Recognition (20 Hz, 51 subjects)
└── processed/    ← preprocessed numpy arrays (X.npy, y.npy, subjects.npy)
```

## Download

| Dataset | URL | Description |
|---------|-----|-------------|
| HAPT | https://archive.ics.uci.edu/dataset/341 | 12 activities + postural transitions |
| WISDM | https://www.cis.fordham.edu/wisdm/dataset.php | 6 daily activities |

## Preprocessing

```bash
python scripts/preprocess.py --data_root data/raw --out data/processed
```

This will generate `X.npy` (N, 150, 6), `y.npy` (N,) and `subjects.npy` (N,) in `data/processed/`.
