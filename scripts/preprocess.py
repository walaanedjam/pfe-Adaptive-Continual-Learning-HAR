"""
Run this script once to homogenize and save all datasets.

Usage:
    python scripts/preprocess.py --data_root data/raw --out data/processed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.homogenization import build_unified_dataset, save_processed
from src.data.har_dataset import dataset_from_arrays
from src.data.dataset_loaders import download_instructions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw",
                        help="Root folder containing per-dataset raw folders")
    parser.add_argument("--out", default="data/processed",
                        help="Where to save processed numpy arrays")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Which datasets to process (default: all found)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Skip per-dataset z-score normalization")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not any((data_root / d).exists() for d in ["hapt", "mobiact", "pamap2", "wisdm"]):
        print("No dataset folders found. Download instructions:")
        download_instructions()
        sys.exit(1)

    print(f"Building unified dataset from: {data_root}")
    X, y, subjects, origins, stats = build_unified_dataset(
        data_root=data_root,
        datasets=args.datasets,
        discard_unknown=True,
        normalize=not args.no_normalize,
    )

    print(f"\nTotal windows: {len(X)}")
    print(f"X shape: {X.shape}")

    ds = dataset_from_arrays(X, y, subjects, origins)
    print("\n" + ds.summary())

    save_processed(args.out, X, y, subjects, origins)
    print(f"\nDone. Files saved to: {args.out}")


if __name__ == "__main__":
    main()
