"""
Train and evaluate the activity anticipation head (Module 2).

Usage:
    python scripts/train_anticipation.py \
        --data data/processed \
        --checkpoint checkpoints/pretrained.pt \
        --epochs 30 --out_dir checkpoints
"""

import argparse, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.homogenization   import load_processed
from src.data.har_dataset      import HARDataset
from src.data.anticipation_dataset import build_anticipation_datasets
from src.models.har_model      import build_model
from src.training.anticipation_trainer import (
    train_anticipation, print_anticipation_summary)


def get_device():
    import torch
    if torch.cuda.is_available():   return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--out_dir",    default="checkpoints")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--seq_len",    type=int,   default=5,
                        help="Number of consecutive windows as context")
    args = parser.parse_args()

    import torch
    device = get_device()
    print(f"Device: {device}")

    # Load data
    X, y, subjects, origins = load_processed(args.data)
    ds = HARDataset(X, y, subjects, origins)
    print(ds.summary())

    n_classes = int(y.max()) + 1

    # Build model and load pretrained backbone
    model = build_model(n_classes=n_classes)
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state)
    print(f"Backbone loaded from {args.checkpoint}")

    # Build anticipation datasets for all three observation ratios
    # transitions_only=True: only train on windows where next label differs
    # This is the meaningful anticipation task (predicting activity CHANGES)
    print("\nBuilding anticipation datasets (transitions only)...")
    ant_datasets = build_anticipation_datasets(
        X, y, test_ratio=0.2, seq_len=args.seq_len,
        transitions_only=True)

    for ratio, (tr, va) in ant_datasets.items():
        print(f"  p={ratio:.2f}: train={len(tr)}, val={len(va)}")

    # Train
    results = train_anticipation(
        model, ant_datasets,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    print_anticipation_summary(results)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "anticipation.pt"))
    np.save(str(out_dir / "anticipation_results.npy"), results)
    print(f"Saved to {out_dir}/anticipation.pt")


if __name__ == "__main__":
    main()
