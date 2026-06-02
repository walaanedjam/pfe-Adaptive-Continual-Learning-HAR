"""
Run the no-replay naive fine-tuning baseline for comparison.

Usage:
    python scripts/run_baseline.py \
        --data data/processed \
        --checkpoint checkpoints/pretrained.pt \
        --scenario user --epochs 10
"""

import argparse, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.homogenization  import load_processed
from src.data.har_dataset     import HARDataset
from src.models.har_model     import build_model
from src.training.baseline_trainer import naive_finetune
from src.evaluation.metrics   import ContinualResultsMatrix


def get_device():
    import torch
    if torch.cuda.is_available():        return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--scenario",   choices=["user", "class"], default="user")
    parser.add_argument("--out_dir",    default="checkpoints")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    import torch
    device = get_device()
    print(f"Device: {device}")

    X, y, subjects, origins = load_processed(args.data)
    ds = HARDataset(X, y, subjects, origins)
    n_classes = int(y.max()) + 1

    model = build_model(n_classes=n_classes)
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state)
    print(f"Backbone loaded | device={device}")

    if args.scenario == "user":
        tasks = ds.user_incremental_tasks()
    else:
        tasks = ds.class_incremental_tasks(classes_per_task=2, seed=args.seed)

    train_tasks, test_tasks = [], []
    for task in tasks:
        tr, te = task.train_test_split(test_ratio=0.2, seed=args.seed)
        if len(tr) > 0 and len(te) > 0:
            train_tasks.append(tr)
            test_tasks.append(te)

    print(f"\nBaseline — {args.scenario}-incremental: {len(train_tasks)} tasks")

    matrix = naive_finetune(
        model, train_tasks, test_tasks,
        n_epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / f"baseline_matrix_{args.scenario}.npy"), matrix.R)
    print(f"\nSaved baseline results to {out_dir}/baseline_matrix_{args.scenario}.npy")


if __name__ == "__main__":
    main()
