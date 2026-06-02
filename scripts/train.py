"""
Main training script.

Usage:
    # 1. Pre-train on merged dataset
    python scripts/train.py --mode pretrain --data data/processed

    # 2. Continual learning (user-incremental)
    python scripts/train.py --mode continual --scenario user --data data/processed --checkpoint checkpoints/pretrained.pt

    # 3. Continual learning (class-incremental)
    python scripts/train.py --mode continual --scenario class --data data/processed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.homogenization import load_processed, UNIFIED_LABELS
from src.data.har_dataset    import HARDataset
from src.models.har_model    import build_model
from src.training.trainer    import pretrain, continual_train
from src.evaluation.metrics  import classification_report


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["pretrain", "continual"], default="pretrain")
    parser.add_argument("--scenario",   choices=["user", "class"],         default="user",
                        help="Continual learning scenario")
    parser.add_argument("--data",       default="data/processed",
                        help="Path to processed data directory (output of preprocess.py)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a pretrained checkpoint to resume from")
    parser.add_argument("--out_dir",    default="checkpoints")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--d_model",    type=int, default=128)
    parser.add_argument("--n_blocks",   type=int, default=4)
    parser.add_argument("--n_heads",    type=int, default=4)
    parser.add_argument("--replay",     type=int, default=2000,
                        help="Replay buffer capacity")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print(f"Loading data from {args.data} ...")
    X, y, subjects, origins = load_processed(args.data)
    ds = HARDataset(X, y, subjects, origins)
    print(ds.summary())

    n_classes = int(y.max()) + 1  # label IDs are 1..K, add 1 for 0-index

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    model = build_model(
        n_classes=n_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        replay_capacity=args.replay,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # Load backbone weights only (classifier head may differ in n_classes)
        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in ckpt["model_state"].items()
            if k.startswith("backbone.")
        }
        model.backbone.load_state_dict(backbone_state)
        print("  Backbone weights loaded (classifier head re-initialized for new n_classes)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Pre-training
    # ------------------------------------------------------------------ #
    if args.mode == "pretrain":
        train_ds, val_ds = ds.train_test_split(test_ratio=0.2, seed=args.seed)
        print(f"\nPre-training | train={len(train_ds)} | val={len(val_ds)}")

        history = pretrain(
            model, train_ds, val_ds,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        # Save checkpoint + history
        ckpt_path = out_dir / "pretrained.pt"
        model.save(str(ckpt_path))
        np.save(str(out_dir / "pretrain_history.npy"), history)
        print(f"\nSaved to {ckpt_path}")

        # Final evaluation
        val_loader = val_ds.dataloader(batch_size=128, shuffle=False)
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds = model(Xb.to(device)).argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(yb.numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        print("\n" + classification_report(y_true, y_pred, label_names=UNIFIED_LABELS))

    # ------------------------------------------------------------------ #
    # Continual training
    # ------------------------------------------------------------------ #
    elif args.mode == "continual":
        if args.scenario == "user":
            tasks = ds.user_incremental_tasks()
            print(f"\nUser-incremental: {len(tasks)} tasks")
        else:
            tasks = ds.class_incremental_tasks(classes_per_task=2, seed=args.seed)
            print(f"\nClass-incremental: {len(tasks)} tasks")

        # Split each task into train/test
        train_tasks, test_tasks = [], []
        for task in tasks:
            tr, te = task.train_test_split(test_ratio=0.2, seed=args.seed)
            if len(tr) > 0 and len(te) > 0:
                train_tasks.append(tr)
                test_tasks.append(te)

        matrix = continual_train(
            model,
            task_sequence=train_tasks,
            test_datasets=test_tasks,
            n_epochs_per_task=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        # Save checkpoint and results
        model.save(str(out_dir / f"continual_{args.scenario}.pt"))
        np.save(str(out_dir / f"results_matrix_{args.scenario}.npy"), matrix.R)
        print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
