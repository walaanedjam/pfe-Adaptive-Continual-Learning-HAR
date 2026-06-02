"""
Entraînement conjoint backbone + LSTM pour l'anticipation.

Contrairement à la version gelée, ici le backbone est entraîné
avec un taux d'apprentissage réduit (10x moins que le LSTM)
pour adapter ses représentations à la tâche de prédiction.

Usage:
    python scripts/train_anticipation_joint.py \
        --data data/processed --checkpoint checkpoints/pretrained.pt
"""

import sys, argparse, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.homogenization       import load_processed
from src.data.har_dataset          import HARDataset
from src.data.anticipation_dataset import build_anticipation_datasets
from src.models.har_model          import build_model
from src.evaluation.metrics        import macro_f1


def get_device():
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def train_joint(model, datasets, n_epochs=20, batch_size=128,
                lr_lstm=5e-4, lr_backbone=5e-5, device="cpu"):
    """
    Entraînement conjoint — backbone + LSTM avec taux différenciés.
    lr_backbone << lr_lstm pour ne pas écraser les représentations.
    """
    device_t = torch.device(device)
    model    = model.to(device_t)

    results = {}
    for obs_ratio, (train_ds, val_ds) in datasets.items():
        if len(train_ds) == 0: continue
        print(f"\n  p={obs_ratio:.2f} | train={len(train_ds)} val={len(val_ds)}")

        # Réinitialiser le LSTM
        for m in model.anticipation_head.modules():
            if isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight" in name: nn.init.orthogonal_(p)
                    elif "bias" in name: nn.init.zeros_(p)

        # Optimiseur avec taux différenciés
        optimizer = AdamW([
            {"params": model.backbone.parameters(),          "lr": lr_backbone},
            {"params": model.anticipation_head.parameters(), "lr": lr_lstm},
        ], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs,
                                       eta_min=lr_lstm * 0.01)

        train_loader = train_ds.dataloader(batch_size=batch_size, shuffle=True)
        val_loader   = val_ds.dataloader(batch_size=batch_size, shuffle=False)

        best_f1, best_acc = 0, 0
        for epoch in range(1, n_epochs + 1):
            # ── Train ──
            model.train()
            for X_seq, y_next in train_loader:
                X_seq  = X_seq.to(device_t)
                y_next = y_next.to(device_t)
                optimizer.zero_grad()
                logits = model.anticipate(X_seq)
                F.cross_entropy(logits, y_next).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # ── Val ──
            model.eval()
            preds, labels, losses = [], [], []
            with torch.no_grad():
                for X_seq, y_next in val_loader:
                    X_seq, y_next = X_seq.to(device_t), y_next.to(device_t)
                    logits = model.anticipate(X_seq)
                    losses.append(F.cross_entropy(logits, y_next).item())
                    preds.append(logits.argmax(dim=-1).cpu().numpy())
                    labels.append(y_next.cpu().numpy())

            y_pred = np.concatenate(preds)
            y_true = np.concatenate(labels)
            val_f1  = macro_f1(y_true, y_pred)
            val_acc = float((y_pred == y_true).mean())
            if val_f1 > best_f1: best_f1 = val_f1
            if val_acc > best_acc: best_acc = val_acc

            if epoch % 5 == 0 or epoch == 1:
                print(f"    epoch {epoch:3d}/{n_epochs} | "
                      f"val_loss={np.mean(losses):.4f} | "
                      f"val_F1={val_f1:.4f} | val_acc={val_acc:.4f}")

        results[obs_ratio] = {"val_f1": best_f1, "val_acc": best_acc}
        print(f"  Best: F1={best_f1:.4f}  Acc={best_acc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr_lstm",    type=float, default=5e-4)
    parser.add_argument("--lr_backbone",type=float, default=5e-5)
    parser.add_argument("--out_dir",    default="checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    X, y, subjects, origins = load_processed(args.data)
    n_classes = int(y.max()) + 1
    model     = build_model(n_classes=n_classes)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k.removeprefix("backbone."): v
             for k, v in ckpt["model_state"].items()
             if k.startswith("backbone.")}
    model.backbone.load_state_dict(state)
    print(f"Backbone chargé. Entraînement conjoint (backbone dégelé).\n")

    datasets = build_anticipation_datasets(X, y, test_ratio=0.2,
                                            seq_len=5, transitions_only=True)

    print("=== Entraînement conjoint backbone + LSTM ===")
    results = train_joint(model, datasets,
                           n_epochs=args.epochs,
                           batch_size=args.batch_size,
                           lr_lstm=args.lr_lstm,
                           lr_backbone=args.lr_backbone,
                           device=device)

    print("\n=== Résultats anticipation (joint) ===")
    print(f"  {'Obs ratio':>10}  {'Val F1':>8}  {'Val Acc':>8}")
    print(f"  {'-'*32}")
    for ratio in sorted(results):
        r = results[ratio]
        print(f"  {ratio*100:>8.0f}%  {r['val_f1']:>8.4f}  {r['val_acc']:>8.4f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "anticipation_joint.pt"))
    np.save(str(out / "anticipation_joint_results.npy"), results)
    print(f"\nSauvegardé : {out}/anticipation_joint.pt")


if __name__ == "__main__":
    main()
