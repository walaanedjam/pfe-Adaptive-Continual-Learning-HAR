"""
Ablation study — compare les variantes du modèle.

Configurations testées (scénario user-incrémental, 10 premières tâches) :
  A. Modèle complet     : replay + prototypes + contrastif
  B. Sans contrastif    : λ_c = 0
  C. Sans re-adaptation : pas de mise à jour des prototypes après backbone update
  D. Sans replay        : baseline naïve (déjà dans baseline_trainer.py)
  E. Replay incertain   : uncertainty-weighted replay (contribution originale)

Usage:
    python scripts/ablation.py --data data/processed --checkpoint checkpoints/pretrained.pt
"""

import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn.functional as F
from torch.optim import AdamW

from src.data.homogenization  import load_processed
from src.data.har_dataset     import HARDataset
from src.models.har_model     import build_model
from src.evaluation.metrics   import macro_f1


def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_backbone(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("backbone."): v
             for k, v in ckpt["model_state"].items()
             if k.startswith("backbone.")}
    model.backbone.load_state_dict(state)


def run_variant(name, model, train_tasks, test_tasks, device,
                use_replay=True, use_contrastive=True,
                use_proto_adapt=True, uncertainty=False,
                n_epochs=10, batch_size=64, lr=5e-5, n_tasks=10):

    model = model.to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    results = []

    for task_i in range(min(n_tasks, len(train_tasks))):
        loader = train_tasks[task_i].dataloader(batch_size=batch_size, shuffle=True)

        for _ in range(n_epochs):
            model.train()
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()

                emb    = model.backbone(X)
                logits = model.har_head(emb)
                loss   = F.cross_entropy(logits, y)

                # Replay
                if use_replay and len(model.replay_buffer) >= 32:
                    if uncertainty:
                        rx, ry = model.replay_buffer.sample_uncertain(
                            model, 32, device=device)
                    else:
                        rx, ry = model.replay_buffer.sample(32, device=device)
                    loss = loss + F.cross_entropy(model.har_head(model.backbone(rx)), ry)

                # Contrastive
                if use_contrastive:
                    loss = loss + 0.1 * model.har_head.contrastive_loss(emb, y)

                loss.backward()
                opt.step()

        # Update prototypes
        model.eval()
        with torch.no_grad():
            for X, y in loader:
                emb = model.backbone(X.to(device))
                model.har_head.update_prototypes(emb, y.to(device))

        # Prototype re-adaptation from replay
        if use_proto_adapt and len(model.replay_buffer) > 0:
            rx, ry = model.replay_buffer.sample(
                min(512, len(model.replay_buffer)), device=device)
            model.har_head.prototype_memory.adapt_prototypes(
                model.backbone, rx, ry, device)

        # Add to replay
        for X, y in loader:
            model.replay_buffer.add_batch(X.numpy(), y.numpy())

        # Eval on all tasks so far
        task_f1s = []
        for j in range(task_i + 1):
            f1 = _eval(model, test_tasks[j], device)
            task_f1s.append(f1)
        results.append(task_f1s)

    # Final avg F1
    final_row = results[-1]
    avg = np.mean(final_row)
    print(f"  {name:<40} Final avg F1 = {avg:.4f}")
    return avg, results


def _eval(model, ds, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in ds.dataloader(batch_size=128, shuffle=False):
            if model.har_head.prototype_memory.n_classes() == 0:
                return 0.0
            emb = model.backbone(X.to(device))
            p   = model.har_head.prototype_memory.predict(emb).cpu().numpy()
            preds.append(p); labels.append(y.numpy())
    return macro_f1(np.concatenate(labels), np.concatenate(preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--n_tasks",    type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--out_dir",    default="checkpoints")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    X, y, subjects, origins = load_processed(args.data)
    ds = HARDataset(X, y, subjects, origins)
    n_classes = int(y.max()) + 1

    tasks = ds.user_incremental_tasks()
    train_tasks, test_tasks = [], []
    for task in tasks:
        tr, te = task.train_test_split(test_ratio=0.2, seed=args.seed)
        if len(tr) > 0 and len(te) > 0:
            train_tasks.append(tr); test_tasks.append(te)

    print(f"Ablation study — {args.n_tasks} tasks, {args.epochs} epochs/task\n")
    print(f"  {'Variant':<40} {'Result'}")
    print(f"  {'-'*55}")

    ablation_results = {}

    configs = [
        ("A. Complet (replay+proto+contrastif)",
         dict(use_replay=True, use_contrastive=True,
              use_proto_adapt=True, uncertainty=False)),
        ("B. Sans perte contrastive (λ_c=0)",
         dict(use_replay=True, use_contrastive=False,
              use_proto_adapt=True, uncertainty=False)),
        ("C. Sans re-adaptation prototypes",
         dict(use_replay=True, use_contrastive=True,
              use_proto_adapt=False, uncertainty=False)),
        ("D. Sans replay (baseline naïve)",
         dict(use_replay=False, use_contrastive=False,
              use_proto_adapt=True, uncertainty=False)),
        ("E. Replay incertain (contribution originale)",
         dict(use_replay=True, use_contrastive=True,
              use_proto_adapt=True, uncertainty=True)),
    ]

    for name, kwargs in configs:
        torch.manual_seed(args.seed)
        model = build_model(n_classes=n_classes,
                            uncertainty_replay=kwargs.get("uncertainty", False))
        load_backbone(model, args.checkpoint)
        avg, _ = run_variant(name, model, train_tasks, test_tasks,
                              device, n_tasks=args.n_tasks,
                              n_epochs=args.epochs, **kwargs)
        ablation_results[name] = avg

    print(f"\n{'='*55}")
    print("Résumé ablation :")
    baseline = ablation_results.get("D. Sans replay (baseline naïve)", 0)
    full     = ablation_results.get("A. Complet (replay+proto+contrastif)", 1)
    for name, val in ablation_results.items():
        delta = val - baseline
        print(f"  {name:<45} {val:.4f}  ({delta:+.4f} vs baseline)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(str(out / "ablation_results.npy"), ablation_results)
    print(f"\nSauvegardé dans {out}/ablation_results.npy")


if __name__ == "__main__":
    main()
