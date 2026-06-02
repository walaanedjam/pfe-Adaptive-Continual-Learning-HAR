"""
Comparaison SOTA : notre méthode vs EWC vs iCaRL vs baseline naïve.

Usage:
    python scripts/run_sota_comparison.py \
        --data data/processed --checkpoint checkpoints/pretrained.pt \
        --n_tasks 15 --epochs 10
"""

import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.homogenization    import load_processed
from src.data.har_dataset       import HARDataset
from src.models.har_model       import build_model
from src.training.trainer       import continual_train
from src.training.baseline_trainer import naive_finetune
from src.training.ewc_trainer   import run_ewc
from src.training.icarl_trainer import run_icarl
from src.evaluation.metrics     import ContinualResultsMatrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_backbone(model, ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("backbone."): v
             for k, v in ckpt["model_state"].items()
             if k.startswith("backbone.")}
    model.backbone.load_state_dict(state)


def plot_comparison(results: dict, save_path: Path):
    """Courbes de F1 final par tâche pour toutes les méthodes."""
    colors = {"Notre méthode": "#2196F3",
              "iCaRL":         "#4CAF50",
              "EWC":           "#FF9800",
              "Baseline naïve":"#F44336"}
    styles = {"Notre méthode": "-o",
              "iCaRL":         "-s",
              "EWC":           "-^",
              "Baseline naïve":"--x"}

    fig, ax = plt.subplots(figsize=(10, 5))
    n = max(m.R.shape[0] for m in results.values())

    for name, matrix in results.items():
        K    = matrix.R.shape[0]
        last = [matrix.R[K-1, j] for j in range(K)
                if not np.isnan(matrix.R[K-1, j])]
        xs   = list(range(len(last)))
        ax.plot(xs, last, styles.get(name, "-o"),
                color=colors.get(name, "gray"),
                label=name, linewidth=2, markersize=5)

    ax.set_xlabel("Indice de tâche (sujet)")
    ax.set_ylabel("Macro-F1 (après toutes les tâches)")
    ax.set_title("Comparaison SOTA — Scénario utilisateur-incrémental")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sauvegardé : {save_path}")


def plot_summary_bar(results: dict, save_path: Path):
    """Barres BWT / Forgetting / F1 final pour chaque méthode."""
    methods = list(results.keys())
    f1s  = [results[m].final_average_accuracy() for m in methods]
    forg = [results[m].forgetting()             for m in methods]
    bwts = [results[m].backward_transfer()      for m in methods]

    x    = np.arange(len(methods))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, f1s,   w, label="F1 final",  color="#4CAF50")
    ax.bar(x,     forg,  w, label="Forgetting (↓)", color="#F44336")
    ax.bar(x + w, [-b for b in bwts], w,
           label="-BWT (↓)", color="#FF9800")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Résumé comparaison SOTA")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sauvegardé : {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--n_tasks",    type=int, default=15)
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out_dir",    default="checkpoints")
    parser.add_argument("--fig_dir",    default="results/figures")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Charger données
    X, y, subjects, origins = load_processed(args.data)
    ds = HARDataset(X, y, subjects, origins)
    n_classes = int(y.max()) + 1

    tasks = ds.user_incremental_tasks()
    train_tasks, test_tasks = [], []
    for task in tasks[:args.n_tasks + 2]:
        tr, te = task.train_test_split(test_ratio=0.2, seed=args.seed)
        if len(tr) > 0 and len(te) > 0:
            train_tasks.append(tr); test_tasks.append(te)
    train_tasks = train_tasks[:args.n_tasks]
    test_tasks  = test_tasks[:args.n_tasks]

    print(f"\nComparaison SOTA — {len(train_tasks)} tâches, "
          f"{args.epochs} époques/tâche\n")

    results = {}

    # ── 1. Baseline naïve ──────────────────────────────────────────
    print("="*50)
    print("1/4  Baseline naïve (sans replay)")
    model = build_model(n_classes=n_classes)
    load_backbone(model, args.checkpoint)
    matrix = naive_finetune(model, train_tasks, test_tasks,
                             n_epochs_per_task=args.epochs,
                             batch_size=args.batch_size, lr=args.lr,
                             device=device, verbose=False)
    results["Baseline naïve"] = matrix
    print(f"  F1={matrix.final_average_accuracy():.4f}  "
          f"Forgetting={matrix.forgetting():.4f}")

    # ── 2. EWC ────────────────────────────────────────────────────
    print("="*50)
    print("2/4  EWC (Elastic Weight Consolidation)")
    model = build_model(n_classes=n_classes)
    load_backbone(model, args.checkpoint)
    matrix = run_ewc(model, train_tasks, test_tasks,
                     n_epochs_per_task=args.epochs,
                     batch_size=args.batch_size, lr=args.lr,
                     ewc_lambda=5000.0, device=device, verbose=False)
    results["EWC"] = matrix
    print(f"  F1={matrix.final_average_accuracy():.4f}  "
          f"Forgetting={matrix.forgetting():.4f}")

    # ── 3. iCaRL ──────────────────────────────────────────────────
    print("="*50)
    print("3/4  iCaRL (herding + distillation)")
    model = build_model(n_classes=n_classes)
    load_backbone(model, args.checkpoint)
    matrix = run_icarl(model, train_tasks, test_tasks,
                       n_epochs_per_task=args.epochs,
                       batch_size=args.batch_size, lr=args.lr,
                       memory_size=2000, device=device, verbose=False)
    results["iCaRL"] = matrix
    print(f"  F1={matrix.final_average_accuracy():.4f}  "
          f"Forgetting={matrix.forgetting():.4f}")

    # ── 4. Notre méthode ──────────────────────────────────────────
    print("="*50)
    print("4/4  Notre méthode (replay + prototypes + contrastif)")
    model = build_model(n_classes=n_classes)
    load_backbone(model, args.checkpoint)
    matrix = continual_train(model, train_tasks, test_tasks,
                              n_epochs_per_task=args.epochs,
                              batch_size=args.batch_size, lr=args.lr,
                              device=device, verbose=False)
    results["Notre méthode"] = matrix
    print(f"  F1={matrix.final_average_accuracy():.4f}  "
          f"Forgetting={matrix.forgetting():.4f}")

    # ── Tableau résumé ────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  {'Méthode':<25} {'F1 final':>9} {'Forgetting':>11} {'BWT':>8}")
    print(f"  {'-'*55}")
    for name, m in results.items():
        print(f"  {name:<25} {m.final_average_accuracy():>9.4f} "
              f"{m.forgetting():>11.4f} {m.backward_transfer():>8.4f}")

    # ── Sauvegarder ───────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for name, m in results.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        np.save(str(out_dir / f"sota_{fname}.npy"), m.R)

    plot_comparison(results,    fig_dir / "sota_comparison.png")
    plot_summary_bar(results,   fig_dir / "sota_summary_bar.png")

    print(f"\nFigures sauvegardées dans {fig_dir}/")


if __name__ == "__main__":
    main()
