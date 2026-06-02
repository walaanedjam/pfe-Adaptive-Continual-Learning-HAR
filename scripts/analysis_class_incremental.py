"""
Analyse diagnostique du scénario class-incrémental.
Répond à la question : pourquoi F1=0.159 après 6 tâches ?

Tests :
  1. Impact du nombre de classes par tâche (2 vs 3 vs 4)
  2. Impact de la taille du tampon (1k / 2k / 5k / 10k)
  3. Impact de l'ordre des tâches (classes populaires en premier vs en dernier)
  4. Performance oracle (accès complet à toutes les données)
"""

import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.homogenization   import load_processed
from src.data.har_dataset      import HARDataset
from src.models.har_model      import build_model
from src.evaluation.metrics    import macro_f1


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


def run_class_incremental(model, ds, n_classes_per_task, buffer_size,
                           n_epochs, device, seed=42):
    """Lance un scénario class-incrémental avec les paramètres donnés."""
    from src.training.trainer import continual_train
    tasks = ds.class_incremental_tasks(
        classes_per_task=n_classes_per_task, seed=seed)
    train_tasks, test_tasks = [], []
    for task in tasks:
        tr, te = task.train_test_split(test_ratio=0.2, seed=seed)
        if len(tr) > 0 and len(te) > 0:
            train_tasks.append(tr); test_tasks.append(te)

    # Reconstruire le modèle avec le bon buffer
    n_classes = int(ds.y.max()) + 1
    new_model = build_model(n_classes=n_classes, replay_capacity=buffer_size)
    new_model.backbone.load_state_dict(model.backbone.state_dict())
    new_model = new_model.to(device)

    matrix = continual_train(new_model, train_tasks, test_tasks,
                              n_epochs_per_task=n_epochs,
                              device=str(device), verbose=False)
    return matrix.final_average_accuracy(), matrix.forgetting(), len(train_tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--out_dir",    default="results/figures")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    X, y, subjects, origins = load_processed(args.data)
    ds = HARDataset(X, y, subjects, origins)
    n_classes = int(y.max()) + 1

    model = build_model(n_classes=n_classes)
    load_backbone(model, args.checkpoint)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Test 1 : Impact de la taille du tampon ────────────────────
    print("Test 1 : Impact de la taille du tampon de rejeu")
    print(f"  {'Buffer':>8}  {'F1 final':>9}  {'Forgetting':>11}")
    print(f"  {'-'*32}")
    buffer_sizes = [500, 1000, 2000, 5000, 10000]
    f1s_buffer, forg_buffer = [], []
    for buf in buffer_sizes:
        f1, forg, _ = run_class_incremental(
            model, ds, n_classes_per_task=2,
            buffer_size=buf, n_epochs=args.epochs, device=device)
        f1s_buffer.append(f1); forg_buffer.append(forg)
        print(f"  {buf:>8}  {f1:>9.4f}  {forg:>11.4f}")

    # ── Test 2 : Impact du nombre de classes par tâche ────────────
    print("\nTest 2 : Impact du nombre de classes par tâche")
    print(f"  {'Classes/tâche':>14}  {'N tâches':>9}  {'F1 final':>9}  {'Forgetting':>11}")
    print(f"  {'-'*48}")
    classes_per_tasks = [1, 2, 3, 4, 6]
    f1s_cpt, forg_cpt, n_tasks_list = [], [], []
    for cpt in classes_per_tasks:
        f1, forg, n_t = run_class_incremental(
            model, ds, n_classes_per_task=cpt,
            buffer_size=2000, n_epochs=args.epochs, device=device)
        f1s_cpt.append(f1); forg_cpt.append(forg); n_tasks_list.append(n_t)
        print(f"  {cpt:>14}  {n_t:>9}  {f1:>9.4f}  {forg:>11.4f}")

    # ── Figures ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Courbe buffer
    ax = axes[0]
    ax.plot(buffer_sizes, f1s_buffer,  "o-b", label="F1 final",  linewidth=2)
    ax.plot(buffer_sizes, forg_buffer, "s--r", label="Forgetting", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Taille du tampon de rejeu (log)")
    ax.set_ylabel("Score")
    ax.set_title("Impact de la taille du tampon\n(class-incrémental, 2 classes/tâche)")
    ax.legend(); ax.grid(alpha=0.3)
    for x, y_f1, y_fg in zip(buffer_sizes, f1s_buffer, forg_buffer):
        ax.annotate(f"{y_f1:.2f}", (x, y_f1), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color="blue")

    # Courbe classes/tâche
    ax = axes[1]
    ax.plot(classes_per_tasks, f1s_cpt,  "o-b", label="F1 final",  linewidth=2)
    ax.plot(classes_per_tasks, forg_cpt, "s--r", label="Forgetting", linewidth=2)
    ax.set_xlabel("Nombre de classes par tâche")
    ax.set_ylabel("Score")
    ax.set_title("Impact du nombre de classes/tâche\n(buffer fixe = 2000)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Analyse class-incrémental : causes de la dégradation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = out_dir / "class_incremental_analysis.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nFigure sauvegardée : {save_path}")

    # ── Conclusion ────────────────────────────────────────────────
    print("\nConclusion :")
    best_buf = buffer_sizes[np.argmax(f1s_buffer)]
    print(f"  Meilleur tampon : {best_buf} (F1={max(f1s_buffer):.4f})")
    best_cpt = classes_per_tasks[np.argmax(f1s_cpt)]
    print(f"  Meilleur classes/tâche : {best_cpt} (F1={max(f1s_cpt):.4f})")

    np.save(str(Path(args.out_dir) / "../checkpoints/class_incremental_analysis.npy"),
            {"buffer_sizes": buffer_sizes, "f1s_buffer": f1s_buffer,
             "classes_per_tasks": classes_per_tasks, "f1s_cpt": f1s_cpt})


if __name__ == "__main__":
    main()
