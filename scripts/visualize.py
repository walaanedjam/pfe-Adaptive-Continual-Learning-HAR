"""
Plot training results: forgetting curves, BWT/FWT, per-task F1.

Usage:
    python scripts/visualize.py --results_dir checkpoints/ --scenario user
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.evaluation.metrics import ContinualResultsMatrix
from src.data.homogenization import UNIFIED_LABELS


def plot_results_matrix(R: np.ndarray, save_path: Path):
    """Heatmap of the results matrix R[i,j]."""
    n = R.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

    masked = np.ma.masked_where(np.isnan(R), R)
    im = ax.imshow(masked, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Macro F1")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"T{j}" for j in range(n)])
    ax.set_yticklabels([f"After T{i}" for i in range(n)])
    ax.set_xlabel("Evaluated on task")
    ax.set_ylabel("After training on task")
    ax.set_title("Continual Learning Results Matrix R[i,j]")

    for i in range(n):
        for j in range(n):
            if not np.isnan(R[i, j]):
                ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if R[i, j] > 0.5 else "white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_forgetting_curve(R: np.ndarray, save_path: Path):
    """Per-task accuracy over time (diagonal vs. final row)."""
    n = R.shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))

    for j in range(n - 1):
        # Accuracy on task j as more tasks are learned
        curve = [R[i, j] for i in range(j, n) if not np.isnan(R[i, j])]
        xs    = list(range(j, j + len(curve)))
        ax.plot(xs, curve, marker="o", label=f"Task {j}")

    ax.set_xlabel("Number of tasks learned")
    ax.set_ylabel("Macro F1")
    ax.set_title("Forgetting Curves (per-task accuracy over time)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_bar(matrix: ContinualResultsMatrix, save_path: Path):
    """Bar chart of the four continual learning metrics."""
    metrics = {
        "Final Avg F1":        matrix.final_average_accuracy(),
        "Backward\nTransfer":  matrix.backward_transfer(),
        "Forgetting":         -matrix.forgetting(),   # negate: lower forgetting = better
        "Forward\nTransfer":   matrix.forward_transfer(),
    }

    colors = ["#4CAF50", "#2196F3", "#F44336", "#FF9800"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, width=0.5)

    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * np.sign(val + 1e-9),
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Score")
    ax.set_title("Continual Learning Summary Metrics")
    ax.set_ylim(-0.5, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_history(history: dict, save_path: Path):
    """Loss and F1 curves from pre-training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Pre-training Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # F1
    axes[1].plot(epochs, history["val_f1"], color="green", label="Val macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_baseline_comparison(R_ours: np.ndarray,
                              R_base: np.ndarray,
                              save_path: Path,
                              scenario: str = "user"):
    """Line plot comparing our method vs no-replay baseline per task."""
    n = min(R_ours.shape[0], R_base.shape[0])
    fig, ax = plt.subplots(figsize=(9, 5))

    # Final row = performance after all tasks
    ours_final = [R_ours[n-1, j] for j in range(n) if not np.isnan(R_ours[n-1, j])]
    base_final = [R_base[n-1, j] for j in range(n) if not np.isnan(R_base[n-1, j])]
    xs_ours = [j for j in range(n) if not np.isnan(R_ours[n-1, j])]
    xs_base = [j for j in range(n) if not np.isnan(R_base[n-1, j])]

    ax.plot(xs_ours, ours_final, "o-", color="#2196F3",
            label="Ours (replay + prototypes + contrastive)", linewidth=2)
    ax.plot(xs_base, base_final, "s--", color="#F44336",
            label="Baseline (naive fine-tuning, no replay)", linewidth=2, alpha=0.8)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Macro F1 (after all tasks)")
    ax.set_title(f"Our method vs baseline — {scenario}-incremental")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_anticipation(results: dict, save_path: Path):
    """Bar chart of anticipation accuracy at each observation ratio."""
    ratios = sorted(results.keys())
    f1s    = [results[r]["val_f1"]  for r in ratios]
    accs   = [results[r]["val_acc"] for r in ratios]
    labels = [f"{int(r*100)}%" for r in ratios]

    x = np.arange(len(ratios))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - 0.2, f1s,  0.35, label="Macro F1",  color="#4CAF50")
    bars2 = ax.bar(x + 0.2, accs, 0.35, label="Accuracy",  color="#2196F3")

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"p={l}" for l in labels])
    ax.set_ylabel("Score")
    ax.set_title("Activity Anticipation — Performance vs Observation Ratio")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="checkpoints")
    parser.add_argument("--scenario", choices=["user", "class"], default="user")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Continual learning results ---
    R_path = results_dir / f"results_matrix_{args.scenario}.npy"
    if R_path.exists():
        R      = np.load(R_path)
        matrix = ContinualResultsMatrix(n_tasks=R.shape[0])
        matrix.R = R

        print(matrix.summary())

        plot_results_matrix(R, out_dir / f"results_matrix_{args.scenario}.png")
        plot_forgetting_curve(R, out_dir / f"forgetting_{args.scenario}.png")
        plot_summary_bar(matrix, out_dir / f"summary_{args.scenario}.png")
    else:
        print(f"No results matrix found at {R_path}")

    # --- Baseline comparison ---
    base_path = results_dir / f"baseline_matrix_{args.scenario}.npy"
    if base_path.exists() and R_path.exists():
        R_base = np.load(base_path)
        plot_baseline_comparison(
            R, R_base,
            out_dir / f"baseline_comparison_{args.scenario}.png",
            scenario=args.scenario)

    # --- Anticipation results ---
    ant_path = results_dir / "anticipation_results.npy"
    if ant_path.exists():
        ant_results = np.load(ant_path, allow_pickle=True).item()
        plot_anticipation(ant_results, out_dir / "anticipation_results.png")

    # --- Pre-training history ---
    hist_path = results_dir / "pretrain_history.npy"
    if hist_path.exists():
        history = np.load(hist_path, allow_pickle=True).item()
        plot_training_history(history, out_dir / "pretrain_history.png")
    else:
        print(f"No training history found at {hist_path}")


if __name__ == "__main__":
    main()
