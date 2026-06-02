"""
Visualisation des embeddings — t-SNE et attention.

Usage:
    python scripts/visualize_embeddings.py \
        --data data/processed --checkpoint checkpoints/pretrained.pt
"""

import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

from src.data.homogenization import load_processed, UNIFIED_LABELS
from src.data.har_dataset    import HARDataset
from src.models.har_model    import build_model


COLORS = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9A6324",
]

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def extract_embeddings(model, X, y, device, max_samples=3000):
    """Extraire les embeddings du backbone pour un sous-ensemble de fenêtres."""
    model.eval()
    idx = np.random.choice(len(X), min(max_samples, len(X)), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    embeddings = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(X_sub), batch_size):
            batch = torch.from_numpy(X_sub[i:i+batch_size]).to(device)
            emb   = model.backbone(batch)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings), y_sub


def plot_tsne(embeddings, labels, save_path, title="t-SNE des embeddings IMU"):
    """Réduction t-SNE en 2D et visualisation par classe."""
    print("  Calcul t-SNE (peut prendre ~30 secondes)...")
    tsne   = TSNE(n_components=2, perplexity=40, random_state=42,
                  max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    unique_labels = sorted(np.unique(labels).tolist())
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        name = UNIFIED_LABELS.get(cls, str(cls))
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[i % len(COLORS)],
                   label=name, alpha=0.6, s=15, edgecolors="none")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", fontsize=8, markerscale=2,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {save_path}")


def plot_attention(model, X, y, device, save_path, n_samples=4):
    """Visualise les poids d'attention moyens sur quelques fenêtres."""
    model.eval()
    rng = np.random.default_rng(42)

    unique_cls = np.unique(y).tolist()
    samples    = []
    for cls in unique_cls[:n_samples]:
        idx  = np.where(y == cls)[0]
        pick = rng.choice(idx)
        samples.append((X[pick], cls))

    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    for ax, (x_np, cls) in zip(axes, samples):
        x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
        with torch.no_grad():
            _, attn_list = model.backbone.get_attention_weights(x_t)

        # Moyenne sur les 4 blocs — attention du CLS token vers les timesteps
        attn = np.stack([a.squeeze(0)[0, 1:].numpy()
                         for a in attn_list], axis=0).mean(axis=0)
        # attn shape: (T,)  — importance de chaque timestep
        ax.plot(attn, color="#2196F3", linewidth=1.5)
        ax.fill_between(range(len(attn)), attn, alpha=0.2, color="#2196F3")
        ax.set_title(UNIFIED_LABELS.get(cls, str(cls)), fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("Timestep (50 Hz, 3s total)")
        ax.set_ylabel("Attention weight")
        ax.set_xlim(0, len(attn))
        ax.grid(alpha=0.3)

    fig.suptitle("Poids d'attention du Transformer (token CLS vers les timesteps)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Sauvegardé : {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--out_dir",    default="results/figures")
    parser.add_argument("--max_samples", type=int, default=3000)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    X, y, subjects, origins = load_processed(args.data)
    n_classes = int(y.max()) + 1
    model = build_model(n_classes=n_classes)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k.removeprefix("backbone."): v
             for k, v in ckpt["model_state"].items()
             if k.startswith("backbone.")}
    model.backbone.load_state_dict(state)
    model = model.to(device)
    print("Backbone chargé.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. t-SNE
    print("\n1. t-SNE des embeddings...")
    embs, y_sub = extract_embeddings(model, X, y, device, args.max_samples)
    plot_tsne(embs, y_sub, out_dir / "tsne_embeddings.png")

    # 2. Attention
    print("\n2. Visualisation d'attention...")
    plot_attention(model, X, y, device, out_dir / "attention_weights.png")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
