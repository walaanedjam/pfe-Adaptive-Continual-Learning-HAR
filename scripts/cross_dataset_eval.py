"""
Évaluation cross-dataset — proxy pour la robustesse en vie réelle.

Scénario : entraîner sur HAPT (30 sujets), évaluer sur WISDM (36 sujets).
Simule un changement de contexte réel : autre appareil, autre port du capteur.

Usage:
    python scripts/cross_dataset_eval.py \
        --data data/processed --checkpoint checkpoints/pretrained.pt
"""

import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.homogenization  import load_processed, UNIFIED_LABELS
from src.data.har_dataset     import HARDataset
from src.models.har_model     import build_model
from src.evaluation.metrics   import macro_f1, classification_report


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


def eval_on_dataset(model, X, y, device_t, mode="prototype"):
    """Évalue le modèle sur un dataset entier."""
    model.eval()
    preds, labels = [], []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_b = torch.from_numpy(X[i:i+batch_size]).to(device_t)
            if mode == "prototype":
                if model.har_head.prototype_memory.n_classes() == 0:
                    return 0.0, np.array([]), np.array([])
                emb = model.backbone(X_b)
                p   = model.har_head.prototype_memory.predict(emb).cpu().numpy()
            else:
                p = model(X_b).argmax(dim=-1).cpu().numpy()
            preds.append(p)
            labels.append(y[i:i+batch_size])

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    return macro_f1(y_true, y_pred), y_pred, y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    args = parser.parse_args()

    device   = get_device()
    device_t = torch.device(device)
    print(f"Device: {device}\n")

    X, y, subjects, origins = load_processed(args.data)
    n_classes = int(y.max()) + 1

    # Séparer HAPT et WISDM
    hapt_mask  = np.array([o == "hapt"  for o in origins])
    wisdm_mask = np.array([o == "wisdm" for o in origins])
    pamap_mask = np.array([o == "pamap2" for o in origins])

    X_hapt,  y_hapt  = X[hapt_mask],  y[hapt_mask]
    X_wisdm, y_wisdm = X[wisdm_mask], y[wisdm_mask]
    X_pamap, y_pamap = X[pamap_mask], y[pamap_mask]

    print(f"HAPT  : {len(X_hapt):,} fenêtres | "
          f"classes: {sorted(np.unique(y_hapt).tolist())}")
    print(f"WISDM : {len(X_wisdm):,} fenêtres | "
          f"classes: {sorted(np.unique(y_wisdm).tolist())}")
    print(f"PAMAP2: {len(X_pamap):,} fenêtres | "
          f"classes: {sorted(np.unique(y_pamap).tolist())}\n")

    model = build_model(n_classes=n_classes)
    load_backbone(model, args.checkpoint)
    model = model.to(device_t)

    # Initialiser les prototypes depuis HAPT uniquement
    print("Initialisation des prototypes depuis HAPT...")
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_hapt), 256):
            X_b = torch.from_numpy(X_hapt[i:i+256]).to(device_t)
            y_b = torch.from_numpy(y_hapt[i:i+256]).to(device_t)
            emb = model.backbone(X_b)
            model.har_head.prototype_memory.update(emb, y_b)
    print(f"  Prototypes initialisés : "
          f"{model.har_head.prototype_memory.n_classes()} classes\n")

    # ── Évaluation intra-dataset (HAPT → HAPT) ─────────────────────
    f1_hapt, _, _ = eval_on_dataset(model, X_hapt, y_hapt, device_t)
    print(f"Intra-dataset  HAPT  → HAPT  : F1 = {f1_hapt:.4f}")

    # ── Évaluation cross-dataset (HAPT → WISDM) ────────────────────
    # Classes communes entre HAPT et WISDM
    common_classes = set(np.unique(y_hapt).tolist()) & \
                     set(np.unique(y_wisdm).tolist())
    mask_common = np.isin(y_wisdm, list(common_classes))
    X_w_common  = X_wisdm[mask_common]
    y_w_common  = y_wisdm[mask_common]

    f1_cross, y_pred, y_true = eval_on_dataset(
        model, X_w_common, y_w_common, device_t)
    print(f"Cross-dataset  HAPT  → WISDM : F1 = {f1_cross:.4f}"
          f"  (classes communes: {sorted(common_classes)})")

    # ── Évaluation cross-dataset (HAPT → PAMAP2) ───────────────────
    common_p = set(np.unique(y_hapt).tolist()) & \
               set(np.unique(y_pamap).tolist())
    mask_p   = np.isin(y_pamap, list(common_p))
    f1_pamap, _, _ = eval_on_dataset(
        model, X_pamap[mask_p], y_pamap[mask_p], device_t)
    print(f"Cross-dataset  HAPT  → PAMAP2: F1 = {f1_pamap:.4f}"
          f"  (classes communes: {sorted(common_p)})")

    # ── Résumé ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Résumé évaluation cross-dataset (proxy vie réelle)")
    print(f"{'='*55}")
    print(f"  HAPT → HAPT  (même dataset)   : {f1_hapt:.4f}  (référence)")
    print(f"  HAPT → WISDM (autre appareil) : {f1_cross:.4f}"
          f"  ({f1_cross/f1_hapt*100:.0f}% de la perf. intra)")
    print(f"  HAPT → PAMAP2 (placement diff): {f1_pamap:.4f}"
          f"  ({f1_pamap/f1_hapt*100:.0f}% de la perf. intra)")

    print("\nInterprétation :")
    if f1_cross > 0.5:
        print("  Le backbone généralise bien entre datasets : robustesse réelle.")
    else:
        print("  Dégradation cross-dataset : domain shift significatif.")
        print("  Piste : fine-tuning avec quelques exemples du nouveau domaine.")

    np.save("checkpoints/cross_dataset_results.npy",
            {"hapt_hapt": f1_hapt,
             "hapt_wisdm": f1_cross,
             "hapt_pamap2": f1_pamap})
    print("\nSauvegardé : checkpoints/cross_dataset_results.npy")


if __name__ == "__main__":
    main()
