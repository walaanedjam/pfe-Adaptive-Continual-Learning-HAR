"""
Interactive demo — run this to test the full system.

    python demo.py

Shows:
  1. Activity recognition on real windows from the dataset
  2. Continual learning: adapt to a new user on the fly
  3. Activity anticipation from partial windows
"""

import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.homogenization    import load_processed, UNIFIED_LABELS
from src.data.har_dataset       import HARDataset
from src.models.har_model       import build_model
from src.data.anticipation_dataset import AnticipationDataset

# ── colours for terminal output ──────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(text):
    print(f"\n{BOLD}{CYAN}{'='*55}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*55}{RESET}\n")

def load():
    print("Loading data and model...")
    X, y, subjects, origins = load_processed("data/processed")
    ds = HARDataset(X, y, subjects, origins)

    n_classes = int(y.max()) + 1
    model = build_model(n_classes=n_classes)
    ckpt  = torch.load("checkpoints/pretrained.pt",
                        map_location="cpu", weights_only=False)
    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state)
    model.eval()
    print(f"  Dataset : {len(ds):,} windows | {len(np.unique(subjects))} subjects")
    print(f"  Model   : {sum(p.numel() for p in model.parameters()):,} parameters\n")
    return ds, model, X, y, subjects, origins


# ── DEMO 1: Activity Recognition ─────────────────────────────────────────────
def demo_recognition(ds, model, X, y):
    banner("DEMO 1 — Activity Recognition")

    # Build prototype memory from first 2000 windows (training set proxy)
    print("Initialising prototype memory from 2000 training windows...")
    model.eval()
    with torch.no_grad():
        X_init = torch.from_numpy(X[:2000])
        y_init = torch.from_numpy(y[:2000])
        emb    = model.backbone(X_init)
        model.har_head.prototype_memory.update(emb, y_init)
    n_cls = model.har_head.prototype_memory.n_classes()
    print(f"  {n_cls} class prototypes ready\n")
    print("Predicting activity for 10 random windows (held-out samples)...\n")

    rng = np.random.default_rng(42)
    idx = rng.choice(range(2000, len(ds)), 10, replace=False)

    correct = 0
    print(f"  {'#':>2}  {'True label':<25} {'Predicted':<25} {'✓/✗'}")
    print(f"  {'-'*60}")

    for i, wi in enumerate(idx):
        x, y_true = ds[wi]
        x = x.unsqueeze(0)

        with torch.no_grad():
            emb  = model.backbone(x)
            pred = model.har_head.prototype_memory.predict(emb).item()

        true_name = UNIFIED_LABELS.get(int(y_true), str(int(y_true)))
        pred_name = UNIFIED_LABELS.get(pred, str(pred))
        ok = "✓" if pred == int(y_true) else "✗"
        if pred == int(y_true): correct += 1

        color = GREEN if pred == int(y_true) else YELLOW
        print(f"  {i+1:>2}  {true_name:<25} {color}{pred_name:<25}{RESET} {color}{ok}{RESET}")

    print(f"\n  Accuracy on these 10 samples: {correct}/10")


# ── DEMO 2: Continual Learning — new user adaptation ─────────────────────────
def demo_continual(ds, model, X, y, subjects):
    banner("DEMO 2 — Continual Learning (new user adaptation)")

    # First: init prototypes from a base set of subjects (1–4)
    print("Step 1: initialize model on subjects 1–4 (existing users)...")
    base_mask = np.isin(subjects, [1, 2, 3, 4])
    X_base = torch.from_numpy(X[base_mask][:500])
    y_base = torch.from_numpy(y[base_mask][:500])
    model.eval()
    with torch.no_grad():
        emb_base = model.backbone(X_base)
        model.har_head.prototype_memory.update(emb_base, y_base)
    print(f"  Prototypes for {model.har_head.prototype_memory.n_classes()} classes initialised\n")

    # Pick subject 5 as the "new user"
    new_user = 5
    mask = subjects == new_user
    X_new, y_new = X[mask], y[mask]

    print(f"Step 2: new user (subject {new_user}) arrives — {mask.sum()} windows available")
    print(f"        Adapt with only 100 samples, no retraining...\n")

    X_t = torch.from_numpy(X_new[:100])
    y_t = torch.from_numpy(y_new[:100])

    with torch.no_grad():
        embeddings = model.backbone(X_t)
        model.har_head.prototype_memory.update(embeddings, y_t)

    known = model.har_head.prototype_memory.known_classes()
    print(f"  Prototype memory now covers {len(known)} classes: "
          f"{[UNIFIED_LABELS.get(c, str(c)) for c in known]}\n")

    # Test on held-out windows from this user
    X_test = torch.from_numpy(X_new[100:110])
    y_test = y_new[100:110]

    with torch.no_grad():
        emb   = model.backbone(X_test)
        preds = model.har_head.prototype_memory.predict(emb).numpy()

    correct = 0
    print(f"  {'#':>2}  {'True label':<25} {'Predicted (prototype)':<25} {'✓/✗'}")
    print(f"  {'-'*60}")
    for i, (p, t) in enumerate(zip(preds, y_test)):
        true_name = UNIFIED_LABELS.get(int(t), str(int(t)))
        pred_name = UNIFIED_LABELS.get(int(p), str(int(p)))
        ok = "✓" if p == t else "✗"
        if p == t: correct += 1
        color = GREEN if p == t else YELLOW
        print(f"  {i+1:>2}  {true_name:<25} {color}{pred_name:<25}{RESET} {color}{ok}{RESET}")

    print(f"\n  Accuracy on new user's test windows: {correct}/10")
    print("  (No retraining — only prototype update with 100 samples)")


# ── DEMO 3: Activity Anticipation ─────────────────────────────────────────────
def demo_anticipation(model, X, y):
    banner("DEMO 3 — Activity Anticipation (predict next activity)")
    print("Given 5 partial windows (50% observed), predicting what comes next...\n")

    ant_ds = AnticipationDataset(X, y, obs_ratio=0.50,
                                  seq_len=5, transitions_only=True)

    rng = np.random.default_rng(7)
    idx = rng.choice(len(ant_ds), 8, replace=False)

    print(f"  {'#':>2}  {'Context (last activity)':<25} {'Next (true)':<20} "
          f"{'Predicted':<20} {'✓/✗'}")
    print(f"  {'-'*72}")

    correct = 0
    model.eval()
    for i, wi in enumerate(idx):
        x_seq, y_next = ant_ds[wi]
        x_seq = x_seq.unsqueeze(0)

        with torch.no_grad():
            logits = model.anticipate(x_seq)
            pred   = logits.argmax(dim=-1).item()

        # Current activity = last window in context sequence
        # Approximate by looking at nearby y values
        next_name = UNIFIED_LABELS.get(int(y_next), str(int(y_next)))
        pred_name = UNIFIED_LABELS.get(pred, str(pred))
        ok = "✓" if pred == int(y_next) else "✗"
        if pred == int(y_next): correct += 1

        color = GREEN if pred == int(y_next) else YELLOW
        print(f"  {i+1:>2}  {'(partial signal)':<25} {next_name:<20} "
              f"{color}{pred_name:<20}{RESET} {color}{ok}{RESET}")

    print(f"\n  Accuracy on 8 transition samples: {correct}/8")
    print("  Note: anticipation is hard — model sees only 50% of current window")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}HAR Continual Learning — Interactive Demo{RESET}")
    print("Adaptive and Continual Learning for Human Activity Recognition\n")

    ds, model, X, y, subjects, origins = load()

    demo_recognition(ds, model, X, y)
    demo_continual(ds, model, X, y, subjects)
    demo_anticipation(model, X, y)

    banner("Summary")
    print("  Module 1 (Recognition):   Transformer backbone + nearest-mean classifier")
    print("  Module 1 (Continual):     Prototype memory updated online — no retraining")
    print("  Module 2 (Anticipation):  LSTM predicts next activity from partial signal")
    print()
    print("  Run  python scripts/train.py --help       to retrain")
    print("  Run  python scripts/visualize.py          to regenerate plots")
    print()

if __name__ == "__main__":
    main()
