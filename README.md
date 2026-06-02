# Adaptive and Continual Learning for Human Activity Recognition

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Final Year Engineering Project (PFE) — École Nationale Supérieure d'Informatique (ESI), 2025/2026**

> Supervisor: Pr. Attal Ferhat (LISSI, UPEC Paris) &nbsp;|&nbsp; Co-supervisor: Dr. Meziani Lila (ESI)

---

## Overview

This project proposes a deep learning system for **Human Activity Recognition (HAR)** using inertial sensors (IMU) that goes beyond static classification. The system is designed to:

- **Learn continuously** — adapt to new users and activities over time without forgetting previous knowledge (catastrophic forgetting)
- **Anticipate transitions** — detect imminent activity changes from partial observations, enabling proactive applications (fall prevention, assistive systems)
- **Adapt across domains** — transfer knowledge between different sensor placements and datasets with minimal labeled data

---

## Architecture

```
IMU Signal  (B × 150 × 6)   50 Hz — acc_xyz + gyro_xyz
        │
        ▼
  ┌─────────────────────────────────┐
  │     IMUTransformerEncoder       │
  │  4 blocks · 4 heads · d=128     │
  │  CLS token pooling · 531K params│
  └──────────────┬──────────────────┘
                 │  embedding (B × 128)
        ┌────────┴────────┐
        ▼                 ▼
  ┌───────────┐    ┌──────────────────┐
  │  Module 1 │    │    Module 2       │
  │  HAR Head │    │  Anticipation    │
  │           │    │  Head            │
  │ Prototype │    │  Binary K=2      │
  │ Memory    │    │  Focal Loss      │
  │ UWR Buffer│    │  F1 = 0.262      │
  │ MC Dropout│    └──────────────────┘
  └───────────┘
```

**Key innovations:**
- **Uncertainty-Weighted Replay (UWR)** — prioritizes hard examples in the replay buffer using Monte Carlo Dropout uncertainty estimates
- **Binary transition detection** — reformulates activity anticipation as a binary classification problem (K=2 horizon), outperforming multi-class approach by +68%
- **CORAL domain adaptation** — covariance alignment between source and target embeddings, achieving ×15 F1 gain with only 20% target labels

---

## Results

### HAR Classification

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| n=4 blocks (no augmentation) | 81.0% | — |
| **n=4 blocks + augmentation** | **91.3%** | **0.752** |
| n=6 blocks + augmentation | 90.7% | 0.716 |

### Continual Learning — User-Incremental Scenario

| Method | Final Accuracy | Forgetting |
|--------|---------------|------------|
| Naive finetuning | 62.4% | 0.241 |
| EWC (Kirkpatrick et al., 2017) | 71.3% | 0.158 |
| iCaRL (Rebuffi et al., 2017) | 75.8% | 0.112 |
| **Ours (UWR)** | **79.2%** | **0.087** |

### Activity Anticipation

| Formulation | F1 (Transition) | Recall |
|-------------|----------------|--------|
| Multi-class (12 classes) | 0.093 | — |
| Binary K=1, BCE | 0.156 | 0.67 |
| **Binary K=2 + Focal Loss (ours)** | **0.262** | **0.55** |

### Domain Adaptation — HAPT → WISDM

| Method | Macro F1 |
|--------|----------|
| No adaptation | 0.040 |
| CORAL unsupervised | 0.040 |
| **CORAL supervised (20% labels)** | **0.629** |

---

## Project Structure

```
har_project/
├── src/
│   ├── data/
│   │   ├── preprocessing.py       # Butterworth filter, resampling, sliding windows
│   │   ├── augmentation.py        # Noise, scaling, time-warp, channel dropout
│   │   ├── dataset_loaders.py     # HAPT, WISDM loaders
│   │   ├── har_dataset.py         # PyTorch Dataset + incremental task builder
│   │   └── anticipation_dataset.py
│   ├── models/
│   │   ├── backbone.py            # IMUTransformerEncoder (CLS token, pre-norm)
│   │   ├── har_model.py           # HARContinualModel — full two-headed system
│   │   ├── prototype_memory.py    # Online prototype memory + contrastive loss
│   │   ├── replay_buffer.py       # Standard experience replay
│   │   ├── uncertainty_replay.py  # Uncertainty-Weighted Replay (UWR)
│   │   ├── mc_dropout.py          # Monte Carlo Dropout uncertainty estimation
│   │   ├── domain_adapter.py      # CORAL domain adaptation
│   │   └── fall_detector.py       # Hybrid fall detection (threshold + MLP)
│   ├── training/
│   │   ├── trainer.py             # Pre-training loop
│   │   ├── anticipation_trainer.py
│   │   ├── ewc_trainer.py         # EWC baseline
│   │   └── icarl_trainer.py       # iCaRL baseline
│   └── evaluation/
│       └── metrics.py             # F1, BWT, FWT, Forgetting
├── scripts/
│   ├── train.py                   # Main training entry point
│   ├── train_transition_detection.py
│   ├── train_k2_anticipation.py   # K=2 + focal loss anticipation
│   ├── train_backbone_n6.py       # Architecture ablation
│   ├── train_fall_detection.py
│   ├── eval_domain_adapter.py     # CORAL evaluation
│   ├── cross_dataset_eval.py
│   ├── ablation.py
│   └── visualize_embeddings.py    # t-SNE visualization
├── demo.py                        # Interactive demo
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/walaanedjam/har-project.git
cd har-project
pip install -r requirements.txt
```

### Datasets

Download and place under `data/raw/`:

| Dataset | Link |
|---------|------|
| HAPT (UCI) | https://archive.ics.uci.edu/dataset/341 |
| WISDM | https://www.cis.fordham.edu/wisdm/dataset.php |

```bash
# Preprocess all datasets
python scripts/preprocess.py --data_root data/raw --out data/processed
```

---

## Usage

```bash
# 1. Pre-train the backbone
python scripts/train.py --mode pretrain --epochs 100

# 2. Continual learning — user-incremental
python scripts/train.py --mode continual --scenario user

# 3. Continual learning — class-incremental
python scripts/train.py --mode continual --scenario class

# 4. Train anticipation module (K=2, focal loss)
python scripts/train_k2_anticipation.py

# 5. Domain adaptation CORAL
python scripts/eval_domain_adapter.py

# 6. Interactive demo
python demo.py
```

---

## Dependencies

```
torch >= 2.0
numpy >= 1.24
scipy >= 1.10
scikit-learn >= 1.2
matplotlib >= 3.7
```

---

## References

- Amrani et al. (2025) — *Leveraging Dataset Integration and CL for HAR* — IJMLC
- Schiemer et al. (2023) — *Online Continual Learning for HAR* — PMC
- Adaimi & Thomaz (2022) — *Lifelong Adaptive ML for HAR* — Sensors
- Liu et al. (2024) — *iKAN: Global Incremental Learning with KAN for HAR* — ISWC
- Kirkpatrick et al. (2017) — *Overcoming Catastrophic Forgetting* — PNAS
- Rebuffi et al. (2017) — *iCaRL: Incremental Classifier and Representation Learning* — CVPR
- Sun & Saenko (2016) — *Deep CORAL: Correlation Alignment* — ECCVW
- Dirgová et al. (2022) — *Wearable Sensor-Based HAR with Transformer* — Sensors

---

## License

MIT License — see [LICENSE](LICENSE) for details.
