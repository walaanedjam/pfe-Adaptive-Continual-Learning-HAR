# Checkpoints

Pre-trained model weights saved during training.

## Available checkpoints

| File | Description | Accuracy |
|------|-------------|----------|
| `pretrained.pt` | Backbone pre-trained on HAPT (100 epochs) | 91.3% |
| `continual_user.pt` | UWR model — user-incremental scenario | 79.2% F1 |
| `continual_class.pt` | UWR model — class-incremental scenario | 79.8% F1 |
| `anticipation.pt` | Binary transition detector (K=2, focal loss) | F1=0.262 |
| `backbone_n6.pt` | Backbone with 6 Transformer blocks | 90.7% |

## Loading a checkpoint

```python
from src.models.har_model import HARContinualModel

model = HARContinualModel.load("checkpoints/pretrained.pt", n_classes=12)
```

> Checkpoint files (.pt) are not tracked by git due to size.
> Contact the author to obtain pre-trained weights.
