"""
Hybrid fall detection module.
Combines physical thresholding (acceleration peak) with a learned MLP head.
"""
import numpy as np
import torch
import torch.nn as nn


ACC_THRESHOLD   = 25.0   # m/s²  (~2.5g — typical fall impact)
GYRO_THRESHOLD  = 3.0    # rad/s — rapid angular rotation


def physical_detector(window: np.ndarray) -> int:
    """
    Rule-based fall detector using acceleration magnitude and angular velocity.
    Args:
        window: (T, 6) — [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    Returns:
        1 if fall detected, 0 otherwise
    """
    acc_mag  = np.linalg.norm(window[:, :3], axis=-1)
    gyro_mag = np.linalg.norm(window[:, 3:], axis=-1)
    return int(acc_mag.max() > ACC_THRESHOLD and gyro_mag.max() > GYRO_THRESHOLD)


class FallDetectionHead(nn.Module):
    """Lightweight MLP on backbone embeddings for fall detection."""

    def __init__(self, d_model: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)


class HybridFallDetector:
    """
    Fusion of physical threshold detector and learned MLP head (OR rule).
    Decision = physical OR (MLP probability > threshold).
    """

    def __init__(self, backbone: nn.Module, threshold: float = 0.5):
        self.backbone = backbone
        self.head = FallDetectionHead(backbone.d_model)
        self.threshold = threshold

    def predict(self, window: np.ndarray, device: torch.device = None) -> int:
        """
        Args:
            window: (T, 6) numpy array
        Returns:
            1 (fall) or 0 (normal)
        """
        if device is None:
            device = torch.device('cpu')

        phys = physical_detector(window)
        if phys:
            return 1

        self.backbone.eval()
        self.head.eval()
        with torch.no_grad():
            x = torch.from_numpy(window[None]).float().to(device)
            emb = self.backbone(x)
            prob = torch.sigmoid(self.head(emb)).item()
        return int(prob > self.threshold)

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.head = self.head.to(device)
        return self
