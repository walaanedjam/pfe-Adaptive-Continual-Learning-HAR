"""
CORAL domain adaptation for cross-dataset HAR.
Aligns second-order statistics (covariance) between source and target embeddings.
Reference: Sun & Saenko (2016) — Deep CORAL.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute CORAL loss between source and target feature distributions.
    Args:
        source: (Ns, d) source embeddings
        target: (Nt, d) target embeddings
    Returns:
        scalar CORAL loss
    """
    d = source.size(1)
    source = source - source.mean(dim=0)
    target = target - target.mean(dim=0)
    Cs = (source.T @ source) / (source.size(0) - 1)
    Ct = (target.T @ target) / (target.size(0) - 1)
    loss = torch.norm(Cs - Ct, p='fro') ** 2 / (4 * d * d)
    return loss


class CORALAdapter:
    """
    Adapts a pre-trained backbone from source to target domain using CORAL.

    Modes:
      - unsupervised: only covariance alignment, no target labels
      - supervised:   covariance alignment + cross-entropy on labeled target samples
    """

    def __init__(self, backbone: nn.Module, n_classes: int,
                 lr_backbone: float = 1e-5, lr_head: float = 1e-3,
                 lambda_coral: float = 1.0):
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, n_classes)
        self.lambda_coral = lambda_coral
        self.optimizer = torch.optim.Adam([
            {'params': backbone.parameters(), 'lr': lr_backbone},
            {'params': self.head.parameters(), 'lr': lr_head},
        ])

    def fit(self, X_source: np.ndarray, y_source: np.ndarray,
            X_target: np.ndarray, y_target: np.ndarray = None,
            n_epochs: int = 20, batch_size: int = 256,
            device: torch.device = None) -> dict:
        """
        Adapt backbone to target domain.
        If y_target is None: unsupervised CORAL only.
        If y_target provided: supervised CORAL (uses labeled subset).
        """
        if device is None:
            device = torch.device('cpu')
        self.backbone = self.backbone.to(device)
        self.head = self.head.to(device)

        supervised = y_target is not None
        results = {'mode': 'supervised' if supervised else 'unsupervised',
                   'history': []}

        N = min(len(X_source), len(X_target))
        for epoch in range(1, n_epochs + 1):
            self.backbone.train()
            self.head.train()
            perm = np.random.permutation(N)
            total_loss = 0.0

            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                xs = torch.from_numpy(X_source[idx]).to(device)
                xt = torch.from_numpy(X_target[idx % len(X_target)]).to(device)

                emb_s = self.backbone(xs)
                emb_t = self.backbone(xt)
                loss = self.lambda_coral * coral_loss(emb_s, emb_t)

                if supervised:
                    ys = torch.from_numpy(y_target[idx % len(y_target)]).long().to(device)
                    logits = self.head(emb_t)
                    loss += nn.functional.cross_entropy(logits, ys)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            results['history'].append(total_loss / max(1, N // batch_size))

        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 device: torch.device = None, batch_size: int = 256) -> float:
        """Evaluate macro-F1 on target data."""
        if device is None:
            device = torch.device('cpu')
        self.backbone.eval()
        self.head.eval()
        preds, labels = [], []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i + batch_size]).to(device)
                emb = self.backbone(xb)
                pred = self.head(emb).argmax(dim=-1).cpu().numpy()
                preds.extend(pred.tolist())
                labels.extend(y[i:i + batch_size].tolist())
        return f1_score(labels, preds, average='macro', zero_division=0)
